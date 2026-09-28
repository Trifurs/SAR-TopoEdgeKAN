"""Shared, range-conditioned building blocks for named comparison networks.

The helpers in this module are deliberately private.  Each public comparison
architecture has its own module and configuration; this file only centralizes
well-tested tensor assembly and repeated convolutional primitives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


COMPARISON_INPUT_SCHEMAS = {
    "dlsim_change_dsm_range": 5,
    "s1_terrain_range": 10,
}


def comparison_input_channels(schema: str) -> int:
    """Return the audited channel count for one comparison input schema."""

    try:
        return COMPARISON_INPUT_SCHEMAS[str(schema)]
    except KeyError as exc:
        supported = ", ".join(sorted(COMPARISON_INPUT_SCHEMAS))
        raise ValueError(f"Unsupported comparison input schema {schema!r}; expected {supported}") from exc


def prepare_comparison_tensor(batch: Mapping[str, Any], schema: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble inputs and the direct ``valid_depth_mask`` flood-range channel.

    ``valid_depth_mask`` is an explicit experimental input for these comparison
    methods, not a predicted range and not an input to SAR-TopoEdgeKAN.  It is
    returned separately as well so every architecture can constrain its final
    depth output to exactly the supplied range.
    """

    try:
        flood_range = batch["masks"]["valid_depth_mask"].float()
        s1_t1 = batch["s1_t1"].float()
        s1_t2 = batch["s1_t2"].float()
        s1_change = batch["s1_change"].float()
        terrain = batch["terrain"].float()
    except (KeyError, TypeError) as exc:
        raise KeyError("Comparison networks require S1, terrain, and valid_depth_mask") from exc
    if flood_range.ndim != 4 or flood_range.shape[1] != 1:
        raise ValueError("valid_depth_mask must have shape [batch, 1, height, width]")
    if schema == "dlsim_change_dsm_range":
        # DLSIM starts from a change product and a DEM.  The three continuous
        # SAR-change bands replace its simulated binary change map; elevation is
        # the first configured terrain band (DSM), and the requested range is
        # appended as a fifth channel.
        tensor = torch.cat((s1_change, terrain[:, :1], flood_range), dim=1)
    elif schema == "s1_terrain_range":
        tensor = torch.cat((s1_t1, s1_t2, s1_change, terrain, flood_range), dim=1)
    else:
        comparison_input_channels(schema)
        raise AssertionError("unreachable comparison input schema")
    expected = comparison_input_channels(schema)
    if tensor.shape[1] != expected:
        raise ValueError(
            f"Input schema {schema!r} produced {tensor.shape[1]} channels; expected {expected}"
        )
    return tensor, flood_range


def _groups(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """A batch-size-stable convolution, GroupNorm, and SiLU sequence."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.second = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.second(self.first(value)) + self.skip(value))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False) -> None:
        super().__init__()
        block = ResidualBlock if residual else DoubleConv
        self.pool = nn.MaxPool2d(2)
        self.block = block(in_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(value))


def resize_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != reference.shape[-2:]:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return value


class UpBlock(nn.Module):
    """Interpolation decoder block with an explicit skip-channel contract."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = ConvNormAct(in_channels, out_channels)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = resize_like(value, skip)
        return self.fuse(torch.cat((self.reduce(value), skip), dim=1))


class LinkDecoderBlock(nn.Module):
    """Additive decoder block used by the LinkNet comparator."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.refine = ResidualBlock(out_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = resize_like(value, skip)
        return self.refine(self.reduce(value) + self.skip(skip))


class AttentionGate(nn.Module):
    """Gated skip attention in the Attention U-Net formulation."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        intermediate = max(1, min(skip_channels, gate_channels) // 2)
        self.skip_projection = nn.Conv2d(skip_channels, intermediate, 1, bias=False)
        self.gate_projection = nn.Conv2d(gate_channels, intermediate, 1, bias=False)
        self.coefficient = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(intermediate, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = resize_like(gate, skip)
        return skip * self.coefficient(self.skip_projection(skip) + self.gate_projection(gate))


class RangeConditionedHead(nn.Module):
    """Map spatial features to non-negative depth within a supplied flood range."""

    def __init__(self, in_channels: int, *, epsilon: float, uncertainty_scale_m: float) -> None:
        super().__init__()
        if epsilon <= 0.0 or uncertainty_scale_m <= 0.0:
            raise ValueError("epsilon and uncertainty_scale_m must be positive")
        self.depth = nn.Conv2d(in_channels, 1, 1)
        self.epsilon = float(epsilon)
        self.uncertainty_scale_m = float(uncertainty_scale_m)

    def forward(self, features: torch.Tensor, flood_range: torch.Tensor) -> dict[str, torch.Tensor]:
        conditional = F.softplus(self.depth(features)) + self.epsilon
        flood_range = (flood_range > 0.5).to(dtype=conditional.dtype)
        depth = conditional * flood_range
        scale = torch.full_like(depth, self.uncertainty_scale_m)
        return {
            "depth": depth,
            "conditional_depth": conditional,
            "positive_depth": conditional,
            "expected_depth": depth,
            "uncertainty_scale": scale,
        }


def channels_from_config(model_config: Mapping[str, Any], expected: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in model_config["channels"])
    if len(values) != expected or any(value <= 0 for value in values):
        raise ValueError(f"model.channels must contain {expected} positive values")
    return values


def head_from_config(in_channels: int, model_config: Mapping[str, Any]) -> RangeConditionedHead:
    return RangeConditionedHead(
        in_channels,
        epsilon=float(model_config.get("depth_epsilon", 0.001)),
        uncertainty_scale_m=float(model_config.get("uncertainty_scale_m", 0.35)),
    )


def validate_input_schema(model_config: Mapping[str, Any]) -> tuple[str, int]:
    schema = str(model_config["input_schema"])
    channels = comparison_input_channels(schema)
    configured = int(model_config.get("input_channels", channels))
    if configured != channels:
        raise ValueError(
            f"model.input_channels={configured} disagrees with {schema!r} ({channels})"
        )
    if str(model_config.get("flood_support", "valid_depth_mask")) != "valid_depth_mask":
        raise ValueError("Comparison networks support only direct valid_depth_mask flood support")
    return schema, channels
