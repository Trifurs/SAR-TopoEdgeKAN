"""Small-batch efficient convolutional blocks used by the production model."""

from __future__ import annotations

import torch
from torch import nn

from models.encoders import ConvNormAct, group_count


def residual_path_scale(
    reference: torch.Tensor,
    drop_probability: float,
    training: bool,
) -> torch.Tensor:
    """Return one inverted-drop-path scale per sample.

    Sharing this scale across a named module's resolutions trains the main
    network against that module's exact ablation while preserving the expected
    residual magnitude.
    """

    probability = float(drop_probability)
    if not 0.0 <= probability < 1.0:
        raise ValueError("drop-path probability must lie in [0, 1)")
    shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    if not training or probability == 0.0:
        return reference.new_ones(shape)
    keep_probability = 1.0 - probability
    keep = torch.rand(shape, device=reference.device) < keep_probability
    return keep.to(reference.dtype) / keep_probability


class EfficientResidualBlock(nn.Module):
    """Depthwise residual block with channel attention."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        groups: int = 8,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(group_count(hidden, groups), hidden),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        squeeze = max(4, channels // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze, channels, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(group_count(channels, groups), channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.block(inputs)
        update = update * self.channel_gate(update)
        return self.activation(inputs + self.norm(update))


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual with batch-size-stable normalization.

    Spatial filtering is performed in the expanded representation, followed by
    a linear pointwise projection.  Unlike :class:`EfficientResidualBlock`, the
    block has no post-addition activation; this keeps the skip path an exact
    identity and avoids repeatedly clipping signed regression features.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        groups: int = 8,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0 or expansion <= 0:
            raise ValueError("inverted residual channels and expansion must be positive")
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(group_count(hidden, groups), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(group_count(hidden, groups), hidden),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
        )
        # Per-channel layer scale provides a cheap, learnable residual budget.
        # A conservative non-zero start lets every convolution receive gradients
        # immediately while the information path remains close to identity.
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layer_scale * self.block(inputs)


class SpatialResidualBlock(nn.Module):
    """Full-convolution residual block for detail-sensitive depth regression.

    The efficient block above is deliberately inexpensive, but its only spatial
    operator is depthwise and therefore cannot learn cross-channel spatial
    patterns in one step.  Flood-depth regression depends on joint SAR and
    terrain neighbourhoods, so this optional block uses two ordinary 3x3
    convolutions while retaining the batch-size-stable GroupNorm contract.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


def residual_block(kind: str, channels: int, dropout: float, groups: int) -> nn.Module:
    """Construct a configured residual block."""

    if kind == "efficient":
        return EfficientResidualBlock(channels, dropout, groups)
    if kind == "inverted":
        return InvertedResidualBlock(channels, dropout, groups)
    if kind == "spatial":
        return SpatialResidualBlock(channels, dropout, groups)
    raise ValueError(
        f"Unsupported residual block {kind!r}; expected 'efficient', 'inverted', "
        "or 'spatial'"
    )


class EfficientPyramidBranch(nn.Module):
    """Four-scale convolutional feature pyramid."""

    def __init__(
        self,
        inputs: int,
        channels: list[int],
        dropout: float,
        groups: int,
        block_kind: str,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(inputs, channels[0], 3, groups=groups),
            residual_block(block_kind, channels[0], dropout, groups),
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels[index - 1], channels[index], 3, 2, groups),
                    residual_block(block_kind, channels[index], dropout, groups),
                )
                for index in range(1, len(channels))
            ]
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        result = [self.stem(inputs)]
        for layer in self.down:
            result.append(layer(result[-1]))
        return result
