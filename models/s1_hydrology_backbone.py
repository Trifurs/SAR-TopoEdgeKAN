"""SAR-first hydrology backbone for flood-depth estimation.

Temporal states, change evidence, SAR detail, and hydrologic terrain proxies
remain separate until each scale is validity-gated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import EfficientPyramidBranch
from models.encoders import ConvNormAct, group_count


def _pool_fraction(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(value, size)


class JointSARHydrologyEncoder(nn.Module):
    """Joint Temporal-Change SAR Encoder (TCSE).

    Learn joint spatial filters from temporal states, signed/absolute change,
    supplied known flood range and explicit branch-validity indicators.
    """

    def __init__(
        self,
        state_channels: int,
        change_channels: int,
        channels: Sequence[int],
        *,
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "spatial",
        known_flood_range_enabled: bool = False,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("JointSARHydrologyEncoder requires four positive scales")
        if min(state_channels, change_channels) <= 0:
            raise ValueError("S1 state and change channel counts must be positive")
        self.widths = widths
        self.state_channels = int(state_channels)
        self.change_channels = int(change_channels)
        self.known_flood_range_enabled = bool(known_flood_range_enabled)
        # pre + event + external change + signed/absolute internal change,
        # optional supplied flood range, and four explicit availability maps.
        input_channels = (
            4 * self.state_channels
            + self.change_channels
            + int(self.known_flood_range_enabled)
            + 4
        )
        self.pyramid = EfficientPyramidBranch(
            input_channels, widths, dropout, groups, block_kind
        )
        self.change_projection = nn.ModuleList(
            [
                ConvNormAct(self.change_channels, width, 1, groups=groups)
                for width in widths
            ]
        )
        # Change quality depends only on learned SAR evidence and explicit
        # branch-validity fractions.
        self.quality_gate = nn.ModuleList(
            [nn.Conv2d(width + 3, 1, 1) for width in widths]
        )

    def forward(
        self,
        pre: torch.Tensor,
        event: torch.Tensor,
        change: torch.Tensor,
        valid: torch.Tensor,
        flood_range: torch.Tensor | None = None,
        branch_validity: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        branch_validity = branch_validity or {}
        pre_valid = branch_validity.get("s1_t1", branch_validity.get("t1", valid))
        event_valid = branch_validity.get("s1_t2", branch_validity.get("t2", valid))
        change_valid = branch_validity.get(
            "s1_change", branch_validity.get("change", valid)
        )
        pair_valid = torch.minimum(pre_valid, event_valid)
        internal_change = (event - pre) * pair_valid
        if self.known_flood_range_enabled:
            if flood_range is None:
                raise KeyError("Supplied flood range was configured but absent")
            if flood_range.shape != event_valid.shape:
                raise ValueError("Supplied flood range must match the single-channel validity map")
            range_features = flood_range * event_valid
        else:
            range_features = pre.new_zeros(pre.shape[0], 0, *pre.shape[-2:])
        joint = torch.cat(
            (
                pre * pre_valid,
                event * event_valid,
                change * change_valid,
                internal_change,
                internal_change.abs(),
                range_features,
                pre_valid,
                event_valid,
                change_valid,
                pair_valid,
            ),
            dim=1,
        )
        features = self.pyramid(joint)

        outputs: list[torch.Tensor] = []
        quality_gates: list[torch.Tensor] = []
        change_evidence: list[torch.Tensor] = []
        pair_fractions: list[torch.Tensor] = []
        pre_fractions: list[torch.Tensor] = []
        for index, feature in enumerate(features):
            size = feature.shape[-2:]
            pre_fraction = _pool_fraction(pre_valid, size).to(feature.dtype)
            event_fraction = _pool_fraction(event_valid, size).to(feature.dtype)
            change_fraction = _pool_fraction(change_valid, size).to(feature.dtype)
            pair_fraction = _pool_fraction(pair_valid, size).to(feature.dtype)
            projected_change = self.change_projection[index](
                F.interpolate(
                    change * change_valid,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            ) * change_fraction
            quality_logit = self.quality_gate[index](
                torch.cat(
                    (
                        feature,
                        event_fraction,
                        change_fraction,
                        pair_fraction,
                    ),
                    dim=1,
                )
            )
            quality = torch.sigmoid(quality_logit) * event_fraction
            output = (
                feature
                + 0.10 * quality * projected_change
            ) * event_fraction
            outputs.append(output)
            quality_gates.append(quality)
            change_evidence.append(projected_change)
            pair_fractions.append(pair_fraction)
            pre_fractions.append(pre_fraction)

        diagnostics: dict[str, Any] = {
            "quality_gates": quality_gates,
            "change_evidence": change_evidence,
            "pair_valid_fraction_mean": torch.stack(
                [value.mean() for value in pair_fractions]
            ).mean(),
            "pre_context_gate_mean": torch.stack(
                [value.mean() for value in pre_fractions]
            ).mean(),
            "quality_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
        }
        return outputs, diagnostics


class HydrologyContext(nn.Module):
    """Multi-dilation context aggregator with a bounded residual update."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        dropout: float = 0.05,
        residual_init: float = 0.15,
        residual_max: float = 0.50,
    ) -> None:
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                          groups=channels, bias=False),
                nn.GroupNorm(group_count(channels, groups), channels),
                nn.SiLU(inplace=True),
            )
            for dilation in (1, 2, 4, 8)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.residual_max = float(residual_max)
        if self.residual_max <= 0.0 or abs(residual_init) > self.residual_max:
            raise ValueError("invalid signed context residual bounds")
        self.gamma = nn.Parameter(torch.tensor(float(residual_init)))
        self.global_context = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(16, channels // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(16, channels // 4), channels, 1),
            )
        )
        final = self.global_context[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def residual(self) -> torch.Tensor:
        return self.gamma.clamp(-self.residual_max, self.residual_max)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs + self.residual * self.project(
            torch.cat([path(inputs) for path in self.paths], dim=1)
        )
        output = output + self.global_context(inputs)
        return output


class S1FeatureAlignment(nn.Module):
    """Lightweight per-scale projection of the SAR pyramid.

    Terrain enters through its own pyramid, decoder skips, hydrostatic prior
    and topographic graph. This block projects only the SAR feature pyramid.
    """

    def __init__(self, channels: Sequence[int], *, groups: int = 8) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("S1FeatureAlignment requires positive scale widths")
        self.projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )

    def forward(
        self,
        sar: Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if len(sar) != len(self.projection):
            raise ValueError("alignment requires one SAR feature per scale")
        outputs: list[torch.Tensor] = []
        sar_gates: list[torch.Tensor] = []
        for projector, sar_value in zip(self.projection, sar):
            aligned = projector(sar_value)
            sensor_fraction = F.adaptive_avg_pool2d(
                sensor_valid, aligned.shape[-2:]
            ).to(aligned.dtype)
            outputs.append(aligned)
            sar_gates.append(sensor_fraction)
        diagnostics = {
            "sar_gates": sar_gates,
            "sar_gate_mean": torch.stack(
                [value.mean() for value in sar_gates]
            ).mean(),
        }
        return outputs, diagnostics
