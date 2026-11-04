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

from models.efficient_blocks import EfficientPyramidBranch, residual_block
from models.encoders import ConvNormAct, group_count


def _pool_fraction(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(value, size)


def masked_two_way_change_mixer(
    internal_change: torch.Tensor,
    external_change: torch.Tensor,
    internal_valid_fraction: torch.Tensor,
    external_valid_fraction: torch.Tensor,
    mixer_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix only available change evidence without an ``event - 0`` fallback.

    The learned logit is anti-symmetrically paired to keep the production checkpoint
    parameterization intact.  Availability is applied before softmax, so a sole
    valid branch receives an exact weight of one and an all-invalid pixel stays
    finite and exactly zero.
    """

    if internal_change.shape != external_change.shape or internal_change.shape != mixer_logit.shape:
        raise ValueError("internal, external, and mixer-logit tensors must share BCHW shape")
    internal_active = internal_valid_fraction > 0.0
    external_active = external_valid_fraction > 0.0
    internal_active, external_active = torch.broadcast_tensors(
        internal_active, external_active
    )
    availability = torch.stack((internal_active, external_active), dim=1)
    if availability.shape[2] != 1:
        raise ValueError("change-valid fractions must have one channel")
    availability = availability.expand(-1, -1, internal_change.shape[1], -1, -1)
    logits = torch.stack((mixer_logit, -mixer_logit), dim=1)
    masked_logits = logits.masked_fill(~availability, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=1) * availability.to(logits.dtype)
    internal_weight = weights[:, 0]
    external_weight = weights[:, 1]
    mixed = internal_weight * internal_change + external_weight * external_change
    return mixed, internal_weight, external_weight


class JointSARHydrologyEncoder(nn.Module):
    """Joint Temporal-Change SAR Encoder (TCSE).

    The original decomposed TCSE evaluates three full feature pyramids for the
    two temporal states, the supplied change product, and a fourth lightweight
    detail representation.  That separation is useful when acquisitions are
    frequently absent, but it is unnecessarily expensive for the audited
    FloodDepthNet release where complete event observations are required.  This
    encoder exposes the same downstream contract while learning joint spatial
    filters from the states, signed/absolute temporal change, supplied known
    flood range, and branch-validity indicators in one pyramid.

    Acquisition-quality metadata is deliberately excluded: repeated matched
    ablations found that its former reliability-conditioning path did not
    improve the final architecture. Branch-validity maps remain explicit and
    are sufficient to prevent unavailable observations from leaking into TCSE.
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

        zero = outputs[0].sum() * 0.0
        diagnostics: dict[str, Any] = {
            "change_gates": quality_gates,
            "quality_gates": quality_gates,
            "detail_gates": quality_gates,
            "change_evidence": change_evidence,
            "internal_change_weights": [],
            "external_change_weights": [],
            "internal_change": [internal_change],
            "pair_valid_fractions": pair_fractions,
            "pre_context_gates": pre_fractions,
            "internal_weight_mean": zero,
            "external_weight_mean": zero,
            "pair_valid_fraction_mean": torch.stack(
                [value.mean() for value in pair_fractions]
            ).mean(),
            "pre_context_gate_mean": torch.stack(
                [value.mean() for value in pre_fractions]
            ).mean(),
            "change_gate_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
            "quality_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
            "detail_gate_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
        }
        return outputs, diagnostics


class SARHydrologyEncoder(nn.Module):
    """Decomposed Temporal-Change SAR Encoder (TCSE)."""

    def __init__(
        self,
        state_channels: int,
        change_channels: int,
        channels: Sequence[int],
        *,
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        known_flood_range_enabled: bool = False,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("SARHydrologyEncoder requires four positive scales")
        if min(state_channels, change_channels) <= 0:
            raise ValueError("S1 state and change channel counts must be positive")
        self.widths = widths
        detail_widths = [max(16, min(96, width // 2)) for width in widths]
        # ``temporal`` is called once for pre and once for event. It must be
        # deterministic in train mode so equal acquisitions cannot manufacture a
        # random internal temporal difference. Regularization remains in the
        # post-fusion/refinement paths below.
        self.temporal = EfficientPyramidBranch(
            int(state_channels), widths, 0.0, groups, block_kind
        )
        self.change = EfficientPyramidBranch(
            int(change_channels), widths, dropout, groups, block_kind
        )
        # Detail sees signed and absolute change at the input resolution.  This
        # retains local SAR scattering edges that a deep state branch can smooth.
        self.detail = EfficientPyramidBranch(
            2 * int(state_channels) + int(change_channels) + 2 * int(state_channels),
            detail_widths,
            dropout,
            groups,
            block_kind,
        )
        self.detail_projection = nn.ModuleList(
            [ConvNormAct(detail_width, width, 1, groups=groups)
             for detail_width, width in zip(detail_widths, widths)]
        )
        self.state_mix = nn.ModuleList(
            [ConvNormAct(2 * width + 2, width, 1, groups=groups) for width in widths]
        )
        self.change_mix = nn.ModuleList(
            [ConvNormAct(3 * width, width, 1, groups=groups) for width in widths]
        )
        # State, internal/external change, detail, and three validity-derived
        # scalars. No acquisition metadata enters the learned SAR path.
        self.change_gate = nn.ModuleList(
            [nn.Conv2d(4 * width + 3, width, 1) for width in widths]
        )
        self.qa_gate = nn.ModuleList(
            [nn.Conv2d(2, 1, 1) for _ in widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, width, dropout, groups) for width in widths]
        )
        self.known_flood_range_enabled = bool(known_flood_range_enabled)
        # Preserve the established parameter names for the optional known-range
        # modulation; this projection consumes only the supplied binary range.
        self.conditioning = (
            nn.ModuleList(
                [nn.Conv2d(1, 4 * width, 1) for width in widths]
            )
            if self.known_flood_range_enabled
            else None
        )
        if self.conditioning is not None:
            for projection in self.conditioning:
                # Known-range modulation starts as an identity path.
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

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
        change_valid = branch_validity.get("s1_change", branch_validity.get("change", valid))
        if self.known_flood_range_enabled:
            if flood_range is None:
                raise KeyError("Supplied flood range was configured but absent")
            if flood_range.shape != event_valid.shape:
                raise ValueError("Supplied flood range must match the single-channel validity map")

        pair_valid = torch.minimum(pre_valid, event_valid)
        pre_features = self.temporal(pre * pre_valid)
        event_features = self.temporal(event * event_valid)
        change_features = self.change(change * change_valid)
        # The detail path must obey the same branch-validity contract as the
        # pyramid paths.  Otherwise a missing pre/event observation can leak
        # arbitrary fill values into the highest-resolution SAR evidence.
        masked_pre = pre * pair_valid
        masked_event = event * pair_valid
        masked_change = change * change_valid
        input_difference = (event - pre) * pair_valid
        detail_features = self.detail(
            torch.cat(
                (masked_pre, masked_event, masked_change,
                 input_difference, input_difference.abs()),
                dim=1,
            )
        )

        outputs: list[torch.Tensor] = []
        diagnostics: dict[str, Any] = {
            "change_gates": [],
            "quality_gates": [],
            "detail_gates": [],
            "range_film_amplitude": [],
            "change_evidence": [],
            "internal_change_weights": [],
            "external_change_weights": [],
            "internal_change": [],
            "pair_valid_fractions": [],
            "pre_context_gates": [],
        }
        for index, (before, after, changed, detail) in enumerate(
            zip(pre_features, event_features, change_features, detail_features)
        ):
            size = before.shape[-2:]
            pre_fraction = _pool_fraction(pre_valid, size)
            event_fraction = _pool_fraction(event_valid, size)
            change_fraction = _pool_fraction(change_valid, size)
            pair_fraction = _pool_fraction(pair_valid, size)
            if self.conditioning is not None:
                assert flood_range is not None
                range_features = F.interpolate(flood_range, size, mode="bilinear", align_corners=False)
                gamma_before, beta_before, gamma_after, beta_after = self.conditioning[index](range_features).chunk(4, dim=1)
                # A shared FiLM residual avoids turning identical pre/event
                # states into a synthetic temporal difference.
                gamma = 0.5 * (gamma_before + gamma_after)
                beta = 0.5 * (beta_before + beta_after)
                before = before * (1.0 + 0.10 * torch.tanh(gamma)) + 0.05 * torch.tanh(beta)
                after = after * (1.0 + 0.10 * torch.tanh(gamma)) + 0.05 * torch.tanh(beta)
                diagnostics["range_film_amplitude"].append(
                    torch.cat((gamma_before, beta_before, gamma_after, beta_after), dim=1).abs().mean()
                )
            else:
                diagnostics["range_film_amplitude"].append(before.sum() * 0.0)

            # The event state is the always-present main path.  Pre-event
            # information can only enter through a fraction-gated context term.
            event_state = after * event_fraction
            pre_context_state = self.state_mix[index](
                torch.cat((before * pre_fraction, event_state, pre_fraction, event_fraction), dim=1)
            )
            pre_context = pre_context_state * pre_fraction
            # Internal differences are legal only where both temporal branches are
            # present; external change stays independently validity-gated.
            internal_change = (after - before) * pair_fraction
            external_change = changed * change_fraction
            detail = self.detail_projection[index](detail) * torch.maximum(
                pair_fraction, change_fraction
            )
            change_descriptor = self.change_mix[index](
                torch.cat(
                    (internal_change, internal_change.abs(), external_change), dim=1
                )
            ) + detail
            quality = torch.sigmoid(
                self.qa_gate[index](
                    torch.cat((event_fraction, change_fraction), dim=1)
                )
            ) * event_fraction
            gate_parts = [
                event_state, internal_change, external_change, change_descriptor,
            ]
            gate_parts.extend((event_fraction, change_fraction, quality))
            mixer_logit = self.change_gate[index](torch.cat(gate_parts, dim=1))
            mixed_change, internal_weight, external_weight = masked_two_way_change_mixer(
                internal_change,
                external_change,
                pair_fraction,
                change_fraction,
                mixer_logit,
            )
            # No valid internal/external evidence produces an exact zero residual;
            # event availability remains the only requirement for an output state.
            output = self.refine[index](
                event_state + pre_context + quality * mixed_change
            ) * event_fraction
            outputs.append(output)
            diagnostics["change_gates"].append(quality)
            diagnostics["detail_gates"].append(
                torch.maximum(pair_fraction, change_fraction) * detail.abs().mean(dim=1, keepdim=True)
            )
            diagnostics["quality_gates"].append(quality)
            diagnostics["change_evidence"].append(mixed_change)
            diagnostics["internal_change_weights"].append(internal_weight)
            diagnostics["external_change_weights"].append(external_weight)
            diagnostics["internal_change"].append(internal_change)
            diagnostics["pair_valid_fractions"].append(pair_fraction)
            diagnostics["pre_context_gates"].append(pre_fraction)

        diagnostics["change_gate_mean"] = torch.stack(
            [value.mean() for value in diagnostics["change_gates"]]
        ).mean()
        diagnostics["quality_mean"] = torch.stack(
            [value.mean() for value in diagnostics["quality_gates"]]
        ).mean()
        diagnostics["detail_gate_mean"] = torch.stack(
            [value.mean() for value in diagnostics["detail_gates"]]
        ).mean()
        diagnostics["internal_weight_mean"] = torch.stack(
            [value.mean() for value in diagnostics["internal_change_weights"]]
        ).mean()
        diagnostics["external_weight_mean"] = torch.stack(
            [value.mean() for value in diagnostics["external_change_weights"]]
        ).mean()
        diagnostics["pair_valid_fraction_mean"] = torch.stack(
            [value.mean() for value in diagnostics["pair_valid_fractions"]]
        ).mean()
        diagnostics["pre_context_gate_mean"] = torch.stack(
            [value.mean() for value in diagnostics["pre_context_gates"]]
        ).mean()
        return outputs, diagnostics


class HydrologyContext(nn.Module):
    """Multi-dilation context aggregator with a bounded residual update."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        dropout: float = 0.05,
        global_context_enabled: bool = False,
        residual_init: float = 0.15,
        residual_max: float = 0.50,
        residual_parameterization: str = "positive",
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
        self.residual_parameterization = str(residual_parameterization)
        self.residual_max = float(residual_max)
        valid_initialization = (
            0.0 <= residual_init <= self.residual_max
            if self.residual_parameterization == "positive"
            else abs(residual_init) <= self.residual_max
        )
        if (
            self.residual_parameterization not in {"positive", "signed"}
            or self.residual_max <= 0.0
            or not valid_initialization
        ):
            raise ValueError("invalid context residual parameterization or bounds")
        self.gamma = nn.Parameter(torch.tensor(float(residual_init)))
        self.global_context = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(16, channels // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(16, channels // 4), channels, 1),
            )
            if global_context_enabled
            else None
        )
        if self.global_context is not None:
            final = self.global_context[-1]
            assert isinstance(final, nn.Conv2d)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @property
    def residual(self) -> torch.Tensor:
        lower = -self.residual_max if self.residual_parameterization == "signed" else 0.0
        return self.gamma.clamp(lower, self.residual_max)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs + self.residual * self.project(
            torch.cat([path(inputs) for path in self.paths], dim=1)
        )
        if self.global_context is not None:
            output = output + self.global_context(inputs)
        return output


class S1FeatureAlignment(nn.Module):
    """Lightweight per-scale projection of the SAR pyramid.

    Terrain-conditioned fusion was removed after repeated matched ablations
    showed no robust test-set benefit.  Terrain remains available through the
    topographic pyramid, decoder skips, hydrostatic prior, and TAE-KAN path;
    this block keeps only the common SAR projection used by the stronger
    ``w/o TCF`` model.
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
