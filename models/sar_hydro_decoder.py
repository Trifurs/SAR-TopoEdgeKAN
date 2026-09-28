"""Independent-gate decoder for SAR-and-terrain flood-depth features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct


class _NestedFusionNode(nn.Module):
    """One compact dense node used by the nested hydrologic decoder."""

    def __init__(
        self,
        high_channels: int,
        channels: int,
        dense_inputs: int,
        dropout: float,
        groups: int,
        block_kind: str,
        refine_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.high = ConvNormAct(high_channels, channels, 1, groups=groups)
        self.terrain = ConvNormAct(channels, channels, 1, groups=groups)
        self.terrain_gate = nn.Conv2d(3 * channels + 1, 1, 1)
        self.merge = ConvNormAct(
            (dense_inputs + 2) * channels,
            channels,
            1,
            groups=groups,
        )
        self.refine = (
            residual_block(block_kind, channels, dropout, groups)
            if refine_enabled
            else nn.Identity()
        )

    def forward(
        self,
        high: torch.Tensor,
        dense: Sequence[torch.Tensor],
        terrain: torch.Tensor,
        dem_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not dense:
            raise ValueError("nested decoder node requires at least one dense skip")
        reference = dense[0]
        high = F.interpolate(
            high, reference.shape[-2:], mode="bilinear", align_corners=False
        )
        high = self.high(high)
        terrain = self.terrain(terrain)
        dem_fraction = dem_fraction.to(high.dtype)
        gate = torch.sigmoid(
            self.terrain_gate(
                torch.cat((high, dense[-1], terrain, dem_fraction), dim=1)
            )
        ) * dem_fraction
        merged = self.merge(torch.cat((high, *dense, gate * terrain), dim=1))
        return self.refine(merged), gate


class SARHydroDecoder(nn.Module):
    """Dual-Gated Decoder (DGD) for SAR and topographic skip features."""

    def __init__(
        self,
        channels: Sequence[int],
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        widths: Sequence[int] = (96, 64, 48, 32),
        auxiliary_count: int = 1,
        auxiliary_stage: int = 0,
        change_injection_scale: float = 0.05,
        skip_fusion: str = "additive",
        nested_refinement: str = "all",
    ) -> None:
        super().__init__()
        widths = [int(value) for value in widths]
        supported_widths = ([96, 64, 48, 32], [128, 96, 64, 32])
        if widths not in supported_widths:
            raise ValueError(
                "SARHydroDecoder widths must be one of "
                f"{supported_widths}, got {widths}"
            )
        if auxiliary_count not in {0, 1, 2}:
            raise ValueError("auxiliary_count must be 0, 1, or 2")
        if auxiliary_stage not in {0, 1}:
            raise ValueError("auxiliary_stage must be 0 (1/4) or 1 (1/2)")
        self.widths = widths
        self.auxiliary_stage = int(auxiliary_stage)
        self.change_injection_scale = float(change_injection_scale)
        self.skip_fusion = str(skip_fusion)
        self.nested_refinement = str(nested_refinement)
        if self.nested_refinement not in {"all", "coarse", "none"}:
            raise ValueError("nested_refinement must be all, coarse, or none")
        if self.skip_fusion not in {"additive", "concatenative", "nested"}:
            raise ValueError(
                "decoder skip_fusion must be additive, concatenative, or nested"
            )
        if self.change_injection_scale < 0.0:
            raise ValueError("change_injection_scale must be nonnegative")
        if self.skip_fusion == "nested":
            c0, c1, c2, c3 = (int(value) for value in channels)
            # U-Net++-style dense semantic paths, implemented with 1x1 fusion and
            # one configured residual block per node.  The graph-refined c3 tensor
            # is the deepest node, so TAE-KAN evidence reaches all three rows.
            self.nested_nodes = nn.ModuleDict(
                {
                    "x2_1": _NestedFusionNode(
                        c3,
                        c2,
                        1,
                        dropout,
                        groups,
                        block_kind,
                        self.nested_refinement != "none",
                    ),
                    "x1_1": _NestedFusionNode(
                        c2,
                        c1,
                        1,
                        dropout,
                        groups,
                        block_kind,
                        self.nested_refinement != "none",
                    ),
                    "x0_1": _NestedFusionNode(
                        c1,
                        c0,
                        1,
                        dropout,
                        groups,
                        block_kind,
                        self.nested_refinement == "all",
                    ),
                    "x1_2": _NestedFusionNode(
                        c2,
                        c1,
                        2,
                        dropout,
                        groups,
                        block_kind,
                        self.nested_refinement != "none",
                    ),
                    "x0_2": _NestedFusionNode(
                        c1,
                        c0,
                        2,
                        dropout,
                        groups,
                        block_kind,
                        self.nested_refinement == "all",
                    ),
                    "x0_3": _NestedFusionNode(
                        c1,
                        c0,
                        3,
                        dropout,
                        groups,
                        block_kind,
                        self.nested_refinement == "all",
                    ),
                }
            )
            output_layers: list[nn.Module] = [
                ConvNormAct(c0, widths[-1], 3, groups=groups)
            ]
            if self.nested_refinement == "all":
                output_layers.append(
                    residual_block(block_kind, widths[-1], dropout, groups)
                )
            self.nested_output = nn.Sequential(*output_layers)
            auxiliary_channels = (c2, c1)
            self.auxiliary = (
                nn.ModuleList(
                    [nn.Conv2d(auxiliary_channels[auxiliary_stage], 1, 1)]
                    if auxiliary_count == 1
                    else [
                        nn.Conv2d(auxiliary_channels[0], 1, 1),
                        nn.Conv2d(auxiliary_channels[1], 1, 1),
                    ]
                )
                if auxiliary_count
                else None
            )
            self.change_projection = nn.Sequential(
                nn.Conv2d(c0, widths[-1], 1),
                nn.SiLU(inplace=True),
            )
            # Unused decoder paths are absent in this architecture so the
            # parameter count reflects only executed computation.
            self.bottleneck = None
            self.up_projection = None
            self.sar_lateral = None
            self.terrain_lateral = None
            self.gates = None
            self.skip_merge = None
            self.refine = None
            self.final = None
            return
        self.nested_nodes = None
        self.nested_output = None
        self.bottleneck = ConvNormAct(int(channels[-1]), widths[0], 1, groups=groups)
        skip_indices = (2, 1, 0)
        target_widths = (widths[1], widths[2], widths[3])
        previous_widths = (widths[0], widths[1], widths[2])
        self.up_projection = nn.ModuleList(
            [nn.Conv2d(previous, target, 1) for previous, target in zip(previous_widths, target_widths)]
        )
        self.sar_lateral = nn.ModuleList(
            [nn.Conv2d(int(channels[index]), target, 1) for index, target in zip(skip_indices, target_widths)]
        )
        self.terrain_lateral = nn.ModuleList(
            [nn.Conv2d(int(channels[index]), target, 1) for index, target in zip(skip_indices, target_widths)]
        )
        self.gates = nn.ModuleList(
            [nn.Conv2d(3 * target + 2, 2, 1) for target in target_widths]
        )
        self.skip_merge = (
            nn.ModuleList(
                [nn.Conv2d(3 * target, target, 1, bias=False) for target in target_widths]
            )
            if self.skip_fusion == "concatenative"
            else None
        )
        if self.skip_merge is not None:
            # Exact additive initialization makes an architecture-transfer run
            # start from the established decoder before learning cross-channel
            # interactions among the three streams.
            with torch.no_grad():
                for layer, target in zip(self.skip_merge, target_widths):
                    layer.weight.zero_()
                    indices = torch.arange(target)
                    layer.weight[indices, indices, 0, 0] = 1.0
                    layer.weight[indices, target + indices, 0, 0] = 1.0
                    layer.weight[indices, 2 * target + indices, 0, 0] = 1.0
        self.refine = nn.ModuleList(
            [residual_block(block_kind, target, dropout, groups) for target in target_widths]
        )
        self.final = residual_block(block_kind, widths[-1], dropout, groups)
        self.auxiliary = (
            nn.ModuleList(
                [nn.Conv2d(target_widths[auxiliary_stage], 1, 1)]
                if auxiliary_count == 1
                else [nn.Conv2d(target_widths[0], 1, 1), nn.Conv2d(target_widths[1], 1, 1)]
            )
            if auxiliary_count
            else None
        )
        self.change_projection = nn.Sequential(
            nn.Conv2d(int(channels[0]), widths[-1], 1),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        sar_skips: Sequence[torch.Tensor],
        terrain_skips: Sequence[torch.Tensor],
        dem_fractions: Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
        change_evidence: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
        if self.skip_fusion == "nested":
            return self._forward_nested(
                bottleneck,
                sar_skips,
                terrain_skips,
                dem_fractions,
                sensor_valid,
                change_evidence,
            )
        assert self.bottleneck is not None
        assert self.up_projection is not None
        assert self.sar_lateral is not None
        assert self.terrain_lateral is not None
        assert self.gates is not None
        assert self.refine is not None
        assert self.final is not None
        decoded = self.bottleneck(bottleneck)
        auxiliaries: list[torch.Tensor] = []
        gate_maps: list[dict[str, Any]] = []
        sensor_valid = sensor_valid.to(decoded.dtype)
        for stage, index in enumerate((2, 1, 0)):
            decoded = F.interpolate(
                decoded, sar_skips[index].shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = self.up_projection[stage](decoded)
            sar_skip = self.sar_lateral[stage](sar_skips[index])
            terrain_skip = self.terrain_lateral[stage](terrain_skips[index])
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, decoded.shape[-2:])
            dem_fraction = dem_fractions[index].to(decoded.dtype)
            logits = self.gates[stage](
                torch.cat((decoded, sar_skip, terrain_skip, sensor_fraction, dem_fraction), dim=1)
            )
            sar_gate = torch.sigmoid(logits[:, 0:1]) * sensor_fraction
            terrain_gate = torch.sigmoid(logits[:, 1:2]) * dem_fraction
            gated_sar = sar_gate * sar_skip
            gated_terrain = terrain_gate * terrain_skip
            if self.skip_merge is None:
                merged = decoded + gated_sar + gated_terrain
            else:
                merged = self.skip_merge[stage](
                    torch.cat((decoded, gated_sar, gated_terrain), dim=1)
                )
            decoded = self.refine[stage](merged)
            gate_maps.append({
                "sar": sar_gate,
                "terrain": terrain_gate,
                "sar_mean": sar_gate.mean(),
                "terrain_mean": terrain_gate.mean(),
            })
            if self.auxiliary is not None:
                if len(self.auxiliary) == 1 and stage == self.auxiliary_stage:
                    auxiliaries.append(F.softplus(self.auxiliary[0](decoded)))
                elif len(self.auxiliary) == 2 and stage in {0, 1}:
                    auxiliaries.append(F.softplus(self.auxiliary[stage](decoded)))
        if change_evidence and self.change_injection_scale != 0.0:
            change = F.interpolate(
                self.change_projection(change_evidence[0]),
                decoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = decoded + self.change_injection_scale * change
        return self.final(decoded), auxiliaries, gate_maps

    def _forward_nested(
        self,
        bottleneck: torch.Tensor,
        sar_skips: Sequence[torch.Tensor],
        terrain_skips: Sequence[torch.Tensor],
        dem_fractions: Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
        change_evidence: Sequence[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
        assert self.nested_nodes is not None
        assert self.nested_output is not None
        x0_0, x1_0, x2_0 = sar_skips[0], sar_skips[1], sar_skips[2]
        x2_1, gate_2_1 = self.nested_nodes["x2_1"](
            bottleneck, (x2_0,), terrain_skips[2], dem_fractions[2]
        )
        x1_1, gate_1_1 = self.nested_nodes["x1_1"](
            x2_0, (x1_0,), terrain_skips[1], dem_fractions[1]
        )
        x0_1, gate_0_1 = self.nested_nodes["x0_1"](
            x1_0, (x0_0,), terrain_skips[0], dem_fractions[0]
        )
        x1_2, gate_1_2 = self.nested_nodes["x1_2"](
            x2_1, (x1_0, x1_1), terrain_skips[1], dem_fractions[1]
        )
        x0_2, gate_0_2 = self.nested_nodes["x0_2"](
            x1_1, (x0_0, x0_1), terrain_skips[0], dem_fractions[0]
        )
        x0_3, gate_0_3 = self.nested_nodes["x0_3"](
            x1_2,
            (x0_0, x0_1, x0_2),
            terrain_skips[0],
            dem_fractions[0],
        )
        decoded = self.nested_output(x0_3)
        if change_evidence and self.change_injection_scale != 0.0:
            change = F.interpolate(
                self.change_projection(change_evidence[0]),
                decoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = decoded + self.change_injection_scale * change

        auxiliaries: list[torch.Tensor] = []
        if self.auxiliary is not None:
            candidates = (x2_1, x1_2)
            if len(self.auxiliary) == 1:
                auxiliaries.append(
                    F.softplus(self.auxiliary[0](candidates[self.auxiliary_stage]))
                )
            else:
                auxiliaries.extend(
                    F.softplus(layer(value))
                    for layer, value in zip(self.auxiliary, candidates)
                )
        sensor_valid = sensor_valid.to(decoded.dtype)
        gate_maps = []
        for gate in (gate_2_1, gate_1_1, gate_0_1, gate_1_2, gate_0_2, gate_0_3):
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, gate.shape[-2:])
            gate_maps.append(
                {
                    "sar": sensor_fraction,
                    "terrain": gate,
                    "sar_mean": sensor_fraction.mean(),
                    "terrain_mean": gate.mean(),
                }
            )
        return decoded, auxiliaries, gate_maps
