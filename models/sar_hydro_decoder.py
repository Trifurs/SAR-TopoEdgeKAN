"""Additive dual-gated SAR/topographic feature decoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct


class SARHydroDecoder(nn.Module):
    """Dual-Gated Decoder with the production additive skip path."""

    def __init__(
        self, channels: Sequence[int], dropout: float = 0.10,
        groups: int = 8, block_kind: str = "spatial",
        widths: Sequence[int] = (96, 64, 48, 32),
        change_injection_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.widths = [int(value) for value in widths]
        if len(self.widths) != 4 or any(value <= 0 for value in self.widths):
            raise ValueError("SARHydroDecoder requires four positive decoder widths")
        self.change_injection_scale = float(change_injection_scale)
        if self.change_injection_scale < 0.0:
            raise ValueError("change_injection_scale must be nonnegative")
        self.bottleneck = ConvNormAct(int(channels[-1]), self.widths[0], 1, groups=groups)
        skip_indices = (2, 1, 0)
        target_widths = self.widths[1:]
        previous_widths = self.widths[:-1]
        self.up_projection = nn.ModuleList([
            nn.Conv2d(previous, target, 1)
            for previous, target in zip(previous_widths, target_widths)
        ])
        self.sar_lateral = nn.ModuleList([
            nn.Conv2d(int(channels[index]), target, 1)
            for index, target in zip(skip_indices, target_widths)
        ])
        self.terrain_lateral = nn.ModuleList([
            nn.Conv2d(int(channels[index]), target, 1)
            for index, target in zip(skip_indices, target_widths)
        ])
        self.gates = nn.ModuleList([
            nn.Conv2d(3 * target + 2, 2, 1) for target in target_widths
        ])
        self.refine = nn.ModuleList([
            residual_block(block_kind, target, dropout, groups) for target in target_widths
        ])
        self.final = residual_block(block_kind, self.widths[-1], dropout, groups)
        self.change_projection = nn.Sequential(
            nn.Conv2d(int(channels[0]), self.widths[-1], 1), nn.SiLU(inplace=True),
        )

    def forward(
        self, bottleneck: torch.Tensor,
        sar_skips: Sequence[torch.Tensor], terrain_skips: Sequence[torch.Tensor],
        dem_fractions: Sequence[torch.Tensor], sensor_valid: torch.Tensor,
        change_evidence: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        decoded = self.bottleneck(bottleneck)
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
            decoded = self.refine[stage](decoded + sar_gate * sar_skip + terrain_gate * terrain_skip)
            gate_maps.append({
                "sar": sar_gate, "terrain": terrain_gate,
                "sar_mean": sar_gate.mean(), "terrain_mean": terrain_gate.mean(),
            })
        if change_evidence and self.change_injection_scale != 0.0:
            change = F.interpolate(
                self.change_projection(change_evidence[0]), decoded.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            decoded = decoded + self.change_injection_scale * change
        return self.final(decoded), gate_maps
