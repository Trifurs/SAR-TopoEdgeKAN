"""DLSIM-inspired Attention U-Net comparator for conditional flood depth."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from compare.common._depth_regression import (
    AttentionGate,
    DoubleConv,
    DownBlock,
    UpBlock,
    channels_from_config,
    head_from_config,
    validate_input_schema,
)


class DLSIMAttentionUNet(nn.Module):
    """Attention U-Net adaptation of the DLSIM water-level regression branch."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        super().__init__()
        _, input_channels = validate_input_schema(model_config)
        c0, c1, c2, c3, c4 = channels_from_config(model_config, 5)
        self.stem = DoubleConv(input_channels, c0)
        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.bottom = DownBlock(c3, c4)
        self.attention3 = AttentionGate(c3, c4)
        self.attention2 = AttentionGate(c2, c3)
        self.attention1 = AttentionGate(c1, c2)
        self.attention0 = AttentionGate(c0, c1)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.up0 = UpBlock(c1, c0, c0)
        self.head = head_from_config(c0, model_config)

    def forward(self, inputs: torch.Tensor, flood_range: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.bottom(x3)
        d3 = self.up3(x4, self.attention3(x3, x4))
        d2 = self.up2(d3, self.attention2(x2, d3))
        d1 = self.up1(d2, self.attention1(x1, d2))
        d0 = self.up0(d1, self.attention0(x0, d1))
        return self.head(d0, flood_range)


def build_dlsim_attention_unet(config: Mapping[str, object]) -> DLSIMAttentionUNet:
    return DLSIMAttentionUNet(config["model"])
