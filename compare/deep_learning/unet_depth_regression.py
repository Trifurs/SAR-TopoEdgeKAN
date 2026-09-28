"""Plain U-Net flood-depth regression comparator."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from compare.common._depth_regression import (
    DoubleConv,
    DownBlock,
    UpBlock,
    channels_from_config,
    head_from_config,
    validate_input_schema,
)


class UNetDepthRegression(nn.Module):
    """A standard U-Net with a direct range-conditioned depth head."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        super().__init__()
        _, input_channels = validate_input_schema(model_config)
        c0, c1, c2, c3, c4 = channels_from_config(model_config, 5)
        self.stem = DoubleConv(input_channels, c0)
        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.bottom = DownBlock(c3, c4)
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
        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)
        return self.head(d0, flood_range)


def build_unet_depth_regression(config: Mapping[str, object]) -> UNetDepthRegression:
    return UNetDepthRegression(config["model"])
