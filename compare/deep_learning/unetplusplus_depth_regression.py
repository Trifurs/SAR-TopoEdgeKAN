"""Nested U-Net++ flood-depth regression comparator."""

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


class UNetPlusPlusDepthRegression(nn.Module):
    """Four-stage U-Net++ with dense decoder skips and a range-conditioned head."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        super().__init__()
        _, input_channels = validate_input_schema(model_config)
        c0, c1, c2, c3 = channels_from_config(model_config, 4)
        self.x0_0 = DoubleConv(input_channels, c0)
        self.x1_0 = DownBlock(c0, c1)
        self.x2_0 = DownBlock(c1, c2)
        self.x3_0 = DownBlock(c2, c3)
        self.x0_1 = UpBlock(c1, c0, c0)
        self.x1_1 = UpBlock(c2, c1, c1)
        self.x2_1 = UpBlock(c3, c2, c2)
        self.x0_2 = UpBlock(c1, 2 * c0, c0)
        self.x1_2 = UpBlock(c2, 2 * c1, c1)
        self.x0_3 = UpBlock(c1, 3 * c0, c0)
        self.head = head_from_config(c0, model_config)

    def forward(self, inputs: torch.Tensor, flood_range: torch.Tensor) -> dict[str, torch.Tensor]:
        x0_0 = self.x0_0(inputs)
        x1_0 = self.x1_0(x0_0)
        x0_1 = self.x0_1(x1_0, x0_0)
        x2_0 = self.x2_0(x1_0)
        x1_1 = self.x1_1(x2_0, x1_0)
        x0_2 = self.x0_2(x1_1, torch.cat((x0_0, x0_1), dim=1))
        x3_0 = self.x3_0(x2_0)
        x2_1 = self.x2_1(x3_0, x2_0)
        x1_2 = self.x1_2(x2_1, torch.cat((x1_0, x1_1), dim=1))
        x0_3 = self.x0_3(x1_2, torch.cat((x0_0, x0_1, x0_2), dim=1))
        return self.head(x0_3, flood_range)


def build_unetplusplus_depth_regression(config: Mapping[str, object]) -> UNetPlusPlusDepthRegression:
    return UNetPlusPlusDepthRegression(config["model"])
