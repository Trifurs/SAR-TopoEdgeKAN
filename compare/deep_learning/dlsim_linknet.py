"""DLSIM-inspired LinkNet comparator for conditional flood depth."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from compare.common._depth_regression import (
    LinkDecoderBlock,
    ResidualBlock,
    channels_from_config,
    head_from_config,
    validate_input_schema,
)


class DLSIMLinkNet(nn.Module):
    """Residual LinkNet adaptation of the DLSIM water-level regression branch."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        super().__init__()
        _, input_channels = validate_input_schema(model_config)
        c0, c1, c2, c3, c4 = channels_from_config(model_config, 5)
        self.stem = ResidualBlock(input_channels, c0)
        self.down1 = ResidualBlock(c0, c1, stride=2)
        self.down2 = ResidualBlock(c1, c2, stride=2)
        self.down3 = ResidualBlock(c2, c3, stride=2)
        self.bottom = ResidualBlock(c3, c4, stride=2)
        self.decode3 = LinkDecoderBlock(c4, c3, c3)
        self.decode2 = LinkDecoderBlock(c3, c2, c2)
        self.decode1 = LinkDecoderBlock(c2, c1, c1)
        self.decode0 = LinkDecoderBlock(c1, c0, c0)
        self.head = head_from_config(c0, model_config)

    def forward(self, inputs: torch.Tensor, flood_range: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.bottom(x3)
        d3 = self.decode3(x4, x3)
        d2 = self.decode2(d3, x2)
        d1 = self.decode1(d2, x1)
        d0 = self.decode0(d1, x0)
        return self.head(d0, flood_range)


def build_dlsim_linknet(config: Mapping[str, object]) -> DLSIMLinkNet:
    return DLSIMLinkNet(config["model"])
