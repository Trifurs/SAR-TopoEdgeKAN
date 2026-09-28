"""Torchvision ResNet18 encoder-decoder flood-depth regression comparator."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from compare.common._depth_regression import (
    LinkDecoderBlock,
    head_from_config,
    validate_input_schema,
)


class ResNet18DepthRegression(nn.Module):
    """A random-initialized ResNet18 encoder with a full-resolution decoder."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        super().__init__()
        _, input_channels = validate_input_schema(model_config)
        try:
            from torchvision.models import resnet18
        except ImportError as exc:  # pragma: no cover - dependency failure is actionable.
            raise ImportError("resnet18_depth_regression requires torchvision") from exc
        encoder = resnet18(weights=None)
        encoder.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.decode3 = LinkDecoderBlock(512, 256, 256)
        self.decode2 = LinkDecoderBlock(256, 128, 128)
        self.decode1 = LinkDecoderBlock(128, 64, 64)
        self.decode0 = LinkDecoderBlock(64, 64, 64)
        self.full_resolution = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
        )
        self.head = head_from_config(48, model_config)

    def forward(self, inputs: torch.Tensor, flood_range: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(inputs)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d3 = self.decode3(x4, x3)
        d2 = self.decode2(d3, x2)
        d1 = self.decode1(d2, x1)
        d0 = self.decode0(d1, stem)
        full = torch.nn.functional.interpolate(
            d0, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.head(self.full_resolution(full), flood_range)


def build_resnet18_depth_regression(config: Mapping[str, object]) -> ResNet18DepthRegression:
    return ResNet18DepthRegression(config["model"])
