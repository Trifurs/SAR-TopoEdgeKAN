"""Shared normalized convolution primitives."""

from __future__ import annotations

from torch import nn


def group_count(channels: int, requested_groups: int) -> int:
    """Choose the largest valid group count no greater than the request."""

    if channels <= 0 or requested_groups <= 0:
        raise ValueError("channels and requested_groups must be positive")
    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    raise RuntimeError("No valid GroupNorm group count")


class ConvNormAct(nn.Sequential):
    """Convolution, GroupNorm, and SiLU activation."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(inputs, outputs, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(group_count(outputs, groups), outputs),
            nn.SiLU(inplace=True),
        )
