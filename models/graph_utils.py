"""Shared tensor operations for eight-neighbour terrain graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _roll_with_boundary_mask(
    tensor: torch.Tensor, dy: int, dx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift a raster and return a mask excluding wrapped boundary pixels."""

    neighbour = torch.roll(tensor, shifts=(dy, dx), dims=(-2, -1))
    boundary = torch.ones(
        (tensor.shape[0], 1, tensor.shape[-2], tensor.shape[-1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if dy > 0:
        boundary[:, :, :dy, :] = 0
    elif dy < 0:
        boundary[:, :, dy:, :] = 0
    if dx > 0:
        boundary[:, :, :, :dx] = 0
    elif dx < 0:
        boundary[:, :, :, dx:] = 0
    return neighbour, boundary


def _masked_pool(
    values: torch.Tensor, valid: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Pool values while excluding invalid source pixels."""

    numerator = F.adaptive_avg_pool2d(values * valid, size)
    denominator = F.adaptive_avg_pool2d(valid, size)
    return numerator / denominator.clamp_min(1.0e-6)
