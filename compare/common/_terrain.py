"""Shared raster primitives for named terrain-based comparison models."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def validate_inputs(support: np.ndarray, terrain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return equally sized finite-terrain support and elevation arrays."""

    flood = np.asarray(support, dtype=bool)
    elevation = np.asarray(terrain, dtype=np.float64)
    if flood.ndim != 2 or elevation.ndim != 2 or flood.shape != elevation.shape:
        raise ValueError("Flood support and terrain must be equally sized 2-D arrays")
    return flood & np.isfinite(elevation), elevation


def outer_boundary(flood: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Return valid dry-land cells bordering a flood-support component."""

    boundary = ndimage.binary_dilation(flood, structure=EIGHT_CONNECTED) & ~flood
    boundary &= np.isfinite(terrain)
    above_sea = boundary & (terrain > 0.0)
    return above_sea if np.any(above_sea) else boundary


def nearest_boundary_surface(flood: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Allocate each cell the elevation of its nearest outer-boundary cell."""

    boundary = outer_boundary(flood, terrain)
    if not np.any(boundary):
        return np.full_like(terrain, np.nan, dtype=np.float64)
    _, indices = ndimage.distance_transform_edt(~boundary, return_indices=True)
    return terrain[tuple(indices)]


def checked_depth(method: str, depth: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Validate a named model output before metrics are accumulated."""

    if depth.shape != np.asarray(terrain).shape or not np.all(np.isfinite(depth)):
        raise RuntimeError(f"{method} produced invalid depth output")
    return depth.astype(np.float32, copy=False)
