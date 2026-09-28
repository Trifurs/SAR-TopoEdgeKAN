"""FwDET v2.0 flood-depth reconstruction."""

from __future__ import annotations

import numpy as np

from compare.common._terrain import checked_depth, nearest_boundary_surface, validate_inputs


METHOD_NAME = "fwdet_v2"


def estimate_depth(support: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Allocate nearest valid outer-boundary elevations as water surface.

    The published ArcGIS workflow uses cost allocation. This inland DSM
    adaptation uses equivalent Euclidean allocation because the subset has no
    permanent-water layer.
    """

    flood, elevation = validate_inputs(support, terrain)
    water_surface = nearest_boundary_surface(flood, elevation)
    depth = np.zeros_like(elevation, dtype=np.float32)
    usable = flood & np.isfinite(water_surface)
    depth[usable] = np.maximum(water_surface[usable] - elevation[usable], 0.0)
    return checked_depth(METHOD_NAME, depth, elevation)
