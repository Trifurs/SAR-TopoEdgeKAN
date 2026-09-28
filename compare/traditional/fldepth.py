"""FlDepth centreline/cross-section flood-depth reconstruction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from compare.common._terrain import (
    EIGHT_CONNECTED,
    checked_depth,
    nearest_boundary_surface,
    outer_boundary,
    validate_inputs,
)


METHOD_NAME = "fldepth"


def _cross_section_surface(component: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    boundary = outer_boundary(component, terrain)
    if not np.any(boundary):
        return nearest_boundary_surface(component, terrain)
    distance = ndimage.distance_transform_edt(component)
    ridges = component & (distance >= ndimage.maximum_filter(distance, size=3)) & (distance >= 1.0)
    if not np.any(ridges):
        return nearest_boundary_surface(component, terrain)
    rows, columns = np.indices(component.shape)
    samples = np.full(component.shape, np.nan, dtype=np.float64)
    for row, column in zip(*np.nonzero(ridges)):
        radius = max(float(distance[row, column]), 1.0)
        radial_distance = np.hypot(rows - row, columns - column)
        bank = boundary & (radial_distance >= max(1.0, 0.70 * radius)) & (radial_distance <= 1.60 * radius)
        if not np.any(bank):
            bank = boundary
        samples[row, column] = float(np.median(terrain[bank]))
    valid_samples = np.isfinite(samples)
    if not np.any(valid_samples):
        return nearest_boundary_surface(component, terrain)
    _, indices = ndimage.distance_transform_edt(~valid_samples, return_indices=True)
    return samples[tuple(indices)]


def estimate_depth(support: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Reconstruct WSE with medial-ridge cross-sections and bank elevations."""

    flood, elevation = validate_inputs(support, terrain)
    depth = np.zeros_like(elevation, dtype=np.float32)
    components, count = ndimage.label(flood, structure=EIGHT_CONNECTED)
    for component_id in range(1, count + 1):
        component = components == component_id
        surface = _cross_section_surface(component, elevation)
        usable = component & np.isfinite(surface)
        depth[usable] = np.maximum(surface[usable] - elevation[usable], 0.0)
    return checked_depth(METHOD_NAME, depth, elevation)
