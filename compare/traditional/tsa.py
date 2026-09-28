"""Trend Surface Analysis flood-depth reconstruction."""

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


METHOD_NAME = "tsa"


def _linear_trend_surface(
    rows: np.ndarray, columns: np.ndarray, values: np.ndarray, shape: tuple[int, int],
) -> np.ndarray:
    if values.size < 3:
        return np.full(shape, float(np.nanmedian(values)) if values.size else np.nan)
    row_center, column_center = float(rows.mean()), float(columns.mean())
    row_scale = max(float(rows.std()), 1.0)
    column_scale = max(float(columns.std()), 1.0)
    design = np.column_stack((
        np.ones(values.size),
        (rows - row_center) / row_scale,
        (columns - column_center) / column_scale,
    ))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    grid_rows, grid_columns = np.indices(shape)
    return (
        coefficients[0]
        + coefficients[1] * (grid_rows - row_center) / row_scale
        + coefficients[2] * (grid_columns - column_center) / column_scale
    )


def estimate_depth(support: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Fit a first-degree WSE trend per connected flood component."""

    flood, elevation = validate_inputs(support, terrain)
    depth = np.zeros_like(elevation, dtype=np.float32)
    components, count = ndimage.label(flood, structure=EIGHT_CONNECTED)
    for component_id in range(1, count + 1):
        component = components == component_id
        boundary = outer_boundary(component, elevation)
        rows, columns = np.nonzero(boundary)
        if rows.size < 3:
            surface = nearest_boundary_surface(component, elevation)
        else:
            surface = _linear_trend_surface(rows, columns, elevation[boundary], elevation.shape)
        usable = component & np.isfinite(surface)
        depth[usable] = np.maximum(surface[usable] - elevation[usable], 0.0)
    return checked_depth(METHOD_NAME, depth, elevation)
