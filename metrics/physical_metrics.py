"""Non-claiming physical diagnostics for DSM-guided predictions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


def local_wse_laplacian(
    depth: np.ndarray, z_hyd: np.ndarray, valid_positive: np.ndarray
) -> float:
    depth = np.asarray(depth).squeeze()
    z_hyd = np.asarray(z_hyd).squeeze()
    valid = np.asarray(valid_positive).squeeze().astype(bool)
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    cross_support = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    neighbourhood = convolve(valid.astype(np.uint8), cross_support, mode="constant")
    interior = valid & (neighbourhood >= 5)
    if not np.any(interior):
        return float("nan")
    curvature = np.abs(convolve(z_hyd + depth, kernel, mode="nearest"))
    return float(curvature[interior].mean())


def local_wse_laplacian_reference_error(
    depth: np.ndarray,
    target: np.ndarray,
    z_hyd: np.ndarray,
    valid_positive: np.ndarray,
) -> float:
    """Mean signed-Laplacian mismatch to the reconstructed reference WSE."""

    depth = np.asarray(depth).squeeze()
    target = np.asarray(target).squeeze()
    z_hyd = np.asarray(z_hyd).squeeze()
    valid = np.asarray(valid_positive).squeeze().astype(bool)
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    cross_support = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    neighbourhood = convolve(valid.astype(np.uint8), cross_support, mode="constant")
    interior = valid & (neighbourhood >= 5)
    if not np.any(interior):
        return float("nan")
    predicted = convolve(z_hyd + depth, kernel, mode="nearest")
    reference = convolve(z_hyd + target, kernel, mode="nearest")
    return float(np.abs(predicted - reference)[interior].mean())


def reference_gated_wse_gradient_mae(
    depth: np.ndarray,
    target: np.ndarray,
    z_hyd: np.ndarray,
    valid_positive: np.ndarray,
    normalized_day_difference: np.ndarray,
    sigma_time: float = 0.25,
    sigma_reference: float = 0.12,
    sigma_terrain: float = 0.75,
) -> float:
    """Weighted first-order WSE error matching the reference-gated train loss."""

    for name, value in (
        ("sigma_time", sigma_time),
        ("sigma_reference", sigma_reference),
        ("sigma_terrain", sigma_terrain),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    depth = np.asarray(depth).squeeze()
    target = np.asarray(target).squeeze()
    z_hyd = np.asarray(z_hyd).squeeze()
    valid = np.asarray(valid_positive).squeeze().astype(bool)
    day = np.asarray(normalized_day_difference).squeeze()
    predicted_wse = z_hyd + depth
    reference_wse = z_hyd + target
    numerator = 0.0
    denominator = 0.0
    for axis in (0, 1):
        if axis == 0:
            pair = valid[:-1, :] & valid[1:, :]
            predicted_gradient = predicted_wse[1:, :] - predicted_wse[:-1, :]
            reference_gradient = reference_wse[1:, :] - reference_wse[:-1, :]
            terrain_step = np.abs(z_hyd[1:, :] - z_hyd[:-1, :])
            pair_time = 0.5 * (day[1:, :] + day[:-1, :])
        else:
            pair = valid[:, :-1] & valid[:, 1:]
            predicted_gradient = predicted_wse[:, 1:] - predicted_wse[:, :-1]
            reference_gradient = reference_wse[:, 1:] - reference_wse[:, :-1]
            terrain_step = np.abs(z_hyd[:, 1:] - z_hyd[:, :-1])
            pair_time = 0.5 * (day[:, 1:] + day[:, :-1])
        weight = (
            pair.astype(np.float64)
            * np.exp(-np.abs(reference_gradient) / sigma_reference)
            * np.exp(-terrain_step / sigma_terrain)
            * np.exp(-np.maximum(pair_time, 0.0) / sigma_time)
        )
        numerator += float(
            (np.abs(predicted_gradient - reference_gradient) * weight).sum()
        )
        denominator += float(weight.sum())
    return numerator / denominator if denominator > 0 else float("nan")


def terrain_order_violation_metrics(
    depth: np.ndarray,
    z_hyd: np.ndarray,
    valid_positive: np.ndarray,
    normalized_day_difference: np.ndarray,
    sigma_time: float = 0.25,
    minimum_terrain_step_m: float = 0.02,
    maximum_terrain_step_m: float = 0.75,
) -> dict[str, float]:
    """Weighted uphill depth-change magnitude and occurrence fraction."""

    if sigma_time <= 0:
        raise ValueError("sigma_time must be positive")
    if minimum_terrain_step_m < 0 or maximum_terrain_step_m <= minimum_terrain_step_m:
        raise ValueError("invalid terrain-order step interval")
    depth = np.asarray(depth).squeeze()
    z_hyd = np.asarray(z_hyd).squeeze()
    valid = np.asarray(valid_positive).squeeze().astype(bool)
    day = np.asarray(normalized_day_difference).squeeze()
    magnitude_numerator = 0.0
    violation_numerator = 0.0
    denominator = 0.0
    for axis in (0, 1):
        if axis == 0:
            pair = valid[:-1, :] & valid[1:, :]
            depth_step = depth[1:, :] - depth[:-1, :]
            terrain_step = z_hyd[1:, :] - z_hyd[:-1, :]
            pair_time = 0.5 * (day[1:, :] + day[:-1, :])
        else:
            pair = valid[:, :-1] & valid[:, 1:]
            depth_step = depth[:, 1:] - depth[:, :-1]
            terrain_step = z_hyd[:, 1:] - z_hyd[:, :-1]
            pair_time = 0.5 * (day[:, 1:] + day[:, :-1])
        absolute_step = np.abs(terrain_step)
        pair &= (absolute_step >= minimum_terrain_step_m) & (
            absolute_step <= maximum_terrain_step_m
        )
        weight = pair.astype(np.float64) * np.exp(
            -np.maximum(pair_time, 0.0) / sigma_time
        )
        signed_depth_step = np.sign(terrain_step) * depth_step
        violation = np.maximum(signed_depth_step, 0.0)
        magnitude_numerator += float((violation * weight).sum())
        violation_numerator += float(((signed_depth_step > 0.0) * weight).sum())
        denominator += float(weight.sum())
    if denominator <= 0:
        return {"mae": float("nan"), "fraction": float("nan")}
    return {
        "mae": magnitude_numerator / denominator,
        "fraction": violation_numerator / denominator,
    }


def prediction_continuity_high_relief(
    depth: np.ndarray, local_relief: np.ndarray, valid: np.ndarray, threshold_m: float = 1.0
) -> float:
    depth = np.asarray(depth).squeeze()
    relief = np.asarray(local_relief).squeeze()
    valid_mask = np.asarray(valid).squeeze().astype(bool) & (relief >= threshold_m)
    differences = []
    pair_x = valid_mask[:, 1:] & valid_mask[:, :-1]
    pair_y = valid_mask[1:, :] & valid_mask[:-1, :]
    if np.any(pair_x):
        differences.append(np.abs(depth[:, 1:] - depth[:, :-1])[pair_x])
    if np.any(pair_y):
        differences.append(np.abs(depth[1:, :] - depth[:-1, :])[pair_y])
    return float(np.concatenate(differences).mean()) if differences else float("nan")
