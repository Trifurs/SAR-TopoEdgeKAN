"""Continuous depth metrics evaluated only on reliable positive labels."""

from __future__ import annotations

from typing import Any

import numpy as np


DEPTH_METRIC_NAMES = (
    "mae",
    "rmse",
    "median_absolute_error",
    "bias",
    "r2",
    "nse",
    "log1p_mae",
    "p90_absolute_error",
    "within_0.25m",
    "within_0.50m",
    "within_1.00m",
)


def depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if target.size == 0:
        return {name: float("nan") for name in DEPTH_METRIC_NAMES} | {"pixels": 0}
    error = prediction - target
    absolute = np.abs(error)
    squared = np.square(error)
    denominator = np.sum((target - target.mean()) ** 2)
    efficiency = float("nan") if denominator <= 0 else float(1.0 - squared.sum() / denominator)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "median_absolute_error": float(np.median(absolute)),
        "bias": float(error.mean()),
        "r2": efficiency,
        "nse": efficiency,
        "log1p_mae": float(np.abs(np.log1p(np.clip(prediction, 0, None)) - np.log1p(target)).mean()),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "within_0.25m": float(np.mean(absolute <= 0.25)),
        "within_0.50m": float(np.mean(absolute <= 0.50)),
        "within_1.00m": float(np.mean(absolute <= 1.00)),
        "pixels": int(target.size),
    }


def prefixed(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}
