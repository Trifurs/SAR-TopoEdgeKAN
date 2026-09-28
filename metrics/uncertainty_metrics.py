"""Laplace uncertainty calibration diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def uncertainty_metrics(
    prediction: np.ndarray, target: np.ndarray, scale: np.ndarray
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(scale) & (scale > 0)
    prediction, target, scale = prediction[valid], target[valid], scale[valid]
    if target.size == 0:
        return {"laplace_nll": float("nan"), "spearman_uncertainty_abs_error": float("nan")}
    absolute = np.abs(prediction - target)
    result: dict[str, float] = {
        "laplace_nll": float(np.mean(absolute / scale + np.log(2.0 * scale))),
    }
    for coverage in (0.50, 0.80, 0.90, 0.95):
        half_width = -scale * np.log(1.0 - coverage)
        name = str(int(coverage * 100))
        result[f"coverage_{name}"] = float(np.mean(absolute <= half_width))
        result[f"mean_interval_width_{name}"] = float(np.mean(2.0 * half_width))
    if target.size < 2 or np.all(scale == scale[0]) or np.all(absolute == absolute[0]):
        correlation = float("nan")
    else:
        correlation = float(spearmanr(scale, absolute).statistic)
    result["spearman_uncertainty_abs_error"] = correlation
    return result
