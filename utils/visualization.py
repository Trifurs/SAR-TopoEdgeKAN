"""Non-interactive prediction panels."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_prediction_panel(
    path: Path,
    *,
    s1_change: np.ndarray,
    dsm: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    valid_label: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_view = np.where(valid_label, target, np.nan)
    error_view = np.where(valid_label, np.abs(prediction - target), np.nan)
    panels = [(s1_change, "SAR change", "coolwarm")]
    panels.extend([
        (dsm, "DSM (m)", "terrain"),
        (target_view, "Target depth (m)", "Blues"),
        (prediction, "Predicted depth (m)", "Blues"),
        (uncertainty, "Uncertainty scale (m)", "magma"),
        (error_view, "Absolute error on valid labels", "inferno"),
    ])
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        shown = axis.imshow(np.asarray(image).squeeze(), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    axes.flat[-1].axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
