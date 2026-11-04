"""Shared robust depth and log-depth regression terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.depth_losses import masked_micro_mean


def task_adaptive_positive_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    *,
    beta_m: float,
    log_weight: float,
    log_beta: float,
) -> dict[str, torch.Tensor]:
    """Apply pixel-micro Smooth-L1 terms on the shared supervision mask."""

    if beta_m <= 0 or log_weight < 0 or log_beta <= 0:
        raise ValueError("beta_m and log_beta must be positive and log_weight nonnegative")
    metric_pixels = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta_m)
    metric = masked_micro_mean(metric_pixels, positive)
    log_pixels = F.smooth_l1_loss(
        torch.log1p(prediction),
        torch.log1p(target.clamp_min(0.0)),
        reduction="none",
        beta=float(log_beta),
    )
    logarithmic = masked_micro_mean(log_pixels, positive)
    total = metric + float(log_weight) * logarithmic
    return {"depth": total, "depth_linear": metric, "depth_log": logarithmic}
