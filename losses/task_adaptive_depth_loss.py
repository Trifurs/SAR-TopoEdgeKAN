"""Train-only task-adaptive depth objective."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from losses.depth_losses import masked_micro_mean, sample_depth_bin_macro_mean, depth_bin_macro_mean
from losses.frozen_soft_depth_balance import FrozenSoftDepthBalance
from losses.soft_depth_balance import soft_depth_balance_weights


def tail_underprediction_factor(prediction: torch.Tensor, target: torch.Tensor,
                                positive: torch.Tensor, train_bins: Sequence[float],
                                alpha: float = 0.0) -> torch.Tensor:
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if alpha == 0 or len(train_bins) < 3:
        return torch.ones_like(target)
    threshold = float(sorted(float(x) for x in train_bins)[2])
    scale = max(float(sorted(float(x) for x in train_bins)[-2]) - threshold, 0.1)
    tail = torch.sigmoid((target - threshold) / (0.25 * scale)) * (target >= threshold).to(target.dtype)
    under = (prediction.detach() < target).to(target.dtype)
    return 1.0 + float(alpha) * tail * under * (positive > 0.5).to(target.dtype)


def task_adaptive_positive_depth_loss(prediction: torch.Tensor, target: torch.Tensor,
                                      positive: torch.Tensor, train_bins: Sequence[float],
                                      beta_m: float = 0.5, log_weight: float = 0.15,
                                      balance: bool = True, under_alpha: float = 0.0,
                                      under_min_m: float = 0.48,
                                      balance_alpha: float = 0.5,
                                      balance_tau: float = 10.0,
                                      train_bin_counts: Sequence[float] | None = None,
                                      frozen_depth_balance: FrozenSoftDepthBalance | None = None,
                                      log_beta: float = 1.0) -> dict[str, torch.Tensor]:
    if beta_m <= 0 or log_weight < 0 or log_beta <= 0:
        raise ValueError("beta_m and log_beta must be positive and log_weight nonnegative")
    metric_pixels = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta_m)
    if balance:
        if frozen_depth_balance is not None:
            metric_pixels = metric_pixels * frozen_depth_balance.weights(target, positive)
        else:
            metric_pixels = metric_pixels * soft_depth_balance_weights(
                target, positive, train_bins, alpha=balance_alpha, tau=balance_tau,
                train_bin_counts=train_bin_counts,
            )
    if under_alpha:
        factor = tail_underprediction_factor(prediction, target, positive, train_bins, under_alpha)
        factor = torch.where(target >= float(under_min_m), factor, torch.ones_like(factor))
        metric_pixels = metric_pixels * factor
    metric = masked_micro_mean(metric_pixels, positive)
    log_pixels = F.smooth_l1_loss(
        torch.log1p(prediction),
        torch.log1p(target.clamp_min(0.0)),
        reduction="none",
        beta=float(log_beta),
    )
    logarithmic = masked_micro_mean(log_pixels, positive)
    total = metric + float(log_weight) * logarithmic
    return {"depth": total, "depth_linear": metric, "depth_log": logarithmic,
            "depth_final": total.detach() * 0.0, "depth_bias": total.detach() * 0.0}
