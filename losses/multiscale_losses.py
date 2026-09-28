"""Masked local-gradient and auxiliary depth objectives."""

from __future__ import annotations

from collections.abc import Sequence
import torch
import torch.nn.functional as F


def masked_average_target(target: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]):
    mask_float = mask.to(target.dtype)
    numerator = F.adaptive_avg_pool2d(target * mask_float, size)
    fraction = F.adaptive_avg_pool2d(mask_float, size)
    averaged = numerator / fraction.clamp_min(1e-8)
    return torch.where(fraction > 0, averaged, torch.zeros_like(averaged)), fraction


def auxiliary_depth_loss(auxiliary_depths: Sequence[torch.Tensor], target: torch.Tensor,
                         positive_mask: torch.Tensor, weights: Sequence[float], beta: float):
    total = target.sum() * 0.0
    terms: list[torch.Tensor] = []
    for index, prediction in enumerate(auxiliary_depths):
        weight = float(weights[index]) if index < len(weights) else 0.0
        if weight == 0.0:
            terms.append(prediction.sum() * 0.0)
            continue
        pooled, fraction = masked_average_target(target, positive_mask, prediction.shape[-2:])
        selected = fraction > 0
        if selected.any():
            per_cell = F.smooth_l1_loss(
                prediction, pooled, beta=beta, reduction="none"
            )
            area = fraction[selected]
            term = (per_cell[selected] * area).sum() / area.sum().clamp_min(1e-8)
        else:
            term = prediction.sum() * 0.0
        terms.append(term)
        total = total + weight * term
    return total, terms


def masked_gradient_consistency_loss(prediction: torch.Tensor, target: torch.Tensor,
                                     positive_mask: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("gradient_huber_beta_m must be positive")
    losses = []
    for axis in (-1, -2):
        if axis == -1:
            pred_delta = prediction[..., 1:] - prediction[..., :-1]
            target_delta = target[..., 1:] - target[..., :-1]
            valid = positive_mask[..., 1:] & positive_mask[..., :-1]
        else:
            pred_delta = prediction[..., 1:, :] - prediction[..., :-1, :]
            target_delta = target[..., 1:, :] - target[..., :-1, :]
            valid = positive_mask[..., 1:, :] & positive_mask[..., :-1, :]
        if valid.any():
            losses.append(F.smooth_l1_loss(pred_delta[valid], target_delta[valid], beta=beta))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0
