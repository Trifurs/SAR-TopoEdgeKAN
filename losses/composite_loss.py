"""Shared production depth objective with strict partial-label masking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from datasets.supervision_masks import canonical_positive_mask_from_batch, supervision_mask_counts
from losses.depth_losses import (
    depth_bin_macro_root_mean_square,
    event_depth_hierarchical_macro_mean,
    event_depth_hierarchical_macro_root_mean_square,
    event_macro_masked_mean,
    event_macro_root_mean_square,
    event_mean_bias_loss,
    masked_micro_mean,
)
from losses.task_adaptive_depth_loss import task_adaptive_positive_depth_loss


def _event_ids(batch: Mapping[str, Any]) -> Sequence[str] | None:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    values = metadata.get("source_event_id")
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    if isinstance(values, str):
        return [values]
    return None


class CompositeFloodDepthLoss(nn.Module):
    """The same robust and metric-alignment objective for every learned model."""

    def __init__(
        self,
        loss_config: Mapping[str, Any],
        train_depth_bins: Sequence[float] | None = None,
        primary_depth_bins: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(loss_config)
        if self.config["objective_mode"] != "task_adaptive":
            raise ValueError("Only the shared task-adaptive objective is supported")
        self.train_depth_bins = tuple(float(value) for value in (train_depth_bins or ()))
        self.primary_depth_bins = tuple(float(value) for value in (primary_depth_bins or ()))

    def forward(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        label = batch["label"]
        positive = canonical_positive_mask_from_batch(batch)
        prediction = outputs["conditional_depth"] if "conditional_depth" in outputs else outputs["depth"]
        events = _event_ids(batch)
        components = task_adaptive_positive_depth_loss(
            prediction, label, positive,
            beta_m=float(self.config["depth_huber_beta_m"]),
            log_weight=float(self.config["lambda_log"]),
            log_beta=float(self.config["log_depth_huber_beta"]),
        )
        absolute_error = (prediction - label).abs()
        squared_error = (prediction - label).square()
        components["metric_pixel_mae"] = masked_micro_mean(absolute_error, positive)
        metric_pixel_mse = masked_micro_mean(squared_error, positive)
        components["metric_pixel_rmse"] = torch.sqrt(metric_pixel_mse.clamp_min(1.0e-12))
        components["metric_event_mae"] = event_macro_masked_mean(absolute_error, positive, events)
        components["metric_event_rmse"] = event_macro_root_mean_square(prediction - label, positive, events)
        components["metric_event_hierarchical_mae"] = event_depth_hierarchical_macro_mean(
            absolute_error, label, positive, events,
            self.primary_depth_bins, self.train_depth_bins,
        )
        components["metric_depth_bin_rmse"] = depth_bin_macro_root_mean_square(
            prediction - label, label, positive, self.train_depth_bins,
        )
        components["metric_event_hierarchical_rmse"] = event_depth_hierarchical_macro_root_mean_square(
            prediction - label, label, positive, events,
            self.primary_depth_bins, self.train_depth_bins,
        )
        components["depth_bias"] = event_mean_bias_loss(
            prediction, label, positive, events,
            beta=float(self.config["depth_bias_beta_m"]),
        )
        total = (
            float(self.config["lambda_depth"]) * components["depth"]
            + float(self.config["lambda_metric_pixel_mae"]) * components["metric_pixel_mae"]
            + float(self.config["lambda_metric_pixel_rmse"]) * components["metric_pixel_rmse"]
            + float(self.config["lambda_metric_event_mae"]) * components["metric_event_mae"]
            + float(self.config["lambda_metric_event_rmse"]) * components["metric_event_rmse"]
            + float(self.config["lambda_metric_event_hierarchical_mae"]) * components["metric_event_hierarchical_mae"]
            + float(self.config["lambda_metric_depth_bin_rmse"]) * components["metric_depth_bin_rmse"]
            + float(self.config["lambda_metric_event_hierarchical_rmse"]) * components["metric_event_hierarchical_rmse"]
            + float(self.config["lambda_depth_bias"]) * components["depth_bias"]
        )
        components["total"] = total
        components.update(supervision_mask_counts(batch))
        return total, components
