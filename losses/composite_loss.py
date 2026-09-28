"""Production flood-depth objective with strict partial-label masking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from losses.depth_losses import (
    depth_bin_macro_root_mean_square,
    event_depth_hierarchical_macro_mean,
    event_depth_hierarchical_macro_root_mean_square,
    event_depth_hierarchical_bias_loss,
    event_mean_bias_loss,
    event_macro_masked_mean,
    event_macro_root_mean_square,
    event_depth_exceedance_loss,
    laplace_nll_loss,
    masked_micro_mean,
    positive_depth_losses,
    tail_underprediction_loss,
)
from losses.physics_losses import (
    gated_terrain_order_loss,
    reference_gated_wse_gradient_loss,
    weak_physics_pair_loss,
    weak_wse_laplacian_loss,
)
from losses.multiscale_losses import auxiliary_depth_loss, masked_gradient_consistency_loss
from datasets.preprocessing import RELIABILITY_NAMES
from datasets.supervision_masks import (
    canonical_positive_mask_from_batch,
    supervision_mask_counts,
)
from losses.task_adaptive_depth_loss import task_adaptive_positive_depth_loss
from losses.frozen_soft_depth_balance import FrozenSoftDepthBalance


def _reliability_names(batch: Mapping[str, Any]) -> tuple[str, ...]:
    values = batch.get("reliability_names")
    if isinstance(values, (list, tuple)) and values and all(isinstance(value, str) for value in values):
        return tuple(values)
    if isinstance(values, (list, tuple)) and values and all(
        isinstance(value, (list, tuple)) and value and isinstance(value[0], str)
        for value in values
    ):
        return tuple(str(value[0]) for value in values)
    metadata = batch.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("reliability_names")
        if isinstance(nested, (list, tuple)) and nested and all(
            isinstance(value, (list, tuple)) and value and isinstance(value[0], str)
            for value in nested
        ):
            return tuple(str(value[0]) for value in nested)
    return RELIABILITY_NAMES


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
    def __init__(
        self,
        loss_config: Mapping[str, Any],
        train_depth_bins: Sequence[float] | None = None,
        primary_depth_bins: Sequence[float] | None = None,
        train_depth_bin_counts: Sequence[float] | None = None,
        frozen_depth_balance: FrozenSoftDepthBalance | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(loss_config)
        self.lambda_absolute_laplacian = float(
            self.config.get("lambda_absolute_laplacian", self.config.get("lambda_wse", 0.0))
        )
        self.train_depth_bins = tuple(float(value) for value in (train_depth_bins or ()))
        self.primary_depth_bins = tuple(
            float(value) for value in (primary_depth_bins or ())
        )
        self.train_depth_bin_counts = tuple(float(value) for value in (train_depth_bin_counts or ()))
        self.frozen_depth_balance = frozen_depth_balance

    def wse_weight(self, epoch: int) -> float:
        target = float(self.config["lambda_wse"])
        if str(self.config.get("wse_mode", "absolute_laplacian")) == "absolute_laplacian" and "lambda_absolute_laplacian" in self.config:
            target = self.lambda_absolute_laplacian
        start = int(self.config["wse_start_epoch"])
        warmup = max(1, int(self.config["wse_warmup_epochs"]))
        if epoch < start:
            return 0.0
        return target * min(1.0, (epoch - start + 1) / warmup)

    def physics_weight(self, epoch: int) -> float:
        """Schedule the explicitly local S1+terrain weak-physics candidate."""

        target = float(
            self.config.get("lambda_phys", self.config.get("lambda_physics", 0.0))
        )
        if target < 0.0:
            raise ValueError("lambda_phys must be nonnegative")
        start = int(
            self.config.get(
                "phys_start_epoch", self.config.get("physics_start_epoch", 15)
            )
        )
        warmup = int(
            self.config.get(
                "phys_warmup_epochs", self.config.get("physics_warmup_epochs", 10)
            )
        )
        if start < 0 or warmup < 0:
            raise ValueError("physics start epoch and warmup epochs must be nonnegative")
        if epoch < start or target == 0.0:
            return 0.0
        return target if warmup == 0 else target * min(1.0, (epoch - start + 1) / warmup)

    def scheduled_weight(self, name: str, epoch: int) -> float:
        target = float(self.config.get(f"lambda_{name}", 0.0))
        start = int(self.config.get(f"{name}_start_epoch", 0))
        warmup = int(self.config.get(f"{name}_warmup_epochs", 0))
        if epoch < start:
            return 0.0
        if warmup <= 0:
            weight = target
        else:
            weight = target * min(1.0, (epoch - start + 1) / warmup)
        if name == "auxiliary":
            decay = str(self.config.get("auxiliary_decay", "constant"))
            decay_start = float(self.config.get("auxiliary_decay_start_fraction", 0.2))
            decay_end = float(self.config.get("auxiliary_decay_end_fraction", 0.5))
            total_epochs = max(1.0, float(self.config.get("schedule_total_epochs", self.config.get("epochs", 1))))
            fraction = float(epoch) / total_epochs
            if decay in {"linear", "cosine"} and fraction > decay_start:
                progress = min(1.0, (fraction - decay_start) / max(decay_end - decay_start, 1e-6))
                multiplier = 1.0 - progress if decay == "linear" else 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
                weight *= multiplier
        return weight

    def forward(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any], epoch: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        label = batch["label"]
        validity = batch["validity"]
        # Every depth-bearing objective receives the exact same output-aware
        # supervision domain.  Do not derive a second mask inside individual
        # loss terms.
        positive = canonical_positive_mask_from_batch(batch)
        aggregation_mode = str(self.config.get("supervised_reduction", "auto"))
        event_independent = aggregation_mode in {
            "pixel_micro",
            "depth_bin_macro",
            "sample_depth_bin",
        }
        # Pixel-first deployment is invariant to event labels.
        events = None if event_independent else _event_ids(batch)
        auxiliary_aggregation = (
            "pixel_micro"
            if aggregation_mode in {"pixel_micro", "depth_bin_macro"}
            else "event_macro"
        )
        effective_unc = self.scheduled_weight("unc", epoch)
        effective_gradient = self.scheduled_weight("gradient", epoch)
        effective_auxiliary = self.scheduled_weight("auxiliary", epoch)
        effective_kan = self.scheduled_weight("kan", epoch)
        effective_tail = self.scheduled_weight("tail", epoch)
        effective_wse = self.wse_weight(epoch)
        effective_physics = self.physics_weight(epoch)
        zero = label.sum() * 0.0
        lambda_final = float(self.config["lambda_final"])
        prediction = outputs.get("conditional_depth", outputs["positive_depth"])
        if str(self.config.get("objective_mode", "task_adaptive")) != "task_adaptive":
            raise ValueError("Only the task-adaptive production objective is supported")
        components = task_adaptive_positive_depth_loss(
            prediction, label, positive, self.train_depth_bins,
            beta_m=float(self.config.get("depth_huber_beta_m", 0.5)),
            log_weight=float(self.config.get("lambda_log", 0.15)),
            balance=bool(self.config.get("soft_depth_balance", True)),
            under_alpha=float(self.config.get("tail_underprediction_alpha", 0.0)),
            under_min_m=float(self.config.get("depth_underprediction_min_m", 0.48)),
            balance_alpha=float(self.config.get("soft_depth_balance_alpha", 0.5)),
            balance_tau=float(self.config.get("soft_depth_balance_tau", 10.0)),
            train_bin_counts=self.train_depth_bin_counts or None,
            frozen_depth_balance=self.frozen_depth_balance,
            log_beta=float(self.config.get("log_depth_huber_beta", 1.0)),
        )
        # Optional metric-alignment terms preserve the robust task-adaptive
        # objective while exposing the three deployment views used for model
        # selection.  In particular, direct RMSE optimization gives rare large
        # errors a proportional gradient without the unbounded scale of a raw
        # sum-of-squares objective, and event MAE prevents large tiled events
        # from monopolizing every update.
        absolute_error = (prediction - label).abs()
        squared_error = (prediction - label).square()
        metric_pixel_mae = masked_micro_mean(absolute_error, positive)
        metric_pixel_mse = masked_micro_mean(squared_error, positive)
        metric_pixel_rmse = torch.sqrt(metric_pixel_mse.clamp_min(1.0e-12))
        metric_event_mae = event_macro_masked_mean(
            absolute_error, positive, _event_ids(batch)
        )
        metric_event_rmse = event_macro_root_mean_square(
            prediction - label, positive, _event_ids(batch)
        )
        metric_event_hierarchical_mae = event_depth_hierarchical_macro_mean(
            absolute_error,
            label,
            positive,
            _event_ids(batch),
            self.primary_depth_bins,
            self.train_depth_bins,
        )
        metric_depth_bin_rmse = depth_bin_macro_root_mean_square(
            prediction - label,
            label,
            positive,
            self.train_depth_bins,
        )
        metric_event_hierarchical_rmse = (
            event_depth_hierarchical_macro_root_mean_square(
                prediction - label,
                label,
                positive,
                _event_ids(batch),
                self.primary_depth_bins,
                self.train_depth_bins,
            )
        )
        components["metric_pixel_mae"] = metric_pixel_mae
        components["metric_pixel_rmse"] = metric_pixel_rmse
        components["metric_event_mae"] = metric_event_mae
        components["metric_event_rmse"] = metric_event_rmse
        components["metric_event_hierarchical_mae"] = (
            metric_event_hierarchical_mae
        )
        components["metric_depth_bin_rmse"] = metric_depth_bin_rmse
        components["metric_event_hierarchical_rmse"] = (
            metric_event_hierarchical_rmse
        )
        if float(self.config.get("lambda_depth_bias", 0.0)) != 0.0:
            bias_beta = float(self.config.get("depth_bias_beta_m", 0.1))
            bias_reduction = str(
                self.config.get("depth_bias_reduction", "pixel_micro")
            )
            if bias_reduction == "pixel_micro":
                signed_bias = masked_micro_mean(prediction - label, positive)
                components["depth_bias"] = torch.nn.functional.smooth_l1_loss(
                    signed_bias,
                    torch.zeros_like(signed_bias),
                    beta=bias_beta,
                )
            elif bias_reduction == "sample_macro":
                components["depth_bias"] = event_mean_bias_loss(
                    prediction,
                    label,
                    positive,
                    None,
                    beta=bias_beta,
                )
            elif bias_reduction == "event_macro":
                components["depth_bias"] = event_mean_bias_loss(
                    prediction,
                    label,
                    positive,
                    _event_ids(batch),
                    beta=bias_beta,
                )
            elif bias_reduction == "event_depth_hierarchical":
                components["depth_bias"] = event_depth_hierarchical_bias_loss(
                    prediction,
                    label,
                    positive,
                    _event_ids(batch),
                    self.primary_depth_bins,
                    self.train_depth_bins,
                    bias_beta,
                )
            else:
                raise ValueError(
                    "depth_bias_reduction must be pixel_micro, sample_macro, "
                    "event_macro, or event_depth_hierarchical"
                )
        exceedance = (
            event_depth_exceedance_loss(
                outputs["depth"], label, positive, events, self.train_depth_bins,
                float(self.config.get("depth_exceedance_temperature_m", 0.1)),
                auxiliary_aggregation,
            ) if float(self.config.get("lambda_depth_exceedance", 0.0)) != 0.0 else zero
        )
        components["depth_exceedance"] = exceedance
        tail_threshold = self.config.get("tail_threshold_m")
        if tail_threshold is None:
            tail_threshold = self.train_depth_bins[-2] if len(self.train_depth_bins) >= 3 else 0.0
        tail = (
            tail_underprediction_loss(
                prediction, label, positive, float(tail_threshold),
                float(self.config.get("tail_margin_m", 0.15)),
                float(self.config.get("tail_huber_beta_m", 0.25)),
                "pixel_micro" if aggregation_mode in {"auto", "pixel_micro", "depth_bin_macro"} else "sample_macro",
            ) if effective_tail != 0.0 else zero
        )
        components["tail"] = tail
        uncertainty = (
            laplace_nll_loss(
                outputs["depth"] if bool(self.config.get("uncertainty_backprop_to_depth", True)) else outputs["depth"].detach(),
                label, outputs["uncertainty_scale"], positive, events,
                self.train_depth_bins, self.primary_depth_bins, aggregation_mode,
            ) if effective_unc != 0.0 else zero
        )
        components["uncertainty"] = uncertainty
        gradient = (
            masked_gradient_consistency_loss(
                outputs.get("conditional_depth", outputs["depth"]), label, positive,
                float(self.config.get("gradient_huber_beta_m", 0.1)),
            ) if effective_gradient != 0.0 else zero
        )
        components["gradient"] = gradient
        auxiliary, auxiliary_terms = (
            auxiliary_depth_loss(
                outputs.get("auxiliary_depths", ()), label, positive,
                self.config.get("auxiliary_depth_weights", ()),
                # The auxiliary head is a fixed supporting objective.  Keep its
                # transition independent from the primary-depth beta so a
                # primary beta ablation remains a genuine one-variable test.
                float(
                    self.config.get(
                        "auxiliary_huber_beta_m",
                        self.config.get("depth_huber_beta_m", 1.0),
                    )
                ),
            ) if effective_auxiliary != 0.0 else (zero, [])
        )
        components["auxiliary"] = auxiliary
        for index, value in enumerate(auxiliary_terms):
            components[f"auxiliary_{index}"] = value
        sensor_valid = validity["s1_valid"]
        day_difference = batch["reliability"].new_zeros(
            batch["reliability"].shape[0], 1, *batch["reliability"].shape[-2:]
        )
        wse_mode = str(self.config.get("wse_mode", "absolute_laplacian"))
        if effective_wse == 0.0:
            wse = zero
        elif wse_mode == "reference_gated_gradient":
            wse = reference_gated_wse_gradient_loss(
                outputs["depth"],
                label,
                outputs["physical_features"]["z_hyd"],
                positive,
                validity["dem_valid"],
                sensor_valid,
                day_difference,
                events,
                float(self.config["wse_time_sigma"]),
                float(self.config["wse_reference_sigma_m"]),
                float(self.config["wse_terrain_sigma_m"]),
                float(self.config["wse_gradient_beta_m"]),
                auxiliary_aggregation,
            )
        elif wse_mode == "terrain_order":
            wse, order_diag = gated_terrain_order_loss(
                outputs["conditional_depth"], outputs["physical_features"]["physics_elevation"],
                positive, validity["dem_valid"], sensor_valid,
                depth_order_tolerance_m=float(self.config.get("depth_order_tolerance_m", 0.02)),
                huber_beta_m=float(self.config.get("terrain_order_huber_beta_m", 0.05)),
                terrain_step_min_m=float(self.config.get("terrain_step_min_m", 0.02)),
                terrain_step_max_m=float(self.config.get("terrain_step_max_m", 0.75)),
                terrain_sigma_m=float(self.config.get("terrain_sigma_m", 0.75)),
                relief=outputs["physical_features"].get("local_relief"),
                relief_sigma_m=float(self.config.get("terrain_relief_sigma_m", 12.0)),
                return_diagnostics=True,
            )
            components["terrain_order_violation_fraction"] = order_diag["violation_fraction"]
            components["terrain_order_violation_magnitude"] = order_diag["violation_magnitude"]
        elif wse_mode == "absolute_laplacian":
            wse = weak_wse_laplacian_loss(
                outputs["depth"],
                outputs["physical_features"]["z_hyd"],
                positive,
                validity["dem_valid"],
                sensor_valid,
                day_difference,
                float(self.config["wse_time_sigma"]),
            )
        else:
            raise ValueError(
                "loss.wse_mode must be 'absolute_laplacian', "
                f"'reference_gated_gradient', or 'terrain_order', got {wse_mode!r}"
            )
        components["wse"] = wse
        if effective_physics == 0.0:
            physics = zero
            physics_diagnostics = {
                "active_pair_fraction": zero,
                "active_pair_count": zero,
                "candidate_pair_count": zero,
                "mean_violation_m": zero,
                "p90_violation_m": zero,
                "mean_sar_compatibility": zero,
                "mean_barrier_weight": zero,
                "mean_complexity_weight": zero,
                "pair_weight_sum": zero,
            }
        else:
            physical = outputs["physical_features"]
            physics, physics_diagnostics = weak_physics_pair_loss(
                prediction,
                physical["physics_elevation"],
                physical["dsm_elevation"],
                physical["z_ground_proxy"],
                physical["local_relief"],
                positive,
                validity["dem_valid"],
                validity["s1_valid"],
                batch["s1_change"],
                mode=str(self.config.get("physics_mode", "terrain_order_margin")),
                elevation_tolerance_m=float(
                    self.config.get("physics_elevation_tolerance_m", 0.05)
                ),
                maximum_elevation_step_m=float(
                    self.config.get("physics_maximum_elevation_step_m", 0.75)
                ),
                depth_tolerance_m=float(
                    self.config.get("physics_depth_tolerance_m", 0.02)
                ),
                order_softplus_temperature_m=float(
                    self.config.get("physics_order_softplus_temperature_m", 0.05)
                ),
                allowed_depth_jump_m=float(
                    self.config.get("physics_allowed_depth_jump_m", 0.25)
                ),
                barrier_sigma_m=float(
                    self.config.get("physics_barrier_sigma_m", 0.75)
                ),
                complexity_sigma_m=float(
                    self.config.get("physics_complexity_sigma_m", 12.0)
                ),
                maximum_complexity_m=float(
                    self.config.get("physics_maximum_complexity_m", 12.0)
                ),
                sar_compatibility_sigma=float(
                    self.config.get("physics_sar_compatibility_sigma", 1.0)
                ),
                wse_consistency_tolerance_m=float(
                    self.config.get("physics_wse_consistency_tolerance_m", 0.10)
                ),
                wse_consistency_softplus_temperature_m=float(
                    self.config.get(
                        "physics_wse_consistency_softplus_temperature_m", 0.05
                    )
                ),
                wse_consistency_maximum_barrier_m=float(
                    self.config.get("physics_wse_consistency_maximum_barrier_m", 3.0)
                ),
            )
        components["physics"] = physics
        for name, value in physics_diagnostics.items():
            components[f"physics_{name}"] = value
        if effective_kan != 0.0:
            kan_magnitude = outputs.get("graph_diagnostics", {}).get(
                "kan_coefficient_magnitude", zero
            )
            kan_smoothness = outputs.get("graph_diagnostics", {}).get(
                "kan_coefficient_smoothness", zero
            )
        else:
            kan_magnitude = zero
            kan_smoothness = zero
        components["kan_magnitude"] = kan_magnitude
        components["kan_smoothness"] = kan_smoothness
        components["kan_monotonicity"] = outputs.get("graph_diagnostics", {}).get("kan_monotonicity", zero)
        components["kan_curve_smoothness"] = outputs.get("graph_diagnostics", {}).get("kan_curve_smoothness", zero)
        total = (
            float(self.config["lambda_depth"]) * components["depth"]
            + float(self.config.get("lambda_metric_pixel_mae", 0.0))
            * metric_pixel_mae
            + float(self.config.get("lambda_metric_pixel_rmse", 0.0))
            * metric_pixel_rmse
            + float(self.config.get("lambda_metric_event_mae", 0.0))
            * metric_event_mae
            + float(self.config.get("lambda_metric_event_rmse", 0.0))
            * metric_event_rmse
            + float(
                self.config.get("lambda_metric_event_hierarchical_mae", 0.0)
            )
            * metric_event_hierarchical_mae
            + float(self.config.get("lambda_metric_depth_bin_rmse", 0.0))
            * metric_depth_bin_rmse
            + float(
                self.config.get("lambda_metric_event_hierarchical_rmse", 0.0)
            )
            * metric_event_hierarchical_rmse
            + float(self.config.get("lambda_depth_bias", 0.0))
            * components["depth_bias"]
            + float(self.config.get("lambda_depth_exceedance", 0.0)) * exceedance
            + effective_tail * tail
            + effective_unc * uncertainty
            + effective_gradient * gradient
            + effective_auxiliary * auxiliary
            + effective_wse * wse
            + effective_physics * physics
            + effective_kan * (kan_magnitude + kan_smoothness)
            + float(self.config.get("lambda_kan_mono", 0.0)) * components["kan_monotonicity"]
            + float(self.config.get("lambda_kan_smooth", 0.0)) * components["kan_curve_smoothness"]
        )
        components["total"] = total
        components["wse_effective_weight"] = total.new_tensor(effective_wse)
        components["physics_effective_weight"] = total.new_tensor(effective_physics)
        components["unc_effective_weight"] = total.new_tensor(effective_unc)
        components["gradient_effective_weight"] = total.new_tensor(effective_gradient)
        components["auxiliary_effective_weight"] = total.new_tensor(effective_auxiliary)
        components["kan_effective_weight"] = total.new_tensor(effective_kan)
        components["tail_effective_weight"] = total.new_tensor(effective_tail)
        mask_counts = supervision_mask_counts(batch)
        components.update(mask_counts)
        # Compatibility aliases retained for existing CSV consumers.  Their value
        # is now the canonical count rather than label validity alone.
        components["positive_pixels"] = components["positive_supervision_pixels"]
        return total, components
