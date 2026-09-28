"""Micro, sample-macro, event-macro and train-bin metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from metrics.depth_metrics import DEPTH_METRIC_NAMES, depth_metrics, prefixed
from metrics.uncertainty_metrics import uncertainty_metrics


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else float("nan")


class EvaluationAggregator:
    def __init__(
        self,
        train_depth_bins: list[float],
        primary_depth_bins: list[float] | None = None,
    ) -> None:
        self.train_depth_bins = sorted(set(float(value) for value in train_depth_bins))
        self.primary_depth_bins = sorted(
            set(
                float(value)
                for value in (primary_depth_bins or train_depth_bins)
            )
        )
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        sample_id: str,
        event_id: str,
        prediction: np.ndarray,
        target: np.ndarray,
        scale: np.ndarray,
        valid_mask: np.ndarray,
        day_difference: np.ndarray | None = None,
        observation_feature: np.ndarray | None = None,
    ) -> dict[str, Any]:
        mask = np.asarray(valid_mask).astype(bool).reshape(-1)
        prediction_flat = np.asarray(prediction).reshape(-1)[mask]
        target_flat = np.asarray(target).reshape(-1)[mask]
        scale_flat = np.asarray(scale).reshape(-1)[mask]
        metrics = depth_metrics(prediction_flat, target_flat)
        row: dict[str, Any] = {"sample_id": sample_id, "source_event_id": event_id, **metrics}
        if day_difference is not None:
            day = np.asarray(day_difference).reshape(-1)
            row["mean_sensor_day_difference"] = float(np.mean(day[mask])) if np.any(mask) else float("nan")
        if observation_feature is not None:
            obs = np.asarray(observation_feature).reshape(-1)
            row["mean_observation_feature"] = float(np.mean(obs[mask])) if np.any(mask) else float("nan")
        self.records.append(
            {
                "sample_id": sample_id,
                "event_id": event_id,
                "prediction": prediction_flat,
                "target": target_flat,
                "scale": scale_flat,
                "row": row,
            }
        )
        return row

    def summarize(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.records:
            raise RuntimeError("No evaluation records were added")
        predictions = np.concatenate([record["prediction"] for record in self.records])
        targets = np.concatenate([record["target"] for record in self.records])
        scales = np.concatenate([record["scale"] for record in self.records])
        summary: dict[str, Any] = prefixed(depth_metrics(predictions, targets), "pixel_micro_")
        sample_rows = [record["row"] for record in self.records]
        for metric in DEPTH_METRIC_NAMES:
            summary[f"sample_macro_{metric}"] = _nanmean(
                [float(row[metric]) for row in sample_rows]
            )

        by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            by_event[record["event_id"]].append(record)
        internal = self.train_depth_bins[1:-1] if len(self.train_depth_bins) > 2 else []
        primary_internal = (
            self.primary_depth_bins[1:-1]
            if len(self.primary_depth_bins) > 2
            else []
        )
        event_rows: list[dict[str, Any]] = []
        for event_id, records in sorted(by_event.items()):
            event_prediction = np.concatenate([record["prediction"] for record in records])
            event_target = np.concatenate([record["target"] for record in records])
            event_row = {
                "source_event_id": event_id,
                **depth_metrics(event_prediction, event_target),
            }
            if internal:
                typed_internal = np.asarray(internal, dtype=event_target.dtype)
                event_bin_ids = np.digitize(event_target, typed_internal, right=True)
                cell_metrics = [
                    depth_metrics(
                        event_prediction[event_bin_ids == depth_bin],
                        event_target[event_bin_ids == depth_bin],
                    )
                    for depth_bin in range(len(internal) + 1)
                    if np.any(event_bin_ids == depth_bin)
                ]
                for metric in DEPTH_METRIC_NAMES:
                    event_row[f"depth_bin_macro_{metric}"] = _nanmean(
                        [float(cell[metric]) for cell in cell_metrics]
                    )
                event_row["depth_bin_nonempty_cells"] = len(cell_metrics)

            typed_primary = np.asarray(primary_internal, dtype=event_target.dtype)
            primary_bin_ids = np.digitize(event_target, typed_primary, right=True)
            typed_refined = np.asarray(internal, dtype=event_target.dtype)
            refined_bin_ids = np.digitize(event_target, typed_refined, right=True)
            primary_metrics: list[dict[str, float]] = []
            hierarchical_nonempty_cells = 0
            for primary_bin in range(len(primary_internal) + 1):
                in_primary = primary_bin_ids == primary_bin
                if not np.any(in_primary):
                    continue
                refined_metrics = [
                    depth_metrics(
                        event_prediction[selected],
                        event_target[selected],
                    )
                    for refined_bin in range(len(internal) + 1)
                    if np.any(
                        selected := in_primary & (refined_bin_ids == refined_bin)
                    )
                ]
                hierarchical_nonempty_cells += len(refined_metrics)
                primary_metrics.append(
                    {
                        metric: _nanmean(
                            [float(cell[metric]) for cell in refined_metrics]
                        )
                        for metric in DEPTH_METRIC_NAMES
                    }
                )
            for metric in DEPTH_METRIC_NAMES:
                event_row[f"depth_hierarchical_macro_{metric}"] = _nanmean(
                    [float(cell[metric]) for cell in primary_metrics]
                )
            event_row["depth_hierarchical_nonempty_primary_groups"] = len(
                primary_metrics
            )
            event_row["depth_hierarchical_nonempty_cells"] = (
                hierarchical_nonempty_cells
            )
            event_rows.append(event_row)
        for metric in DEPTH_METRIC_NAMES:
            summary[f"event_macro_{metric}"] = _nanmean(
                [float(row[metric]) for row in event_rows]
            )
            summary[f"event_depth_bin_macro_{metric}"] = _nanmean(
                [float(row.get(f"depth_bin_macro_{metric}", row[metric])) for row in event_rows]
            )
            summary[f"event_depth_hierarchical_macro_{metric}"] = _nanmean(
                [
                    float(
                        row.get(f"depth_hierarchical_macro_{metric}", row[metric])
                    )
                    for row in event_rows
                ]
            )
        summary["event_depth_bin_nonempty_cells"] = int(
            sum(int(row.get("depth_bin_nonempty_cells", 1)) for row in event_rows)
        )
        summary["event_depth_hierarchical_nonempty_primary_groups"] = int(
            sum(
                int(row.get("depth_hierarchical_nonempty_primary_groups", 1))
                for row in event_rows
            )
        )
        summary["event_depth_hierarchical_nonempty_cells"] = int(
            sum(
                int(row.get("depth_hierarchical_nonempty_cells", 1))
                for row in event_rows
            )
        )
        # Both terms are event-level MAE in metres. Their equal-weight mean prevents
        # selection from improving depth strata by sacrificing whole-event accuracy,
        # or vice versa; pixel count never enters this checkpoint criterion.
        summary["event_hierarchical_composite_mae"] = 0.5 * (
            summary["event_macro_mae"]
            + summary["event_depth_hierarchical_macro_mae"]
        )
        # All four terms are metre-valued errors.  Their equal-weight mean is a
        # checkpoint-selection score aligned with pixel accuracy, large-error
        # control, whole-event fairness, and train-depth-stratum robustness.
        # R²/NSE need not be inserted separately because, for a fixed validation
        # target, their ordering is already determined by squared error.
        summary["balanced_composite_error_m"] = 0.25 * (
            summary["pixel_micro_mae"]
            + summary["pixel_micro_rmse"]
            + summary["event_macro_mae"]
            + summary["event_depth_hierarchical_macro_mae"]
        )
        summary.update(prefixed(uncertainty_metrics(predictions, targets, scales), "uncertainty_"))
        for diagnostic in (
            "positive_region_wse_laplacian",
            "reference_positive_region_wse_laplacian",
            "positive_region_wse_laplacian_reference_error",
            "reference_gated_wse_gradient_mae",
            "terrain_order_violation_mae",
            "terrain_order_violation_fraction",
            "high_relief_prediction_continuity",
        ):
            summary[f"physical_sample_macro_{diagnostic}"] = _nanmean(
                [float(row.get(diagnostic, float("nan"))) for row in sample_rows]
            )
        for feature, output_name in (
            ("mean_sensor_day_difference", "error_vs_sensor_day_difference_spearman"),
            ("mean_observation_feature", "error_vs_observation_feature_spearman"),
        ):
            x = np.asarray([row.get(feature, np.nan) for row in sample_rows], dtype=np.float64)
            y = np.asarray([row.get("mae", np.nan) for row in sample_rows], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            summary[f"physical_{output_name}"] = (
                float(spearmanr(x[finite], y[finite]).statistic)
                if np.count_nonzero(finite) >= 2
                and not np.all(x[finite] == x[finite][0])
                and not np.all(y[finite] == y[finite][0])
                else float("nan")
            )

        boundaries = [-np.inf, *internal, np.inf]
        typed_internal = np.asarray(internal, dtype=targets.dtype)
        global_bin_ids = np.digitize(targets, typed_internal, right=True)
        bin_rows: list[dict[str, Any]] = []
        for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            selected = global_bin_ids == index
            metrics = depth_metrics(predictions[selected], targets[selected])
            bin_rows.append(
                {
                    "bin": index,
                    "lower_train_boundary_m": lower,
                    "upper_train_boundary_m": upper,
                    **metrics,
                }
            )
        return summary, sample_rows, event_rows, bin_rows
