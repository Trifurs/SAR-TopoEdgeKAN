#!/usr/bin/env python3
"""Evaluate SAR-TopoEdgeKAN with partial-positive metrics."""

from __future__ import annotations

from copy import deepcopy
from itertools import islice
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.preprocessing import RELIABILITY_NAMES, RobustNormalizer, resolve_depth_stratification_bins, reliability_spec_for_mode
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import ReliabilitySpec
from datasets.supervision_masks import (
    CANONICAL_POSITIVE_MASK,
    canonical_positive_mask_from_batch,
    supervision_mask_counts,
    validate_supervision_config,
)


def _batch_reliability_names(batch: dict[str, Any]) -> tuple[str, ...]:
    values = batch.get("reliability_names")
    if isinstance(values, tuple) and values and all(isinstance(item, str) for item in values):
        return values
    if isinstance(values, list) and values and all(isinstance(item, str) for item in values):
        return tuple(values)
    if isinstance(values, (list, tuple)) and values and all(
        isinstance(item, (list, tuple)) and item and isinstance(item[0], str)
        for item in values
    ):
        return tuple(str(item[0]) for item in values)
    metadata = batch.get("metadata")
    if isinstance(metadata, dict):
        return _batch_reliability_names({"reliability_names": metadata.get("reliability_names")})
    return RELIABILITY_NAMES
from losses.composite_loss import CompositeFloodDepthLoss
from metrics.aggregator import EvaluationAggregator
from metrics.physical_metrics import (
    local_wse_laplacian,
    local_wse_laplacian_reference_error,
    prediction_continuity_high_relief,
    reference_gated_wse_gradient_mae,
    terrain_order_violation_metrics,
)
from utils.checkpoint import load_checkpoint
from utils.config import jsonable_config, load_config
from utils.efficiency import (
    InferenceEfficiency,
    checkpoint_size_metrics,
    model_size_metrics,
)
from utils.logging import write_rows
from utils.misc import (
    atomic_write_json,
    move_to_device,
)
from utils.raster_io import write_geotiff
from utils.registry import build_model


from utils.visualization import save_prediction_panel
from utils.graph_metadata import resolved_graph_identity, runtime_graph_identity
from datasets.contract import DatasetContract, sha256_file
from datasets.band_selection import resolve_band_spec


def dataset_fingerprint(config: dict[str, Any]) -> dict[str, str]:
    contract = DatasetContract.load(config["dataset"]["contract"])
    fingerprint = {
        "contract_sha256": contract.hash,
        "manifest_sha256": sha256_file(contract.manifest_path),
        "normalization_sha256": sha256_file(Path(config["dataset"]["train_stats"])),
    }
    input_spec = ModelInputSpec.from_config(config)
    fingerprint["model_input_spec_sha256"] = input_spec.sha256
    fingerprint["active_groups_sha256"] = input_spec.active_groups_sha256
    fingerprint["reliability_spec_sha256"] = ReliabilitySpec.from_mode(input_spec.mode).sha256
    edge_stats = config.get("model", {}).get("graph_edge_stats")
    if edge_stats is not None:
        path = Path(edge_stats).expanduser().resolve(strict=True)
        fingerprint["graph_edge_stats_sha256"] = sha256_file(path)
    return fingerprint


def embed_source_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    """Place audited key-source hashes into every saved resolved configuration."""

    validate_supervision_config(config)
    contract = DatasetContract.load(config["dataset"]["contract"])
    config["dataset"]["source_file_sha256"] = dict(contract.payload["key_file_sha256"])
    config["dataset"]["contract_sha256"] = contract.hash
    config["dataset"]["normalization_sha256"] = sha256_file(
        Path(config["dataset"]["train_stats"])
    )
    config["dataset"]["resolved_model_bands"] = resolve_band_spec(
        config, contract
    ).as_dict()
    input_spec = ModelInputSpec.from_config(config)
    config["dataset"]["resolved_model_input_spec"] = input_spec.as_dict()
    config["dataset"]["resolved_reliability_schema"] = list(
        reliability_spec_for_mode(input_spec.mode).names
    )
    edge_stats = config.get("model", {}).get("graph_edge_stats")
    if edge_stats is not None:
        path = Path(edge_stats).expanduser().resolve(strict=True)
        config["model"]["graph_edge_stats_sha256"] = sha256_file(path)
    graph_identity = resolved_graph_identity(config)
    if graph_identity is not None:
        config["model"]["resolved_graph_identity"] = graph_identity
    return config


def metadata_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        selected = value[index].detach().cpu()
        return selected.item() if selected.ndim == 0 else selected.tolist()
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, torch.Tensor) for item in value):
            return tuple(metadata_item(item, index) for item in value)
        return value[index]
    return value


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_depth_bins: list[float],
    *,
    primary_depth_bins: list[float] | None = None,
    criterion: CompositeFloodDepthLoss | None = None,
    max_batches: int | None = None,
    output_dir: Path | None = None,
    save_predictions: bool = False,
    progress: bool = True,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    input_spec: ModelInputSpec | None = None,
    validity_mask: str | None = None,
    measure_efficiency: bool = False,
    efficiency_warmup_batches: int | None = None,
    progress_label: str = "Evaluate",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    aggregator = EvaluationAggregator(train_depth_bins, primary_depth_bins)
    loss_values: list[float] = []
    component_values: dict[str, list[float]] = {}
    mask_totals = {
        "valid_depth_mask_pixels": 0,
        "output_valid_pixels": 0,
        "positive_supervision_pixels": 0,
        "positive_excluded_by_output_valid_pixels": 0,
    }
    efficiency = InferenceEfficiency(
        device,
        enabled=measure_efficiency,
        warmup_batches=efficiency_warmup_batches,
    )
    disable_progress = (not progress) or os.environ.get("FLOOD_DEPTH_DISABLE_TQDM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    selected_batches = loader if max_batches is None else islice(loader, max_batches)
    progress_total = len(loader) if max_batches is None else min(len(loader), max_batches)
    iterator = tqdm(
        selected_batches,
        total=progress_total,
        desc=progress_label,
        leave=False,
        disable=disable_progress,
    )
    for batch_index, cpu_batch in enumerate(iterator):
        batch = move_to_device(cpu_batch, device)
        batch_mask_counts = supervision_mask_counts(batch)
        for name in mask_totals:
            mask_totals[name] += int(batch_mask_counts[name].detach().cpu())
        reliability_names = _batch_reliability_names(cpu_batch)
        reliability_index = {name: index for index, name in enumerate(reliability_names)}
        input_model = model.module if hasattr(model, "module") else model
        model_inputs = prepare_model_inputs(
            batch,
            input_spec,
            include_valid_depth_mask=bool(
                getattr(input_model, "known_flood_range_enabled", False)
            ),
        )
        inference_started = efficiency.start()
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled and device.type == "cuda",
            dtype=amp_dtype,
        ):
            outputs = model(model_inputs)
        efficiency.stop(
            inference_started,
            samples=int(outputs["depth"].shape[0]),
            output_pixels=int(outputs["depth"].numel()),
        )
        if "physical_features" not in outputs and hasattr(input_model, "discussion_model_config"):
            from models.discuss.inputs import evaluation_terrain_features

            # Shared evaluation diagnostics are not part of comparator inference.
            outputs["physical_features"] = evaluation_terrain_features(
                model_inputs,
                input_model.discussion_raw_names,
                input_model.discussion_model_config,
            )
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled and device.type == "cuda",
            dtype=amp_dtype,
        ):
            if criterion is not None:
                loss, components = criterion(outputs, batch)
                loss_values.append(float(loss.detach().cpu()))
                for name, value in components.items():
                    component_values.setdefault(name, []).append(float(value.detach().cpu()))
        batch_size = outputs["depth"].shape[0]
        for sample_index in range(batch_size):
            prediction = outputs["depth"][sample_index].detach().float().cpu().numpy()
            conditional_depth = (
                outputs["conditional_depth"][sample_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            target = batch["label"][sample_index].detach().float().cpu().numpy()
            positive_mask = (
                batch["masks"]["valid_depth_mask"][sample_index].detach().cpu().numpy() > 0.5
            )
            canonical_mask = (
                canonical_positive_mask_from_batch(batch)[sample_index]
                .detach()
                .cpu()
                .numpy()
            )
            metric_valid_mask = canonical_mask
            if validity_mask not in {None, CANONICAL_POSITIVE_MASK, "output_valid"}:
                if validity_mask == "common_s1":
                    metric_valid_mask = positive_mask & (
                        batch["validity"]["s1_event_support"][sample_index].detach().cpu().numpy() > 0.5
                    ) & (
                        batch["validity"]["dem_valid"][sample_index].detach().cpu().numpy() > 0.5
                    )
                elif validity_mask not in batch["validity"]:
                    raise KeyError(f"Unknown evaluation validity mask: {validity_mask!r}")
                else:
                    metric_valid_mask = positive_mask & (
                        batch["validity"][validity_mask][sample_index].detach().cpu().numpy() > 0.5
                    )
            sample_id = str(metadata_item(cpu_batch["metadata"]["sample_id"], sample_index))
            event_id = str(
                metadata_item(cpu_batch["metadata"]["source_event_id"], sample_index)
            )
            day_index = reliability_index.get("absolute_normalized_sensor_day_difference")
            sensor_day = (
                batch["reliability"][sample_index, day_index:day_index + 1].detach().cpu().numpy()
                if day_index is not None else np.zeros_like(target)
            )
            observation_indices = [
                reliability_index[name]
                for name in ("s1_event_observation_count_z",)
                if name in reliability_index
            ]
            observation = batch["reliability"][sample_index, observation_indices].mean(dim=0, keepdim=True).detach().cpu().numpy() if observation_indices else np.zeros_like(sensor_day)
            row = aggregator.add(
                sample_id,
                event_id,
                prediction,
                target,
                metric_valid_mask,
                sensor_day,
                observation,
            )
            z_hyd = outputs["physical_features"]["z_hyd"][sample_index].detach().float().cpu().numpy()
            local_relief = (
                outputs["physical_features"]["local_relief"][sample_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            output_valid_np = (
                batch["validity"]["output_valid"][sample_index].detach().cpu().numpy() > 0.5
            )
            positive_output_valid = metric_valid_mask & output_valid_np
            row["evaluation_valid_pixel_fraction"] = float(np.mean(metric_valid_mask))
            day_difference_np = sensor_day
            row["positive_region_wse_laplacian"] = local_wse_laplacian(
                prediction, z_hyd, positive_output_valid
            )
            row["reference_positive_region_wse_laplacian"] = local_wse_laplacian(
                target, z_hyd, positive_output_valid
            )
            row["positive_region_wse_laplacian_reference_error"] = (
                local_wse_laplacian_reference_error(
                    prediction, target, z_hyd, positive_output_valid
                )
            )
            physics_config = criterion.config if criterion is not None else {}
            row["reference_gated_wse_gradient_mae"] = (
                reference_gated_wse_gradient_mae(
                    prediction,
                    target,
                    z_hyd,
                    positive_output_valid,
                    day_difference_np,
                    float(physics_config.get("wse_time_sigma", 0.25)),
                    float(physics_config.get("wse_reference_sigma_m", 0.12)),
                    float(physics_config.get("wse_terrain_sigma_m", 0.75)),
                )
            )
            terrain_order = terrain_order_violation_metrics(
                prediction,
                z_hyd,
                positive_output_valid,
                day_difference_np,
                float(physics_config.get("wse_time_sigma", 0.25)),
                float(physics_config.get("terrain_order_min_step_m", 0.02)),
                float(physics_config.get("terrain_order_max_step_m", 0.75)),
            )
            row["terrain_order_violation_mae"] = terrain_order["mae"]
            row["terrain_order_violation_fraction"] = terrain_order["fraction"]
            row["high_relief_prediction_continuity"] = prediction_continuity_high_relief(
                prediction, local_relief, output_valid_np
            )
            if save_predictions and output_dir is not None:
                sample_dir = output_dir / sample_id
                crs = str(metadata_item(cpu_batch["metadata"]["crs"], sample_index))
                transform = metadata_item(cpu_batch["metadata"]["transform"], sample_index)
                output_valid = (
                    batch["validity"]["output_valid"][sample_index].detach().cpu().numpy()
                    > 0.5
                )
                write_geotiff(
                    sample_dir / "predicted_depth_m.tif",
                    prediction,
                    crs=crs,
                    transform=transform,
                    valid_mask=output_valid,
                    descriptions=["predicted_depth_m"],
                )
                write_geotiff(
                    sample_dir / "conditional_depth_m.tif",
                    conditional_depth,
                    crs=crs,
                    transform=transform,
                    valid_mask=output_valid,
                    descriptions=["conditional_depth_m"],
                )
                save_prediction_panel(
                    sample_dir / "prediction_panel.png",
                    s1_change=cpu_batch["s1_change"][sample_index, 0].numpy(),
                    dsm=cpu_batch["terrain_raw"][sample_index, 0].numpy(),
                    target=target[0],
                    prediction=prediction[0],
                    valid_label=canonical_mask[0],
                )
                atomic_write_json(sample_dir / "metrics.json", row)
    summary, sample_rows, event_rows, bin_rows = aggregator.summarize()
    unwrapped = model.module if hasattr(model, "module") else model
    summary["depth_output_semantics"] = str(
        getattr(
            getattr(unwrapped, "heads", None),
            "depth_output_semantics",
            getattr(unwrapped, "depth_output_semantics", "unknown"),
        )
    )
    summary["evaluation_validity_mask"] = (
        CANONICAL_POSITIVE_MASK
        if validity_mask in {None, CANONICAL_POSITIVE_MASK, "output_valid"}
        else str(validity_mask)
    )
    summary.update(mask_totals)
    summary["positive_excluded_by_output_valid_fraction"] = (
        float(mask_totals["positive_excluded_by_output_valid_pixels"])
        / max(float(mask_totals["valid_depth_mask_pixels"]), 1.0)
    )
    graph_identity = runtime_graph_identity(unwrapped)
    if graph_identity is not None:
        summary["graph_identity"] = graph_identity
    if loss_values:
        summary["objective_mean"] = float(np.mean(loss_values))
        for name, values in component_values.items():
            summary[f"objective_{name}_mean"] = float(np.mean(values))
    summary.update(efficiency.summary())
    return summary, sample_rows, event_rows, bin_rows


def run_evaluation_from_config(
    config_value: Mapping[str, Any],
    checkpoint_path: Path,
    split: str,
    device_name: str,
    output_dir: Path,
    save_predictions: bool,
    max_batches: int | None = None,
    weights: str = "raw",
    validity_mask: str | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Evaluate SAR-TopoEdgeKAN from an immutable resolved training config."""

    evaluation_started = time.perf_counter()
    config = embed_source_fingerprints(deepcopy(dict(config_value)))
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        config["training"]["num_workers"] = int(num_workers)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    input_spec = ModelInputSpec.from_config(config)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], split,
        band_spec=band_spec,
        input_spec=input_spec,
        minimum_event_band_fraction=float(
            config["dataset"].get("minimum_event_band_fraction", 1.0)
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    workers = int(config["training"]["num_workers"])
    loader_options: dict[str, Any] = {
        "batch_size": int(config["training"]["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "persistent_workers": bool(
            config["training"].get("persistent_workers", False)
        )
        and workers > 0,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_options["prefetch_factor"] = int(
            config["training"].get("prefetch_factor", 2)
        )
    loader = DataLoader(dataset, **loader_options)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        # Keep non-model checkpoint state (optimizer, RNG, scheduler) off the
        # accelerator so peak memory describes inference rather than loading.
        map_location="cpu",
    )
    if weights != "raw":
        raise ValueError("Only raw model weights are supported")
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    criterion = CompositeFloodDepthLoss(
        config["loss"], depth_bins, normalizer.train_depth_bins,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, samples, events, bins = evaluate_loader(
        model,
        loader,
        device,
        depth_bins,
        primary_depth_bins=normalizer.train_depth_bins,
        criterion=criterion,
        max_batches=max_batches,
        output_dir=output_dir,
        save_predictions=save_predictions,
        amp_enabled=bool(config["training"].get("amp", False)),
        amp_dtype=(
            torch.bfloat16
            if str(config["training"].get("amp_dtype", "auto")) in {"auto", "bfloat16"}
            and device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        input_spec=input_spec,
        validity_mask=validity_mask or CANONICAL_POSITIVE_MASK,
        progress=bool(config["logging"].get("progress_bar", False)),
        measure_efficiency=True,
        efficiency_warmup_batches=int(
            config.get("runtime", {}).get("test", {}).get(
                "cuda_warmup_batches", 1
            )
        )
        if device.type == "cuda"
        else 0,
        progress_label=split.capitalize(),
    )
    summary["checkpoint_epoch"] = checkpoint_epoch
    summary["weights"] = weights
    summary["model_family"] = "deep_learning"
    summary["efficiency_precision"] = (
        str(
            torch.bfloat16
            if str(config["training"].get("amp_dtype", "auto"))
            in {"auto", "bfloat16"}
            and device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            else torch.float16
        ).replace("torch.", "")
        if bool(config["training"].get("amp", False)) and device.type == "cuda"
        else "float32"
    )
    summary["efficiency_memory_format"] = "contiguous"
    summary.update(model_size_metrics(model))
    summary.update(checkpoint_size_metrics(checkpoint_path))
    summary["efficiency_end_to_end_seconds"] = float(
        time.perf_counter() - evaluation_started
    )
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "resolved_config.json", jsonable_config(config))
    write_rows(output_dir / "metrics_by_sample.csv", samples)
    write_rows(output_dir / "metrics_by_event.csv", events)
    write_rows(output_dir / "metrics_by_train_depth_bin.csv", bins)
    return summary


def run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    device_name: str,
    output_dir: Path,
    save_predictions: bool,
    max_batches: int | None = None,
    weights: str = "raw",
    validity_mask: str | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    return run_evaluation_from_config(
        load_config(config_path),
        checkpoint_path,
        split,
        device_name,
        output_dir,
        save_predictions,
        max_batches,
        weights,
        validity_mask,
        num_workers,
    )


if __name__ == "__main__":
    raise SystemExit("Use `python evaluate.py <model-config.xml>` from the project root.")
