#!/usr/bin/env python3
"""Train SAR-TopoEdgeKAN with strict reproducibility."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
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
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from datasets.contract import DatasetContract
from datasets.band_selection import resolve_band_spec
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import resolve_depth_stratification_bins
from datasets.supervision_masks import (
    canonical_positive_mask_from_batch,
    validate_supervision_config,
)
from datasets.samplers import (
    BalancedRemainderBatchSampler,
    make_event_balanced_sampler,
)
from datasets.transforms import SynchronousAugment
from datasets.train_depth_calibration import collect_canonical_train_depths
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate_sar_topo_edge_kan import dataset_fingerprint, embed_source_fingerprints, evaluate_loader
from utils.checkpoint import (
    checkpoint_depth_output_semantics,
    load_checkpoint,
    save_checkpoint,
    training_identity_sha256,
)
from utils.config import jsonable_config, load_config
from utils.logging import (
    append_csv,
    configure_training_warning_filters,
    log_epoch_summary,
    log_training_header,
    setup_logging,
)
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model
from utils.seed import seed_everything, seed_worker
from utils.amp import resolve_amp
from utils.graph_metadata import resolved_graph_identity, runtime_graph_identity
from utils.optim import build_optimizer, build_scheduler
from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


LOGGER = logging.getLogger("train")


def normalize_accumulated_gradients(model: torch.nn.Module, sample_count: int) -> None:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)


def _training_diagnostic_values(
    outputs: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, float]:
    """Compact production observability values for CSV and TensorBoard logs."""

    graph = outputs.get("graph_diagnostics", {})
    sar = outputs.get("sar_diagnostics", {})
    fusion = outputs.get("fusion_diagnostics", {})
    positive = canonical_positive_mask_from_batch(batch)
    diagnostics = {
        "pair_valid_fraction_mean": sar.get("pair_valid_fraction_mean"),
        "pre_context_gate_mean": sar.get("pre_context_gate_mean"),
        "quality_mean": sar.get("quality_mean"),
        "sar_alignment_valid_mean": fusion.get("sar_gate_mean"),
        "topographic_kan_logit_mean": graph.get("topographic_kan_logit_mean"),
        "observation_amplitude_mean": graph.get("observation_amplitude_mean"),
        "latent_compatibility_mean": graph.get("latent_compatibility_mean"),
        "final_graph_gate_mean": graph.get("final_graph_gate_mean", graph.get("gate_mean")),
        "graph_gamma_mean": graph.get("graph_gamma_mean", graph.get("gamma_mean")),
        "graph_update_input_rms_ratio": graph.get("graph_update_rms_ratio"),
        "spline_base_rms_ratio": graph.get("spline_base_rms_ratio"),
        "knot_boundary_saturation_fraction": graph.get("knot_boundary_saturation_fraction"),
    }
    values = {"canonical_positive_pixel_count": float(positive.sum().item())}
    for name, value in diagnostics.items():
        if isinstance(value, torch.Tensor) and value.numel():
            reduced = value.detach().float().mean()
            if torch.isfinite(reduced):
                values[name] = float(reduced.cpu())
    return values


def resolve_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.no_amp:
        config["training"]["amp"] = False
    if args.seed is not None:
        config["seed"] = args.seed
    config["training"]["persistent_workers"] = (
        bool(config["training"]["persistent_workers"])
        and int(config["training"]["num_workers"]) > 0
    )
    return config


def make_training_context(
    config: Mapping[str, Any], loader: DataLoader, epochs: int, accumulation: int,
    max_train_batches: int | None,
) -> dict[str, Any]:
    effective_batches = min(len(loader), max_train_batches or len(loader))
    steps_per_epoch = math.ceil(effective_batches / accumulation)
    return {
        "epochs": int(epochs),
        "batch_size": int(config["training"]["batch_size"]),
        "gradient_accumulation_steps": int(accumulation),
        "sampler": type(getattr(loader, "sampler", None)).__name__,
        "batch_sampler": type(getattr(loader, "batch_sampler", None)).__name__,
        "sampling": jsonable_config(config["dataset"]["sampling"]),
        "augmentation": jsonable_config(config["dataset"]["augmentation"]),
        "seed": int(config["seed"]),
        "effective_train_batches": int(effective_batches),
        "steps_per_epoch": int(steps_per_epoch),
        "planned_total_optimizer_steps": int(epochs * steps_per_epoch),
        "warmup_steps": int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch,
        "depth_initialization_bias": config.get("model", {}).get(
            "depth_initialization_bias"
        ),
    }


def create_dataloaders(
    config: dict[str, Any]
) -> tuple[DataLoader, DataLoader, FloodDepthDataset, FloodDepthDataset]:
    validate_supervision_config(config)
    augmentation = config["dataset"]["augmentation"]
    input_spec = ModelInputSpec.from_config(config)
    transform = SynchronousAugment(
        horizontal_flip_probability=float(augmentation["horizontal_flip_probability"]),
        vertical_flip_probability=float(augmentation["vertical_flip_probability"]),
        rotate90_probability=float(augmentation["rotate90_probability"]),
        input_mode=input_spec.mode,
    )
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    train_dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "train",
        transform=transform,
        band_spec=band_spec,
        input_spec=input_spec,
        minimum_event_band_fraction=float(
            config["dataset"].get("minimum_event_band_fraction", 1.0)
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    val_dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "val",
        band_spec=band_spec,
        input_spec=input_spec,
        minimum_event_band_fraction=float(
            config["dataset"].get("minimum_event_band_fraction", 1.0)
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    sampler = make_event_balanced_sampler(
        train_dataset.event_ids, int(config["seed"]),
        float(config["dataset"]["sampling"]["event_balance_power"]),
    )
    workers = int(config["training"]["num_workers"])
    common = {
        "num_workers": workers,
        "persistent_workers": bool(config["training"]["persistent_workers"]) if workers > 0 else False,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        common["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BalancedRemainderBatchSampler(
            sampler,
            int(config["training"]["batch_size"]),
            bool(config["training"].get("drop_last", False)),
        ),
        **common,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 100_000),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def _inverse_softplus(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("train positive-depth median must be finite and positive")
    # log(expm1(x)) overflows for a sufficiently deep but still valid target.
    return float(value + math.log(-math.expm1(-value)))


def prepare_train_only_calibration(
    config: dict[str, Any], train_dataset: FloodDepthDataset,
) -> dict[str, Any]:
    """Resolve the configured depth-head bias from train-only valid targets."""

    if config["model"].get("depth_initialization_mode") != "train_positive_median":
        return {}
    scan = collect_canonical_train_depths(
        train_dataset.contract, train_dataset.band_spec,
        minimum_event_band_fraction=float(train_dataset.minimum_event_band_fraction),
    )
    summary = scan.summary()
    median = float(summary["depth_median_m"])
    return {
        "train_depth_scan": summary,
        "depth_initialization": {
            "source": "canonical_train_positive_depth_median",
            "train_positive_depth_median_m": median,
            "depth_initialization_bias": _inverse_softplus(median),
            "head_transform": "softplus(raw_depth_bias) + depth_epsilon",
            "train_depth_scan": summary,
        },
    }


def apply_train_only_calibration(
    config: dict[str, Any], payload: Mapping[str, Any],
) -> None:
    """Apply the train-only depth-head initialization to the resolved config."""

    initialization = payload.get("depth_initialization")
    if isinstance(initialization, Mapping):
        bias = float(initialization["depth_initialization_bias"])
        config["model"]["depth_initialization_bias"] = bias
        config["model"]["depth_initialization_source"] = str(initialization["source"])
        config["model"]["train_positive_depth_median_m"] = float(
            initialization["train_positive_depth_median_m"]
        )


def _calibration_artifact_paths(
    config: Mapping[str, Any], run_dir: Path, filename: str,
) -> list[Path]:
    paths = [run_dir / filename]
    configured = config.get("artifacts_root")
    if configured is not None:
        root = Path(configured)
        target = root / filename
        if target not in paths:
            paths.append(target)
    return paths


def write_train_only_calibration_artifacts(
    config: Mapping[str, Any], run_dir: Path, payload: Mapping[str, Any],
) -> None:
    initialization = payload.get("depth_initialization")
    if isinstance(initialization, Mapping):
        for path in _calibration_artifact_paths(
            config, run_dir, "train_positive_depth_initialization.json"
        ):
            atomic_write_json(path, dict(initialization))


@torch.no_grad()
def initialized_depth_distribution(
    model: torch.nn.Module,
    train_dataset: FloodDepthDataset,
    input_spec: ModelInputSpec,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    """Record an initialization-only depth distribution from one train sample."""

    original_transform = train_dataset.transform
    train_dataset.transform = None
    try:
        batch = default_collate([train_dataset[0]])
    finally:
        train_dataset.transform = original_transform
    batch = move_to_device(batch, device, non_blocking=device.type == "cuda")
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(
                prepare_model_inputs(
                    batch,
                    input_spec,
                    include_valid_depth_mask=bool(
                        getattr(model, "known_flood_range_enabled", False)
                    ),
                )
            )
    finally:
        model.train(was_training)
    values = outputs["conditional_depth"].detach().float()[
        batch["validity"]["output_valid"] > 0.5
    ]
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise RuntimeError("initial depth distribution is empty or non-finite on train data")
    quantiles = torch.quantile(values, values.new_tensor([0.05, 0.50, 0.95]))
    return {
        "initial_depth_valid_pixels": int(values.numel()),
        "initial_depth_mean_m": float(values.mean().cpu()),
        "initial_depth_p05_m": float(quantiles[0].cpu()),
        "initial_depth_p50_m": float(quantiles[1].cpu()),
        "initial_depth_p95_m": float(quantiles[2].cpu()),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    criterion: CompositeFloodDepthLoss,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    grad_clip: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None,
    run_dir: Path,
    log_every_steps: int = 10,
    csv_enabled: bool = True,
    non_blocking: bool = True,
    input_spec: ModelInputSpec | None = None,
    progress: bool = False,
    total_epochs: int | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    batches = 0
    samples = 0
    effective_batches = min(
        len(loader), max_batches if max_batches is not None else len(loader)
    )
    disable_progress = (
        not progress
        or os.environ.get("FLOOD_DEPTH_DISABLE_TQDM", "").lower() in {
            "1",
            "true",
            "yes",
        }
    )
    epoch_label = (
        f"Epoch {epoch + 1:03d}/{total_epochs:03d}"
        if total_epochs
        else f"Epoch {epoch + 1}"
    )
    iterator = tqdm(
        loader,
        total=effective_batches,
        desc=epoch_label,
        leave=False,
        disable=disable_progress,
        dynamic_ncols=True,
    )
    accumulated_samples = 0
    optimizer_steps = 0
    skipped_steps = 0
    interval_start = time.perf_counter()
    interval_samples = 0
    interval_data_time = 0.0
    interval_compute_time = 0.0
    last_batch_end = interval_start
    for batch_index, cpu_batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_received = time.perf_counter()
        interval_data_time += batch_received - last_batch_end
        compute_start = batch_received
        batch = move_to_device(cpu_batch, device, non_blocking=non_blocking)
        batch_size = int(cpu_batch["label"].shape[0])
        final_batch = batch_index + 1 >= effective_batches
        should_step = (batch_index + 1) % accumulation_steps == 0 or final_batch
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            outputs = model(
                prepare_model_inputs(
                    batch,
                    input_spec,
                    include_valid_depth_mask=bool(
                        getattr(model, "known_flood_range_enabled", False)
                    ),
                )
            )
            loss, components = criterion(outputs, batch)
        nonfinite = [name for name, value in components.items()
                     if not torch.isfinite(value.detach()).all()]
        if not torch.isfinite(loss).all() or nonfinite:
            def value_range(value: Any) -> list[float | None]:
                if not isinstance(value, torch.Tensor):
                    return [None, None]
                finite = value.detach().float()[torch.isfinite(value.detach().float())]
                if finite.numel() == 0:
                    return [None, None]
                return [float(finite.min().cpu()), float(finite.max().cpu())]
            positive_mask = canonical_positive_mask_from_batch(batch)
            unlabeled_mask = (
                (batch["validity"]["output_valid"] > 0.5) & ~positive_mask
            )
            graph_payload = {
                key: float(value.detach().float().mean().cpu())
                for key, value in outputs.get("graph_diagnostics", {}).items()
                if isinstance(value, torch.Tensor)
            }
            payload = {
                "epoch": epoch,
                "batch": batch_index,
                "sample_id": str(cpu_batch.get("metadata", {}).get("sample_id", "unknown")),
                "nonfinite_components": nonfinite,
                "loss": float(loss.detach().float().cpu()) if torch.isfinite(loss).all() else None,
                "prediction_range": value_range(outputs.get("depth")),
                "target_range": value_range(batch.get("label")),
                "positive_pixels": int(positive_mask.sum().item()),
                "unlabeled_pixels": int(unlabeled_mask.sum().item()),
                "graph_diagnostics": graph_payload,
                "amp_scale": float(scaler.get_scale()),
            }
            with (run_dir / "nonfinite_losses.jsonl").open("a", encoding="utf-8") as handle:
                import json
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}; components={nonfinite}")
        # Accumulate sums over samples, then normalize the complete (including
        # short final) window once before clipping.  This prevents a singleton
        # or max-batches remainder from receiving a full-sized update.
        scaler.scale(loss * batch_size).backward()
        accumulated_samples += batch_size
        if should_step:
            scaler.unscale_(optimizer)
            normalize_accumulated_gradients(model, accumulated_samples)
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            successful_step = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
            if successful_step:
                scheduler.step()
                optimizer_steps += 1
            else:
                skipped_steps += 1
            accumulated_samples = 0
        else:
            raw_grad_norm = loss.new_tensor(float("nan"))
            clipped_grad_norm = loss.new_tensor(float("nan"))
        batches += 1
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        diagnostic_values = _training_diagnostic_values(outputs, batch)
        for name, value in diagnostic_values.items():
            sums[name] = sums.get(name, 0.0) + value * batch_size
        samples += batch_size
        interval_samples += batch_size
        interval_compute_time += time.perf_counter() - compute_start
        if not disable_progress:
            iterator.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )
        if csv_enabled and (
            batch_index % max(1, log_every_steps) == 0 or final_batch
        ):
            now = time.perf_counter()
            elapsed = max(now - interval_start, 1e-9)
            step_row = {
                "epoch": epoch,
                "batch": batch_index,
                "loss": float(loss.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "raw_gradient_norm": float(raw_grad_norm.detach().cpu()),
                "clipped_gradient_norm": float(clipped_grad_norm.detach().cpu()),
                "amp_scale": float(scaler.get_scale()),
                "optimizer_steps": optimizer_steps,
                "skipped_steps": skipped_steps,
                "step_time_seconds": elapsed,
                "samples_per_second": interval_samples / elapsed,
                "interval_samples": interval_samples,
                "data_time_seconds": interval_data_time,
                "compute_time_seconds": interval_compute_time,
                "s1_event_support_fraction": float(
                    batch["validity"]["s1_event_support"].float().mean().detach().cpu()
                ),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                "gpu_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
            }
            step_row.update(diagnostic_values)
            append_csv(run_dir / "train_steps.csv", step_row)
            interval_start = now
            interval_samples = 0
            interval_data_time = 0.0
            interval_compute_time = 0.0
        last_batch_end = time.perf_counter()
    if batches == 0:
        raise RuntimeError("No train batches were executed")
    result = {name: value / samples for name, value in sums.items()}
    result.update({"optimizer_steps": float(optimizer_steps), "amp_skipped_steps": float(skipped_steps)})
    return result


def environment_payload(device: torch.device) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def run_training(args: argparse.Namespace) -> Path:
    config = resolve_cli(embed_source_fingerprints(load_config(args.config)), args)
    requested_device = str(config["device"])
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ) if requested_device == "auto" else torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    selected_run_dir = (
        args.resume.resolve().parent
        if args.resume is not None
        else args.output.resolve() if args.output is not None
        else Path(config["runs_root"]) / "train" / str(config["run_name"]) / timestamp
    )
    run_dir = selected_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        run_dir / "train.log",
        level=logging.INFO,
        show_python_warnings=bool(config["logging"].get("show_python_warnings", True)),
    )
    configure_training_warning_filters(
        deterministic=bool(config["deterministic"]), logger=LOGGER
    )
    run_started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    LOGGER.info("Preparing data loaders and train-only calibration …")
    train_loader, val_loader, train_dataset, _ = create_dataloaders(config)
    input_spec = train_dataset.input_spec
    normalizer = train_dataset.normalizer
    calibration_payload = prepare_train_only_calibration(config, train_dataset)
    apply_train_only_calibration(config, calibration_payload)
    write_train_only_calibration_artifacts(config, run_dir, calibration_payload)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    LOGGER.info("Train depth bins: %s", depth_bins)
    selection_depth_mode = str(
        config["training"].get("selection_depth_stratification", "objective")
    )
    if selection_depth_mode == "objective":
        selection_depth_bins = depth_bins
    elif selection_depth_mode == "primary":
        selection_depth_bins = normalizer.train_depth_bins
    else:
        raise ValueError(
            "training.selection_depth_stratification must be objective or primary"
        )
    if selection_depth_bins != depth_bins:
        LOGGER.info(
            "Checkpoint selection uses common primary depth bins: %s",
            selection_depth_bins,
        )

    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(
        device, bool(config["training"]["amp"]),
        str(config["training"].get("amp_dtype", "float16")),
    )
    model = build_model(config).to(device)
    fingerprint = dataset_fingerprint(config)
    if isinstance(calibration_payload.get("depth_initialization"), Mapping):
        initialization = dict(calibration_payload["depth_initialization"])
        initialization.update(
            initialized_depth_distribution(
                model,
                train_dataset,
                input_spec,
                device,
                amp_enabled,
                amp_dtype,
            )
        )
        calibration_payload = {**dict(calibration_payload), "depth_initialization": initialization}
        write_train_only_calibration_artifacts(config, run_dir, calibration_payload)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters >= 25_000_000:
        raise RuntimeError(f"Model exceeds the 25M parameter target: {total_parameters}")
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    effective_train_batches = min(
        len(train_loader),
        args.max_train_batches if args.max_train_batches is not None else len(train_loader),
    )
    steps_per_epoch = math.ceil(effective_train_batches / accumulation)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch
    training_context = make_training_context(
        config, train_loader, epochs, accumulation, args.max_train_batches
    )
    scheduler = build_scheduler(optimizer, config, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    criterion = CompositeFloodDepthLoss(
        config["loss"], depth_bins, normalizer.train_depth_bins,
    )
    monitor = str(config["training"]["best_metric"])
    start_epoch, best_metric, patience = 0, float("inf"), 0
    best_epoch = -1
    global_step = 0
    best_raw_metric = float("inf")
    if args.resume is not None:
        current_identity = training_identity_sha256(
            jsonable_config(config),
            fingerprint,
            training_context=training_context,
        )
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            expected_fingerprint=fingerprint,
            allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
            restore_rng=True,
            map_location=device,
            expected_training_identity_sha256=current_identity,
        )
        checkpoint_monitor = str(
            checkpoint.get("extra", {}).get("best_metric_name", "event_macro_mae")
        )
        if checkpoint_monitor != monitor:
            raise RuntimeError(
                "Resume checkpoint monitor differs from the active configuration: "
                f"{checkpoint_monitor!r} != {monitor!r}"
            )
        checkpoint_semantics = checkpoint_depth_output_semantics(checkpoint)
        configured_semantics = str(
            config["model"].get("depth_output_semantics", "conditional_positive")
        )
        if checkpoint_semantics != configured_semantics:
            raise RuntimeError(
                "Resume checkpoint depth semantics differs from the active configuration: "
                f"{checkpoint_semantics!r} != {configured_semantics!r}"
            )
        checkpoint_config = checkpoint.get("resolved_config", {})
        checkpoint_loss = (
            checkpoint_config.get("loss", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        checkpoint_depth_bins = resolve_depth_stratification_bins(
            checkpoint_loss if isinstance(checkpoint_loss, Mapping) else {}, normalizer
        )
        if checkpoint_depth_bins != depth_bins:
            raise RuntimeError(
                "Resume checkpoint depth strata differ from the active configuration: "
                f"{checkpoint_depth_bins} != {depth_bins}"
            )
        active_loss = jsonable_config(config["loss"])
        saved_loss = jsonable_config(dict(checkpoint_loss))
        if saved_loss != active_loss:
            raise RuntimeError(
                "Resume checkpoint loss configuration differs from the active "
                "configuration. Start a new run for a changed objective."
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint["best_metric"])
        saved_extra = checkpoint.get("extra", {})
        best_raw_metric = float(saved_extra.get("best_raw_metric", best_metric))
        best_epoch = int(saved_extra.get("best_epoch", -1))
        patience = int(checkpoint.get("extra", {}).get("early_stop_patience", 0))
        LOGGER.info("Resumed %s at epoch %d", args.resume, start_epoch)

    writer = create_summary_writer(
        run_dir / "tensorboard",
        enabled=bool(config["logging"].get("tensorboard", False)),
        flush_seconds=int(config["logging"].get("tensorboard_flush_seconds", 30)),
        logger=LOGGER,
    )
    log_training_header(
        LOGGER,
        model=str(config["model"]["display_name"]),
        run_dir=run_dir,
        device=str(device),
        epochs=epochs,
        batch_size=int(config["training"]["batch_size"]),
        parameters=trainable_parameters,
        tensorboard_dir=run_dir / "tensorboard" if writer is not None else None,
    )
    add_metadata(
        writer,
        {
            "model": config["model"]["display_name"],
            "run_id": run_dir.name,
            "device": device,
            "seed": config["seed"],
            "epochs": epochs,
            "batch_size": config["training"]["batch_size"],
            "best_metric": monitor,
        },
    )
    atomic_write_json(run_dir / "resolved_config.json", jsonable_config(config))
    atomic_write_json(run_dir / "environment.json", environment_payload(device))
    atomic_write_json(run_dir / "dataset_fingerprint.json", fingerprint)
    run_metadata = {
        "run_id": run_dir.name,
        "started_at": run_started_at.isoformat(),
        "stage": "train_from_scratch",
    }
    atomic_write_json(run_dir / "run_metadata.json", run_metadata)
    atomic_write_json(
        run_dir / "model_summary.json",
        {
            "name": str(config["model"]["name"]),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "depth_output_semantics": config["model"].get(
                "depth_output_semantics", "conditional_positive"
            ),
            "best_metric": monitor,
            "depth_stratification_edges_m": depth_bins,
            "primary_depth_stratification_edges_m": normalizer.train_depth_bins,
            "selection_depth_stratification_edges_m": selection_depth_bins,
            "resolved_model_bands": config["dataset"].get("resolved_model_bands"),
            "amp_dtype": str(amp_dtype),
            "graph_identity": resolved_graph_identity(config),
            "depth_initialization": calibration_payload.get("depth_initialization"),
        },
    )
    epoch_durations: list[float] = []
    completed_epochs = start_epoch
    early_stopped = False
    try:
        for epoch in range(start_epoch, epochs):
            if (
                epoch >= int(config["training"].get("minimum_epochs", 0))
                and patience >= int(config["training"]["early_stop_patience"])
            ):
                early_stopped = True
                LOGGER.info(
                    "Resume checkpoint already satisfies early stopping at epoch %d",
                    epoch - 1,
                )
                break
            epoch_started = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                criterion,
                device,
                epoch,
                accumulation,
                float(config["training"]["grad_clip_norm"]),
                amp_enabled,
                amp_dtype,
                args.max_train_batches,
                run_dir,
                int(config["logging"]["log_every_steps"]),
                bool(config["logging"]["csv"]),
                bool(config["training"].get("non_blocking", device.type == "cuda")),
                input_spec,
                bool(config["logging"].get("progress_bar", False)),
                epochs,
            )
            global_step += int(train_metrics.get("optimizer_steps", 0.0))
            validation_interval = max(
                1, int(config["training"].get("validation_interval", 1))
            )
            should_validate = (
                (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
            )
            if not should_validate:
                if bool(config["checkpoint"]["save_last"]):
                    save_checkpoint(
                        run_dir / "last.pth",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_metric,
                        jsonable_config(config),
                        fingerprint,
                        extra={
                            "total_parameters": total_parameters,
                            "best_metric_name": monitor,
                            "depth_stratification_edges_m": depth_bins,
                            "primary_depth_stratification_edges_m": normalizer.train_depth_bins,
                            "early_stop_patience": patience,
                            "best_epoch": best_epoch,
                            "best_raw_metric": best_raw_metric,
                            "graph_identity": runtime_graph_identity(model),
                        },
                        training_context=training_context,
                        global_step=global_step,
                    )
                epoch_seconds = time.perf_counter() - epoch_started
                epoch_durations.append(epoch_seconds)
                elapsed_seconds = time.perf_counter() - start_time
                eta_seconds = (
                    sum(epoch_durations) / len(epoch_durations) * (epochs - epoch - 1)
                )
                add_scalars(writer, train_metrics, step=epoch + 1, prefix="train")
                add_scalars(
                    writer,
                    {
                        "epoch_seconds": epoch_seconds,
                        "elapsed_seconds": elapsed_seconds,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "early_stop_patience": patience,
                        "early_stop_remaining": max(
                            0,
                            int(config["training"]["early_stop_patience"])
                            - patience,
                        ),
                    },
                    step=epoch + 1,
                    prefix="system",
                )
                flush(writer)
                log_epoch_summary(
                    LOGGER,
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    epoch_seconds=epoch_seconds,
                    elapsed_seconds=elapsed_seconds,
                    eta_seconds=eta_seconds,
                    train_loss=float(train_metrics["total"]),
                    metric_name=None,
                    metric_value=None,
                    best_metric=best_metric,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    patience=patience,
                    patience_limit=int(config["training"]["early_stop_patience"]),
                    minimum_epochs=int(config["training"].get("minimum_epochs", 0)),
                    improved=None,
                )
                completed_epochs = epoch + 1
                continue
            val_summary, _, _, _ = evaluate_loader(
                model,
                val_loader,
                device,
                selection_depth_bins,
                primary_depth_bins=normalizer.train_depth_bins,
                criterion=criterion,
                max_batches=args.max_val_batches,
                progress=bool(config["logging"].get("progress_bar", False)),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                input_spec=input_spec,
            )
            selected_summary = val_summary
            if monitor not in selected_summary:
                raise KeyError(
                    f"Configured best metric {monitor!r} is absent from validation summary"
                )
            raw_metric = float(val_summary[monitor])
            raw_improved = raw_metric < best_raw_metric - float(config["training"].get("min_delta", 0.0))
            if raw_improved:
                best_raw_metric = raw_metric
            metric = float(selected_summary[monitor])
            improved = metric < best_metric - float(config["training"].get("min_delta", 0.0))
            if improved:
                best_metric, best_epoch, patience = metric, epoch, 0
            else:
                patience += 1
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_summary.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                f"best_{monitor}": best_metric,
            }
            common = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                resolved_config=jsonable_config(config),
                dataset_fingerprint=fingerprint,
                extra={
                    "total_parameters": total_parameters,
                    "best_metric_name": monitor,
                    "depth_stratification_edges_m": depth_bins,
                    "primary_depth_stratification_edges_m": normalizer.train_depth_bins,
                    "early_stop_patience": patience,
                    "best_epoch": best_epoch,
                    "best_raw_metric": best_raw_metric,
                    "graph_identity": runtime_graph_identity(model),
                },
            )
            if bool(config["checkpoint"]["save_last"]):
                save_checkpoint(
                    run_dir / "last.pth", training_context=training_context,
                    global_step=global_step, **common
                )
            if improved:
                if bool(config["checkpoint"]["save_best"]):
                    save_checkpoint(
                        run_dir / "best.pth", training_context=training_context,
                        global_step=global_step, **common
                    )
            if raw_improved and bool(config["checkpoint"]["save_best"]):
                raw_common = dict(common)
                raw_common["best_metric"] = best_raw_metric
                save_checkpoint(
                    run_dir / "best_raw.pth",
                    training_context=training_context,
                    global_step=global_step, **raw_common
                )
            epoch_seconds = time.perf_counter() - epoch_started
            epoch_durations.append(epoch_seconds)
            elapsed_seconds = time.perf_counter() - start_time
            eta_seconds = (
                sum(epoch_durations) / len(epoch_durations) * (epochs - epoch - 1)
            )
            row.update(
                {
                    "epoch_seconds": epoch_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "early_stop_patience": patience,
                    "early_stop_remaining": max(
                        0, int(config["training"]["early_stop_patience"]) - patience
                    ),
                    "improved": improved,
                }
            )
            if bool(config["logging"].get("csv", True)):
                append_csv(run_dir / "metrics_by_epoch.csv", row)
            add_scalars(writer, train_metrics, step=epoch + 1, prefix="train")
            add_scalars(writer, val_summary, step=epoch + 1, prefix="validation/raw")
            add_scalars(
                writer,
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "best_metric": best_metric,
                    "epoch_seconds": epoch_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "early_stop_patience": patience,
                    "early_stop_remaining": max(
                        0, int(config["training"]["early_stop_patience"]) - patience
                    ),
                },
                step=epoch + 1,
                prefix="system",
            )
            flush(writer)
            log_epoch_summary(
                LOGGER,
                epoch=epoch + 1,
                total_epochs=epochs,
                epoch_seconds=epoch_seconds,
                elapsed_seconds=elapsed_seconds,
                eta_seconds=eta_seconds,
                train_loss=float(train_metrics["total"]),
                metric_name=monitor,
                metric_value=metric,
                best_metric=best_metric,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                patience=patience,
                patience_limit=int(config["training"]["early_stop_patience"]),
                minimum_epochs=int(config["training"].get("minimum_epochs", 0)),
                improved=improved,
            )
            completed_epochs = epoch + 1
            if (
                epoch + 1 >= int(config["training"].get("minimum_epochs", 0))
                and patience >= int(config["training"]["early_stop_patience"])
            ):
                early_stopped = True
                LOGGER.info(
                    "Early stopping triggered after epoch %d/%d (patience %d/%d).",
                    epoch + 1,
                    epochs,
                    patience,
                    int(config["training"]["early_stop_patience"]),
                )
                break
    finally:
        if writer is not None:
            writer.close()
        elapsed = time.perf_counter() - start_time
        atomic_write_json(
            run_dir / "training_runtime.json",
            {
                "elapsed_seconds": elapsed,
                "started_at": run_started_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": completed_epochs,
                "early_stopped": early_stopped,
                "early_stop_patience": patience,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
            },
        )
    atomic_write_json(
        run_dir / "training_summary.json",
        {
            "model": str(config["model"]["name"]),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "best_metric_name": monitor,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return run_dir


if __name__ == "__main__":
    raise SystemExit("Use `python train.py <model-config.xml>` from the project root.")
