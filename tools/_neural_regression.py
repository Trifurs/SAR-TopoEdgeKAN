"""Shared training and evaluation mechanics for named learned comparators.

The root-level XML dispatchers are the only public operation entry points. This
private module keeps comparator data loading, objectives, checkpoints, and
metric schema consistent while each architecture remains in ``compare/``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.preprocessing import resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from datasets.supervision_masks import (
    CANONICAL_POSITIVE_MASK,
    canonical_positive_mask_from_batch,
    supervision_mask_counts,
)
from metrics.aggregator import EvaluationAggregator
from compare.common._depth_regression import prepare_comparison_tensor
from compare.common.comparison_factory import build_comparison_model
from tools.evaluate_sar_topo_edge_kan import dataset_fingerprint, embed_source_fingerprints, metadata_item
from tools.train_sar_topo_edge_kan import create_dataloaders
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint, save_checkpoint, training_identity_sha256
from utils.config import jsonable_config, load_config
from utils.efficiency import (
    InferenceEfficiency,
    checkpoint_size_metrics,
    model_size_metrics,
)
from utils.spatial_tta import spatial_tta_forward
from utils.logging import (
    append_csv,
    configure_training_warning_filters,
    log_epoch_summary,
    log_training_header,
    setup_logging,
    write_rows,
)
from utils.misc import atomic_write_json, move_to_device
from utils.optim import build_optimizer, build_scheduler
from utils.seed import seed_everything
from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


LOGGER = logging.getLogger("comparison.train")


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _validated_config_value(
    config_value: Mapping[str, Any], expected_model: str
) -> dict[str, Any]:
    config = embed_source_fingerprints(deepcopy(dict(config_value)))
    model = config.get("model", {})
    if str(model.get("name")) != expected_model:
        raise ValueError(
            f"This entry point is reserved for {expected_model!r}; config requests {model.get('name')!r}"
        )
    if str(model.get("flood_support")) != "valid_depth_mask":
        raise ValueError("Learned comparators must use direct valid_depth_mask flood support")
    if str(model.get("depth_output_semantics", "conditional_positive")) != "conditional_positive":
        raise ValueError("Learned comparators require conditional_positive depth semantics")
    return config


def _validated_config(config_path: Path, expected_model: str) -> dict[str, Any]:
    return _validated_config_value(load_config(config_path), expected_model)


def _apply_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device is not None:
        config["device"] = args.device
    if getattr(args, "epochs", None) is not None:
        config["training"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if getattr(args, "num_workers", None) is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        config["training"]["num_workers"] = int(args.num_workers)
    if getattr(args, "seed", None) is not None:
        config["seed"] = int(args.seed)
    if getattr(args, "no_amp", False):
        config["training"]["amp"] = False
    config["training"]["persistent_workers"] = bool(
        config["training"].get("persistent_workers", False)
    ) and int(config["training"]["num_workers"]) > 0
    return config


def _masked_objective(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    *,
    criterion: CompositeFloodDepthLoss | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if str(loss_config["objective_mode"]) != "task_adaptive":
        raise ValueError("Learned comparisons require the shared task-adaptive objective")
    if criterion is None:
        raise ValueError("The shared objective requires its train-bin criterion")
    return criterion(outputs, batch, epoch)


def _comparison_criterion(
    config: Mapping[str, Any], normalizer: Any, device: torch.device
) -> CompositeFloodDepthLoss:
    if str(config["loss"]["objective_mode"]) != "task_adaptive":
        raise ValueError("Learned comparisons require the shared task-adaptive objective")
    return CompositeFloodDepthLoss(
        config["loss"],
        resolve_depth_stratification_bins(config["loss"], normalizer),
        normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
    ).to(device)


def _batch_metadata(batch: Mapping[str, Any], index: int) -> tuple[str, str]:
    metadata = batch.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return str(index), "unknown"
    return (
        str(metadata_item(metadata.get("sample_id", (str(index),)), index)),
        str(metadata_item(metadata.get("source_event_id", ("unknown",)), index)),
    )


@torch.inference_mode()
def evaluate_comparison_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    depth_bins: list[float],
    primary_bins: list[float],
    *,
    max_batches: int | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    progress: bool = False,
    measure_efficiency: bool = False,
    progress_label: str = "Validation",
    efficiency_warmup_batches: int | None = None,
    tta_transforms: list[str] | tuple[str, ...] | None = None,
    criterion: CompositeFloodDepthLoss | None = None,
    epoch: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    if criterion is None:
        criterion = _comparison_criterion(config, loader.dataset.normalizer, device)
    aggregator = EvaluationAggregator(depth_bins, primary_bins)
    objectives: list[float] = []
    mask_totals = {
        "valid_depth_mask_pixels": 0,
        "output_valid_pixels": 0,
        "positive_supervision_pixels": 0,
        "positive_excluded_by_output_valid_pixels": 0,
    }
    schema = str(config["model"]["input_schema"])
    efficiency = InferenceEfficiency(
        device,
        enabled=measure_efficiency,
        warmup_batches=efficiency_warmup_batches,
    )
    disable_progress = not progress or os.environ.get("FLOOD_DEPTH_DISABLE_TQDM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    selected_batches = loader if max_batches is None else islice(loader, max_batches)
    progress_total = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, cpu_batch in enumerate(
        tqdm(
            selected_batches,
            total=progress_total,
            desc=progress_label,
            leave=False,
            disable=disable_progress,
        )
    ):
        batch = move_to_device(
            cpu_batch,
            device,
            non_blocking=bool(config["training"].get("non_blocking", device.type == "cuda")),
        )
        for name, value in supervision_mask_counts(batch).items():
            if name in mask_totals:
                mask_totals[name] += int(value.detach().cpu())
        inputs, flood_range = prepare_comparison_tensor(batch, schema)
        inference_started = efficiency.start()
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = spatial_tta_forward(
                lambda values: model(values[0], values[1]),
                (inputs, flood_range),
                tta_transforms,
            )
        efficiency.stop(
            inference_started,
            samples=int(outputs["depth"].shape[0]),
            output_pixels=int(outputs["depth"].numel()),
        )
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            objective, _ = _masked_objective(
                outputs, batch, config["loss"], criterion=criterion, epoch=epoch
            )
        objectives.append(float(objective.detach().cpu()))
        valid = canonical_positive_mask_from_batch(batch).detach().cpu().numpy()
        for sample_index in range(outputs["depth"].shape[0]):
            sample_id, event_id = _batch_metadata(cpu_batch, sample_index)
            aggregator.add(
                sample_id,
                event_id,
                outputs["depth"][sample_index].detach().float().cpu().numpy(),
                batch["label"][sample_index].detach().float().cpu().numpy(),
                outputs["uncertainty_scale"][sample_index].detach().float().cpu().numpy(),
                valid[sample_index],
            )
    summary, samples, events, bins = aggregator.summarize()
    summary["spatial_tta_transforms"] = list(
        outputs.get("spatial_tta_transforms", ("identity",))
    )
    summary.update(mask_totals)
    summary.update({
        "method": str(config["model"]["name"]),
        "flood_support": "valid_depth_mask",
        "input_schema": schema,
        "evaluation_mask": CANONICAL_POSITIVE_MASK,
        "positive_excluded_by_output_valid_fraction": float(
            mask_totals["positive_excluded_by_output_valid_pixels"]
        ) / max(float(mask_totals["valid_depth_mask_pixels"]), 1.0),
        "objective_mean": float(np.mean(objectives)) if objectives else float("nan"),
    })
    summary.update(efficiency.summary())
    return summary, samples, events, bins


def _write_evaluation(
    output: Path,
    summary: Mapping[str, Any],
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", dict(summary))
    atomic_write_json(output / "resolved_config.json", jsonable_config(config))
    write_rows(output / "metrics_by_sample.csv", samples)
    write_rows(output / "metrics_by_event.csv", events)
    write_rows(output / "metrics_by_train_depth_bin.csv", bins)


def _parameter_payload(model: torch.nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": str(config["model"]["name"]),
        **model_size_metrics(model),
        "input_schema": str(config["model"]["input_schema"]),
        "flood_support": "valid_depth_mask",
        "architecture": dict(config["model"]),
    }


def run_training(args: argparse.Namespace, expected_model: str) -> Path:
    config = _apply_cli(_validated_config(args.config, expected_model), args)
    resume = getattr(args, "resume", None)
    device = _device(str(config["device"]))
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = (
        args.output.resolve()
        if args.output is not None
        else Path(config["runs_root"]) / "train" / str(config["run_name"]) / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        run_dir / "train.log",
        show_python_warnings=bool(config["logging"].get("show_python_warnings", True)),
    )
    configure_training_warning_filters(
        deterministic=bool(config["deterministic"]), logger=LOGGER
    )
    run_started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    LOGGER.info("Preparing learned comparison model and data loaders …")
    train_loader, val_loader, train_dataset, _ = create_dataloaders(config)
    model = build_comparison_model(config).to(device)
    parameters = _parameter_payload(model, config)
    atomic_write_json(run_dir / "parameters.json", parameters)
    atomic_write_json(run_dir / "resolved_config.json", jsonable_config(config))
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_dir.name,
            "started_at": run_started_at.isoformat(),
            "stage": "resume" if resume is not None else "train_from_scratch",
            "resume_checkpoint": str(resume) if resume is not None else None,
        },
    )
    optimizer_name = str(config["optimizer"]["name"]).lower()
    if optimizer_name != "adamw":
        raise ValueError("Learned comparison models currently support optimizer.name=adamw")
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    max_train = args.max_train_batches
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(
        device, bool(config["training"].get("amp", False)), str(config["training"].get("amp_dtype", "auto"))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    fingerprint = dataset_fingerprint(config)
    depth_bins = resolve_depth_stratification_bins(config["loss"], train_dataset.normalizer)
    primary_bins = train_dataset.normalizer.train_depth_bins
    criterion = _comparison_criterion(config, train_dataset.normalizer, device)
    monitor = str(config["training"].get("best_metric", "pixel_micro_mae"))
    if monitor not in {
        "pixel_micro_mae",
        "event_macro_mae",
        "event_hierarchical_composite_mae",
        "balanced_composite_error_m",
    }:
        raise ValueError(f"Unsupported learned-comparator best metric {monitor!r}")
    best = float("inf")
    patience = 0
    best_epoch = -1
    max_val = args.max_val_batches
    writer = create_summary_writer(
        run_dir / "tensorboard",
        enabled=bool(config["logging"].get("tensorboard", False)),
        flush_seconds=int(config["logging"].get("tensorboard_flush_seconds", 30)),
        logger=LOGGER,
    )
    log_training_header(
        LOGGER,
        model=str(config["model"].get("display_name", expected_model)),
        run_dir=run_dir,
        device=str(device),
        epochs=epochs,
        batch_size=int(config["training"]["batch_size"]),
        parameters=int(parameters["trainable_parameters"]),
        tensorboard_dir=run_dir / "tensorboard" if writer is not None else None,
    )
    add_metadata(
        writer,
        {
            "model": config["model"].get("display_name", expected_model),
            "run_id": run_dir.name,
            "device": device,
            "seed": config["seed"],
            "epochs": epochs,
            "batch_size": config["training"]["batch_size"],
            "best_metric": monitor,
        },
    )
    epoch_durations: list[float] = []
    completed_epochs = 0
    early_stopped = False
    training_context = {
        "epochs": epochs,
        "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
        "sampler": type(train_loader.sampler).__name__,
    }
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if accumulation <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    effective_train_batches = min(
        len(train_loader), max_train if max_train is not None else len(train_loader)
    )
    if effective_train_batches <= 0:
        raise RuntimeError("No comparison training batches were configured")
    steps_per_epoch = math.ceil(effective_train_batches / accumulation)
    scheduler = build_scheduler(
        optimizer,
        config,
        total_steps=max(1, steps_per_epoch * epochs),
        warmup_steps=int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch,
    )
    global_step = 0
    start_epoch = 0
    try:
        if resume is not None:
            checkpoint = load_checkpoint(
                resume,
                model,
                optimizer,
                scheduler,
                scaler,
                expected_fingerprint=fingerprint,
                allow_fingerprint_mismatch=bool(
                    getattr(args, "allow_fingerprint_mismatch", False)
                ),
                restore_rng=True,
                map_location=device,
                expected_training_identity_sha256=training_identity_sha256(
                    jsonable_config(config),
                    fingerprint,
                    training_context=training_context,
                ),
            )
            checkpoint_monitor = str(
                checkpoint.get("extra", {}).get("best_metric_name", monitor)
            )
            if checkpoint_monitor != monitor:
                raise RuntimeError(
                    "Resume checkpoint monitor differs from the active configuration: "
                    f"{checkpoint_monitor!r} != {monitor!r}"
                )
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint.get("global_step", 0))
            best = float(checkpoint["best_metric"])
            checkpoint_extra = checkpoint.get("extra", {})
            patience = int(checkpoint_extra.get("early_stop_patience", 0))
            best_epoch = int(checkpoint_extra.get("best_epoch", -1))
            if best_epoch < 0:
                validation_summary = checkpoint_extra.get("validation_summary", {})
                if isinstance(validation_summary, Mapping):
                    checkpoint_metric = float(
                        validation_summary.get(monitor, float("nan"))
                    )
                    if math.isclose(
                        checkpoint_metric,
                        best,
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-12,
                    ):
                        best_epoch = int(checkpoint["epoch"])
            completed_epochs = start_epoch
            LOGGER.info(
                "Resumed %s | completed epochs=%d | next epoch=%d/%d",
                resume,
                start_epoch,
                min(start_epoch + 1, epochs),
                epochs,
            )

        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            model.train()
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            total_loss, batches = 0.0, 0
            progress = bool(config["logging"].get("progress_bar", False))
            disable_progress = (
                not progress
                or os.environ.get("FLOOD_DEPTH_DISABLE_TQDM", "").lower()
                in {"1", "true", "yes"}
            )
            optimizer.zero_grad(set_to_none=True)
            iterator = tqdm(
                train_loader,
                total=effective_train_batches,
                desc=f"Epoch {epoch + 1:03d}/{epochs:03d}",
                leave=False,
                disable=disable_progress,
                dynamic_ncols=True,
            )
            for batch_index, cpu_batch in enumerate(iterator):
                if batch_index >= effective_train_batches:
                    break
                batch = move_to_device(
                    cpu_batch,
                    device,
                    non_blocking=bool(
                        config["training"].get("non_blocking", device.type == "cuda")
                    ),
                )
                inputs, flood_range = prepare_comparison_tensor(
                    batch, str(config["model"]["input_schema"])
                )
                window_start = (batch_index // accumulation) * accumulation
                window_size = min(accumulation, effective_train_batches - window_start)
                should_step = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == effective_train_batches
                )
                with torch.autocast(
                    device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
                ):
                    outputs = model(inputs, flood_range)
                    loss, _ = _masked_objective(
                        outputs, batch, config["loss"], criterion=criterion, epoch=epoch
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                scaler.scale(loss / window_size).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["training"]["grad_clip_norm"])
                    )
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if not scaler.is_enabled() or scaler.get_scale() >= scale_before:
                        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step()
                        global_step += 1
                total_loss += float(loss.detach().cpu())
                batches += 1
                if not disable_progress:
                    iterator.set_postfix(
                        loss=f"{float(loss.detach()):.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        refresh=False,
                    )
            if batches == 0:
                raise RuntimeError("No comparison training batches were executed")

            should_validate = (
                (epoch + 1) % max(1, int(config["training"].get("validation_interval", 1)))
                == 0
                or epoch + 1 == epochs
            )
            summary: dict[str, Any] | None = None
            improved: bool | None = None
            metric_value: float | None = None
            if should_validate:
                summary, samples, events, bins = evaluate_comparison_loader(
                    model,
                    val_loader,
                    device,
                    config,
                    depth_bins,
                    primary_bins,
                    max_batches=max_val,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    progress=progress,
                    criterion=criterion,
                    epoch=epoch,
                )
                summary["epoch"] = epoch
                summary["total_parameters"] = parameters["total_parameters"]
                summary["trainable_parameters"] = parameters["trainable_parameters"]
                _write_evaluation(run_dir / "validation", summary, samples, events, bins, config)
                metric_value = float(summary[monitor])
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metric_value)
                improved = metric_value < best - float(config["training"].get("min_delta", 0.0))
                if improved:
                    best, best_epoch, patience = metric_value, epoch, 0
                else:
                    patience += 1

            checkpoint_extra = {
                "best_metric_name": monitor,
                "validation_summary": dict(summary or {}),
                "early_stop_patience": patience,
                "best_epoch": best_epoch,
            }
            if bool(config["checkpoint"].get("save_last", True)):
                save_checkpoint(
                    run_dir / "last_raw.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best,
                    jsonable_config(config),
                    fingerprint,
                    extra=checkpoint_extra,
                    training_context=training_context,
                    global_step=global_step,
                )
            if improved and bool(config["checkpoint"].get("save_best", True)):
                save_checkpoint(
                    run_dir / "best_raw.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best,
                    jsonable_config(config),
                    fingerprint,
                    extra=checkpoint_extra,
                    training_context=training_context,
                    global_step=global_step,
                )

            epoch_seconds = time.perf_counter() - epoch_started
            epoch_durations.append(epoch_seconds)
            elapsed_seconds = time.perf_counter() - start_time
            eta_seconds = (
                sum(epoch_durations) / len(epoch_durations) * (epochs - epoch - 1)
            )
            row = {
                "epoch": epoch,
                "train_total_loss": total_loss / batches,
                "validation_objective": float(summary["objective_mean"])
                if summary is not None
                else float("nan"),
                "pixel_micro_mae": float(summary["pixel_micro_mae"])
                if summary is not None
                else float("nan"),
                "event_macro_mae": float(summary["event_macro_mae"])
                if summary is not None
                else float("nan"),
                "event_hierarchical_composite_mae": float(
                    summary["event_hierarchical_composite_mae"]
                )
                if summary is not None
                else float("nan"),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "early_stop_patience": patience,
                "early_stop_remaining": max(
                    0, int(config["training"]["early_stop_patience"]) - patience
                ),
                "improved": improved,
            }
            if bool(config["logging"].get("csv", True)):
                append_csv(run_dir / "metrics.csv", row)
            add_scalars(
                writer,
                {"total_loss": total_loss / batches},
                step=epoch + 1,
                prefix="train",
            )
            if summary is not None:
                add_scalars(writer, summary, step=epoch + 1, prefix="validation/raw")
            add_scalars(
                writer,
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "best_metric": best,
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
                train_loss=total_loss / batches,
                metric_name=monitor if summary is not None else None,
                metric_value=metric_value,
                best_metric=best,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                patience=patience,
                patience_limit=int(config["training"]["early_stop_patience"]),
                minimum_epochs=int(config["training"].get("minimum_epochs", 1)),
                improved=improved,
            )
            completed_epochs = epoch + 1
            if (
                should_validate
                and epoch + 1 >= int(config["training"].get("minimum_epochs", 1))
                and patience >= int(config["training"].get("early_stop_patience", epochs))
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
        atomic_write_json(
            run_dir / "training_runtime.json",
            {
                "elapsed_seconds": time.perf_counter() - start_time,
                "started_at": run_started_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": completed_epochs,
                "early_stopped": early_stopped,
                "early_stop_patience": patience,
                "best_metric": best,
                "resumed_from": str(resume) if resume is not None else None,
            },
        )
    atomic_write_json(
        run_dir / "training_summary.json",
        {
            **parameters,
            "best_metric_name": monitor,
            "best_metric": best,
            "best_epoch": best_epoch,
            "resumed_from": str(resume) if resume is not None else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return run_dir


def run_evaluation_from_config(
    config_value: Mapping[str, Any],
    checkpoint_path: Path,
    split: str,
    device_name: str,
    output: Path,
    expected_model: str,
    *,
    max_batches: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Evaluate a learned comparator from an immutable resolved run config."""

    evaluation_started = time.perf_counter()
    config = _validated_config_value(config_value, expected_model)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        config["training"]["num_workers"] = int(num_workers)
        config["training"]["persistent_workers"] = False
    device = _device(device_name)
    from datasets.band_selection import resolve_band_spec
    from datasets.contract import DatasetContract
    from datasets.flooddepth_dataset import FloodDepthDataset
    from datasets.model_input_spec import ModelInputSpec

    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        split,
        band_spec=resolve_band_spec(config, contract),
        input_spec=ModelInputSpec.from_config(config),
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
    model = build_comparison_model(config).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        # Evaluation needs only model weights/metadata. Keeping optimizer and
        # scheduler state on CPU prevents checkpoint payloads from inflating the
        # reported inference-memory peak.
        map_location="cpu",
    )
    depth_bins = resolve_depth_stratification_bins(config["loss"], dataset.normalizer)
    amp_enabled, amp_dtype, _ = resolve_amp(
        device, bool(config["training"].get("amp", False)), str(config["training"].get("amp_dtype", "auto"))
    )
    summary, samples, events, bins = evaluate_comparison_loader(
        model, loader, device, config, depth_bins, dataset.normalizer.train_depth_bins,
        max_batches=max_batches, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        progress=bool(config["logging"].get("progress_bar", False)),
        measure_efficiency=True,
        progress_label=split.capitalize(),
        efficiency_warmup_batches=int(
            config.get("runtime", {}).get("test", {}).get(
                "cuda_warmup_batches", 1
            )
        )
        if device.type == "cuda"
        else 0,
        tta_transforms=config.get("inference", {}).get("tta_transforms"),
        criterion=_comparison_criterion(config, dataset.normalizer, device),
        epoch=int(checkpoint.get("epoch", 0)),
    )
    summary["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    summary["model_family"] = "deep_learning"
    summary["efficiency_precision"] = (
        str(amp_dtype).replace("torch.", "") if amp_enabled else "float32"
    )
    summary.update(_parameter_payload(model, config))
    summary.update(checkpoint_size_metrics(checkpoint_path))
    summary["efficiency_end_to_end_seconds"] = float(
        time.perf_counter() - evaluation_started
    )
    output = output.expanduser().resolve()
    _write_evaluation(output, summary, samples, events, bins, config)
    return summary


def run_evaluation(args: argparse.Namespace, expected_model: str) -> dict[str, Any]:
    config = load_config(args.config)
    output = (
        args.output.resolve()
        if args.output
        else Path(config["runs_root"])
        / "evaluate"
        / expected_model
        / args.split
        / args.checkpoint.stem
    )
    return run_evaluation_from_config(
        config,
        args.checkpoint,
        args.split,
        args.device,
        output,
        expected_model,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    raise SystemExit("Use `python train.py <model-config.xml>` from the project root.")
