"""Atomic, fingerprint-strict checkpoints with optional RNG restoration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


class CheckpointError(RuntimeError):
    """Raised for incompatible or incomplete checkpoint state."""


def initialize_from_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    weights: str = "raw",
    transfer: str = "strict",
    expected_fingerprint: Mapping[str, str] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize a new training stage from raw or EMA checkpoint weights.

    ``strict`` preserves the ordinary exact architecture contract.
    ``compatible`` is deliberately narrow: identical tensors are transferred,
    while the SAR-TopoEdgeKAN terrain stem may append zero-initialized input
    channels.  Newly introduced residual adapters retain their identity/zero
    initialization.  This makes architecture screening reproducible without
    weakening the strict semantics of resume or evaluation.
    """

    if weights not in {"raw", "ema"}:
        raise CheckpointError("initial checkpoint weights must be raw or ema")
    if transfer not in {"strict", "compatible"}:
        raise CheckpointError("initial checkpoint transfer must be strict or compatible")
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if expected_fingerprint is not None and dict(
        checkpoint.get("dataset_fingerprint", {})
    ) != dict(expected_fingerprint):
        raise CheckpointError(
            "Dataset contract, manifest, or normalization fingerprint differs from checkpoint"
        )
    source = checkpoint.get("ema_model") if weights == "ema" else checkpoint.get("model")
    if source is None:
        if weights == "ema":
            source = checkpoint.get("model")
            weights = "raw"
        if source is None:
            raise CheckpointError("initial checkpoint contains no model state")
    if not isinstance(source, Mapping):
        raise CheckpointError("initial checkpoint model state is not a mapping")

    unwrapped = model.module if hasattr(model, "module") else model
    if transfer == "strict":
        unwrapped.load_state_dict(source, strict=True)
        report = {
            "transfer": transfer,
            "weights": weights,
            "exact_tensor_count": len(source),
            "expanded_tensor_count": 0,
            "new_tensor_count": 0,
            "unused_source_tensor_count": 0,
            "loaded_parameter_fraction": 1.0,
        }
    else:
        target = unwrapped.state_dict()
        exact: list[str] = []
        expanded: list[str] = []
        loaded_numel = 0
        for name, source_value in source.items():
            target_value = target.get(name)
            if target_value is None:
                continue
            if tuple(source_value.shape) == tuple(target_value.shape):
                target[name] = source_value.to(
                    device=target_value.device, dtype=target_value.dtype
                )
                exact.append(name)
                loaded_numel += target_value.numel()
                continue
            if (
                name == "terrain.stem.0.0.weight"
                and source_value.ndim == target_value.ndim == 4
                and source_value.shape[0] == target_value.shape[0]
                and source_value.shape[2:] == target_value.shape[2:]
                and source_value.shape[1] < target_value.shape[1]
            ):
                initialized = torch.zeros_like(target_value)
                initialized[:, : source_value.shape[1]] = source_value.to(
                    device=target_value.device, dtype=target_value.dtype
                )
                target[name] = initialized
                expanded.append(name)
                loaded_numel += source_value.numel()
        loaded_names = set(exact).union(expanded)
        # A same-named tensor with a different shape was not transferred and
        # must be treated as newly initialized.  This distinction is critical
        # when ``trainable_scope=new_adapters`` is used for compact architecture
        # transfer; otherwise those tensors would be silently frozen at random
        # initialization.
        new_names = sorted(set(target).difference(loaded_names))
        unused_names = sorted(set(source).difference(loaded_names))
        unwrapped.load_state_dict(target, strict=True)
        total_numel = sum(value.numel() for value in target.values())
        loaded_fraction = loaded_numel / max(total_numel, 1)
        if loaded_fraction < 0.90:
            raise CheckpointError(
                "Compatible initialization transferred too little model state: "
                f"{loaded_fraction:.3%} < 90.000%"
            )
        report = {
            "transfer": transfer,
            "weights": weights,
            "exact_tensor_count": len(exact),
            "expanded_tensor_count": len(expanded),
            "expanded_tensors": expanded,
            "new_tensor_count": len(new_names),
            "new_tensors": new_names,
            "unused_source_tensor_count": len(unused_names),
            "unused_source_tensors": unused_names,
            "loaded_parameter_fraction": loaded_fraction,
        }
    heads = getattr(unwrapped, "heads", None)
    setter = getattr(heads, "set_depth_output_semantics", None)
    if callable(setter):
        setter(checkpoint_depth_output_semantics(checkpoint))
    return checkpoint, report


def training_identity_sha256(
    resolved_config: Mapping[str, Any],
    dataset_fingerprint: Mapping[str, str],
    *,
    training_context: Mapping[str, Any] | None = None,
) -> str:
    """Hash the training semantics that must remain fixed for a resume."""

    identity_fields = {
        key: resolved_config.get(key)
        for key in ("model", "loss", "optimizer", "scheduler")
    }
    dataset_config = resolved_config.get("dataset", {})
    if isinstance(dataset_config, Mapping):
        identity_fields["model_bands"] = dataset_config.get("resolved_model_bands")
        identity_fields["reliability_schema"] = dataset_config.get(
            "resolved_reliability_schema"
        )
    identity_fields["dataset_fingerprint"] = dict(dataset_fingerprint)
    identity_fields["training_context"] = dict(training_context or {})
    return hashlib.sha256(
        json.dumps(identity_fields, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def checkpoint_depth_output_semantics(checkpoint: Mapping[str, Any]) -> str:
    """Read the sole supported depth-output semantic contract."""

    explicit = checkpoint.get("output_semantics")
    if explicit is not None:
        return str(explicit)
    resolved = checkpoint.get("resolved_config", {})
    if isinstance(resolved, Mapping):
        model_config = resolved.get("model", {})
        if isinstance(model_config, Mapping):
            return str(model_config.get("depth_output_semantics", "conditional_positive"))
    return "conditional_positive"


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu = state["torch_cpu"]
    if torch.is_tensor(torch_cpu):
        torch_cpu = torch_cpu.detach().to(device="cpu", dtype=torch.uint8)
    else:
        torch_cpu = torch.as_tensor(torch_cpu, dtype=torch.uint8, device="cpu")
    torch.set_rng_state(torch_cpu)
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = []
        for cuda_state in state["torch_cuda"]:
            if torch.is_tensor(cuda_state):
                cuda_state = cuda_state.detach().to(device="cpu", dtype=torch.uint8)
            else:
                cuda_state = torch.as_tensor(
                    cuda_state, dtype=torch.uint8, device="cpu"
                )
            cuda_states.append(cuda_state)
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    resolved_config: Mapping[str, Any],
    dataset_fingerprint: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
    ema: Any = None,
    training_context: Mapping[str, Any] | None = None,
    global_step: int = 0,
) -> None:
    """Save model and optimizer state atomically."""

    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "rng_states": capture_rng_state(),
        "resolved_config": dict(resolved_config),
        "dataset_fingerprint": dict(dataset_fingerprint),
        "extra": dict(extra or {}),
        "training_identity_sha256": training_identity_sha256(
            resolved_config, dataset_fingerprint, training_context=training_context
        ),
        "training_identity_context": dict(training_context or {}),
        "output_semantics": "conditional_positive",
        "ema": ema.state_dict() if ema is not None else None,
        "ema_model": ema.model_state_dict() if ema is not None else None,
    }
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    expected_fingerprint: Mapping[str, str] | None = None,
    allow_fingerprint_mismatch: bool = False,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
    adopt_checkpoint_output_semantics: bool = True,
    expected_training_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint after validating its dataset and training identity."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if expected_fingerprint is not None and dict(checkpoint.get("dataset_fingerprint", {})) != dict(
        expected_fingerprint
    ):
        if not allow_fingerprint_mismatch:
            raise CheckpointError(
                "Dataset contract, manifest, or normalization fingerprint differs from checkpoint"
            )
    if expected_training_identity_sha256 is not None:
        saved_identity = checkpoint.get("training_identity_sha256")
        if saved_identity != expected_training_identity_sha256:
            raise CheckpointError(
                "Training identity differs from checkpoint; start a new run for changed "
                "model, data, loss, optimizer, or scheduler semantics"
            )
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.load_state_dict(checkpoint["model"], strict=True)
    if adopt_checkpoint_output_semantics:
        heads = getattr(unwrapped, "heads", None)
        setter = getattr(heads, "set_depth_output_semantics", None)
        if callable(setter):
            setter(checkpoint_depth_output_semantics(checkpoint))
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("grad_scaler") is not None:
        scaler.load_state_dict(checkpoint["grad_scaler"])
    if restore_rng and checkpoint.get("rng_states") is not None:
        restore_rng_state(checkpoint["rng_states"])
    return checkpoint
