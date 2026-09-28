#!/usr/bin/env python3
"""Repeat every trainable experiment on the fixed split with different seeds.

Usage:
    python validate_random_seeds.py          # k from base.xml (default: 5)
    python validate_random_seeds.py 10
    python validate_random_seeds.py --resume
    python validate_random_seeds.py --resume runs/.../random_seed_validation/k5/<session>

This entry never repartitions the dataset. Every
repeat uses the configured train split for optimization, the configured val
split for checkpoint selection/early stopping, and evaluates the selected
checkpoint once on the same fixed test split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET


LOGGER = logging.getLogger("random_seed_validation")
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_COMPLETION_FILES = (
    "best_raw.pth",
    "resolved_config.json",
    "training_summary.json",
)
EVALUATION_COMPLETION_FILES = (
    "summary.json",
    "resolved_config.json",
    "metrics_by_sample.csv",
    "metrics_by_event.csv",
    "metrics_by_train_depth_bin.csv",
)
PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_NAME = "fixed_split_repeated_random_seeds"
MAX_NUMPY_SEED = 2**32 - 1


class RandomSeedJobError(RuntimeError):
    """Raised when one isolated model/seed worker exits unsuccessfully."""


class RandomSeedProtocolError(RuntimeError):
    """Raised before incompatible repeated-run results can be mixed."""


class CudaHealthError(RuntimeError):
    """Raised when a formal repeated run cannot use the configured CUDA device."""


def seed_schedule(base_seed: int, stride: int, k: int) -> tuple[int, ...]:
    """Return a deterministic, unique seed schedule accepted by NumPy/PyTorch."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if stride <= 0:
        raise ValueError("runtime.random_seed_validation.seed_stride must be positive")
    seeds = tuple(int(base_seed) + index * int(stride) for index in range(k))
    if len(set(seeds)) != k:
        raise ValueError("The random-seed schedule contains duplicates")
    if any(seed < 0 or seed > MAX_NUMPY_SEED for seed in seeds):
        raise ValueError(
            f"Every seed must lie in [0, {MAX_NUMPY_SEED}] for NumPy compatibility"
        )
    return seeds


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read repeated-run JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Repeated-run JSON is not a mapping: {path}")
    return value


def _typed_element(parent: ET.Element, name: str, type_name: str, value: Any) -> None:
    child = ET.SubElement(parent, name, {"type": type_name})
    child.text = "none" if value is None else str(value)


def write_seed_config(
    source_config: Path,
    destination: Path,
    *,
    training_output: Path,
    session_root: Path,
    seed: int,
    seed_index: int,
    k: int,
    runtime_source_sha256: str,
    dataset_identity: Mapping[str, Any],
    resume_checkpoint: Path | None = None,
) -> Path:
    """Write a minimal overlay while preserving the configured dataset split."""

    from utils.config import load_config
    from utils.experiment_inventory import resolved_config_sha256
    from utils.misc import atomic_write_text
    from utils.model_dispatch import configured_model_kind

    source = source_config.expanduser().resolve(strict=True)
    resolved = load_config(source)
    kind, _ = configured_model_kind(resolved)
    if kind == "traditional":
        raise ValueError("Traditional methods have no random-seed training stage")

    root = ET.Element("config")
    ET.SubElement(root, "include", {"path": str(source)})
    _typed_element(root, "runs_root", "path", session_root)
    _typed_element(root, "artifacts_root", "path", training_output.parent / "artifacts")
    _typed_element(root, "seed", "int", seed)

    runtime = ET.SubElement(root, "runtime")
    train = ET.SubElement(runtime, "train")
    _typed_element(train, "output", "path", training_output)
    _typed_element(
        train,
        "resume",
        "path" if resume_checkpoint is not None else "none",
        resume_checkpoint,
    )
    _typed_element(train, "allow_existing_output", "bool", "false")
    evaluation = ET.SubElement(runtime, "evaluation")
    _typed_element(evaluation, "split", "str", "test")

    if kind == "sar_topo_edge_kan":
        loss = ET.SubElement(root, "loss")
        _typed_element(
            loss,
            "frozen_depth_weights_artifact",
            "path",
            training_output / "frozen_depth_weights.json",
        )

    repeated = ET.SubElement(root, "random_seed_validation")
    _typed_element(repeated, "protocol", "str", PROTOCOL_NAME)
    _typed_element(repeated, "k", "int", k)
    _typed_element(repeated, "seed_index", "int", seed_index)
    _typed_element(repeated, "seed", "int", seed)
    _typed_element(repeated, "fixed_split", "bool", "true")
    _typed_element(repeated, "source_config", "path", source)
    _typed_element(
        repeated,
        "source_config_sha256",
        "str",
        resolved_config_sha256(source),
    )
    _typed_element(
        repeated,
        "runtime_source_sha256",
        "str",
        runtime_source_sha256,
    )
    dataset_payload = dict(dataset_identity)
    for config_key, identity_key in (
        ("dataset_contract_sha256", "contract_sha256"),
        ("dataset_manifest_sha256", "manifest_sha256"),
        ("train_stats_sha256", "train_stats_sha256"),
    ):
        _typed_element(
            repeated,
            config_key,
            "str",
            dataset_payload[identity_key],
        )
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
    atomic_write_text(destination, xml + "\n")
    return destination


def _training_is_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in TRAINING_COMPLETION_FILES)


def _evaluation_is_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in EVALUATION_COMPLETION_FILES)


def _completed_job_result(
    model_root: Path,
    *,
    model_id: str,
    display_name: str,
    seed: int,
    seed_index: int,
    source_config: Path,
) -> dict[str, Any] | None:
    """Recover one complete model/seed result from durable summaries."""

    from utils.reporting import scalar_row

    if not _training_is_complete(model_root / "train"):
        return None
    evaluation_root = model_root / "test"
    if not _evaluation_is_complete(evaluation_root):
        return None
    return scalar_row(
        {
            **_load_mapping(evaluation_root / "summary.json"),
            "model": model_id,
            "display_name": display_name,
            "seed": int(seed),
            "seed_index": int(seed_index),
            "split": "test",
            "evaluation_role": "fixed_test",
            "source_config": str(source_config),
        }
    )


def _archive_incomplete_attempt(model_root: Path) -> Path:
    """Move a non-resumable partial attempt aside without deleting evidence."""

    archive = (
        model_root
        / "failed_attempts"
        / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    )
    archive.mkdir(parents=True, exist_ok=False)
    moved = False
    for name in ("train", "test", "artifacts", "job_result.json", "job_status.json"):
        source = model_root / name
        if source.exists():
            source.rename(archive / name)
            moved = True
    if not moved:
        archive.rmdir()
        raise RuntimeError(f"No incomplete attempt exists below {model_root}")
    return archive


def _resume_checkpoint_or_archive(model_root: Path, *, model_kind: str) -> Path | None:
    training = model_root / "train"
    if not training.exists() or _training_is_complete(training):
        return None
    checkpoint_name = "last.pth" if model_kind == "sar_topo_edge_kan" else "last_raw.pth"
    checkpoint = training / checkpoint_name
    if checkpoint.is_file():
        return checkpoint
    archive = _archive_incomplete_attempt(model_root)
    LOGGER.warning(
        "Archived incomplete attempt without a resumable checkpoint: %s", archive
    )
    return None


def _isolated_worker_environment(mkl_threading_layer: str) -> dict[str, str]:
    layer = str(mkl_threading_layer).strip().upper()
    if layer != "GNU":
        raise ValueError(
            "runtime.random_seed_validation.worker_mkl_threading_layer must be GNU "
            "because this PyTorch build loads libgomp"
        )
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = layer
    return environment


def _require_healthy_cuda(required: bool) -> None:
    if not required:
        return
    try:
        probe = subprocess.run(
            ("nvidia-smi", "-L"),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CudaHealthError(
            f"Random-seed validation requires a healthy CUDA device: {exc}"
        ) from exc
    if probe.returncode != 0 or not probe.stdout.strip():
        detail = (probe.stderr or probe.stdout).strip()
        raise CudaHealthError(
            "Random-seed validation requires CUDA, but nvidia-smi failed "
            f"({detail or f'exit code {probe.returncode}'})."
        )


def _worker_preflight(*, require_cuda: bool) -> None:
    """Import NumPy before PyTorch, then exercise the optional CUDA context."""

    import numpy as np
    import torch

    if not np.isfinite(np.asarray([0.0], dtype=np.float32)).all():
        raise RuntimeError("NumPy worker preflight produced a non-finite value")
    device_name = "cpu"
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Worker preflight requires CUDA, but it is unavailable")
        probe = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        device_name = torch.cuda.get_device_name(probe.device)
        del probe
    print(
        "Worker preflight passed | "
        f"MKL_THREADING_LAYER={os.environ.get('MKL_THREADING_LAYER', '<unset>')} | "
        f"NumPy={np.__version__} | PyTorch={torch.__version__} | device={device_name}",
        flush=True,
    )


def _preflight_isolated_worker(
    *, mkl_threading_layer: str, require_cuda: bool
) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-preflight"]
    if require_cuda:
        command.append("--require-cuda")
    completed = subprocess.run(
        command,
        check=False,
        env=_isolated_worker_environment(mkl_threading_layer),
    )
    if completed.returncode != 0:
        raise RandomSeedJobError(
            "Isolated worker preflight failed; no model job was started."
        )


def _worker_require_cuda(config: Mapping[str, Any]) -> None:
    from utils.run_paths import runtime_section

    required = bool(
        runtime_section(config, "random_seed_validation").get("require_cuda", True)
    )
    if not required:
        return
    import torch

    if str(config.get("device", "auto")).lower() == "cpu":
        raise RuntimeError("Formal random-seed validation cannot use device=cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the required CUDA device")
    probe = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    del probe


def _release_cuda_context_objects() -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _worker_run(
    config_path: Path,
    run_directory: Path,
    model_root: Path,
    seed: int,
    seed_index: int,
    source_config: Path,
) -> dict[str, Any]:
    """Train and fixed-test one model/seed combination in an isolated process."""

    # Import order matters for the GNU OpenMP runtime used by this PyTorch build.
    import numpy as np
    import torch

    del np, torch
    import train as unified_train
    from utils.config import load_config
    from utils.experiment_inventory import resolved_config_sha256
    from utils.evaluation_suite import evaluate_deep_run
    from utils.experiment_catalog import experiment_display_name, experiment_id
    from utils.misc import atomic_write_json
    from utils.reporting import scalar_row

    config_path = config_path.expanduser().resolve(strict=True)
    run_directory = run_directory.expanduser().resolve()
    model_root = model_root.expanduser().resolve()
    source_config = source_config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    repeated = config.get("random_seed_validation", {})
    if str(repeated.get("protocol", "")) != PROTOCOL_NAME:
        raise RuntimeError("Worker received an unsupported repeated-seed protocol")
    if not bool(repeated.get("fixed_split", False)):
        raise RuntimeError("Repeated-seed worker requires the immutable fixed split")
    if int(repeated.get("seed", -1)) != int(seed) or int(config["seed"]) != int(seed):
        raise RuntimeError("Worker seed differs from the generated configuration")
    if int(repeated.get("seed_index", -1)) != int(seed_index):
        raise RuntimeError("Worker seed index differs from the generated configuration")
    expected_source_sha256 = str(repeated.get("source_config_sha256", ""))
    if resolved_config_sha256(source_config) != expected_source_sha256:
        raise RuntimeError(
            "The source model configuration changed before worker startup"
        )
    current_runtime_sha256, _ = _campaign_runtime_sha256()
    if str(repeated.get("runtime_source_sha256", "")) != current_runtime_sha256:
        raise RuntimeError("Runtime source code changed before worker startup")
    current_dataset = _dataset_identity((source_config,))
    for config_key, identity_key in (
        ("dataset_contract_sha256", "contract_sha256"),
        ("dataset_manifest_sha256", "manifest_sha256"),
        ("train_stats_sha256", "train_stats_sha256"),
    ):
        if str(repeated.get(config_key, "")) != str(current_dataset[identity_key]):
            raise RuntimeError(
                f"Dataset identity changed before worker startup ({identity_key})"
            )
    _worker_require_cuda(config)

    model_id = experiment_id(config)
    display_name = experiment_display_name(config)
    model_root.mkdir(parents=True, exist_ok=True)
    status_path = model_root / "job_status.json"
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "model": model_id,
            "display_name": display_name,
            "seed": int(seed),
            "seed_index": int(seed_index),
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        if _training_is_complete(run_directory):
            completed_run = run_directory
            print(
                f"Reusing completed training | model={display_name} | seed={seed}",
                flush=True,
            )
        else:
            completed_run = unified_train.run_from_config(config_path)
            if completed_run.expanduser().resolve() != run_directory:
                raise RuntimeError(
                    "Worker received an unexpected training output: "
                    f"{completed_run} != {run_directory}"
                )
        if not _training_is_complete(completed_run):
            raise RuntimeError(
                f"Training returned without complete metadata: {completed_run}"
            )
        _release_cuda_context_objects()

        evaluation_root = model_root / "test"
        if _evaluation_is_complete(evaluation_root):
            summary: Mapping[str, Any] = _load_mapping(
                evaluation_root / "summary.json"
            )
            print(
                f"Reusing completed test | model={display_name} | seed={seed}",
                flush=True,
            )
        else:
            print(
                f"Testing {display_name} | seed={seed} | split=test",
                flush=True,
            )
            summary = evaluate_deep_run(
                completed_run,
                evaluation_root,
                split="test",
            )
        result = scalar_row(
            {
                **summary,
                "model": model_id,
                "display_name": display_name,
                "seed": int(seed),
                "seed_index": int(seed_index),
                "split": "test",
                "evaluation_role": "fixed_test",
                "source_config": str(source_config),
            }
        )
        payload = {
            "status": "complete",
            "model": model_id,
            "display_name": display_name,
            "seed": int(seed),
            "seed_index": int(seed_index),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        atomic_write_json(model_root / "job_result.json", payload)
        atomic_write_json(
            status_path,
            {key: value for key, value in payload.items() if key != "result"},
        )
        return result
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "model": model_id,
                "display_name": display_name,
                "seed": int(seed),
                "seed_index": int(seed_index),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def _run_isolated_job(
    *,
    overlay: Path,
    training_output: Path,
    model_root: Path,
    seed: int,
    seed_index: int,
    source_config: Path,
    mkl_threading_layer: str,
) -> None:
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(overlay),
        "--run-directory",
        str(training_output),
        "--model-root",
        str(model_root),
        "--seed",
        str(seed),
        "--seed-index",
        str(seed_index),
        "--source-config",
        str(source_config),
    )
    completed = subprocess.run(
        command,
        check=False,
        env=_isolated_worker_environment(mkl_threading_layer),
    )
    if completed.returncode != 0:
        signal_text = (
            f"signal {-completed.returncode}"
            if completed.returncode < 0
            else f"exit code {completed.returncode}"
        )
        raise RandomSeedJobError(
            f"Isolated worker failed with {signal_text}: model_root={model_root}. "
            "Completed jobs are preserved; continue with "
            "`python validate_random_seeds.py --resume`."
        )


def _campaign_runtime_sha256() -> tuple[str, int]:
    """Fingerprint shared runtime sources plus this controller/worker script."""

    from utils.experiment_inventory import runtime_source_sha256

    shared_sha256, shared_count = runtime_source_sha256(PROJECT_ROOT)
    script = Path(__file__).resolve()
    payload = script.read_bytes()
    digest = hashlib.sha256()
    digest.update(shared_sha256.encode("ascii"))
    digest.update(script.name.encode("utf-8"))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest(), shared_count + 1


def _dataset_identity(model_configs: Iterable[Path]) -> dict[str, Any]:
    """Verify that every model uses the same immutable train/val/test assets."""

    from datasets.contract import DatasetContract, sha256_file
    from utils.config import load_config

    reference_paths: tuple[Path, Path, Path] | None = None
    for config_path in model_configs:
        config = load_config(config_path)
        paths = (
            Path(config["dataset"]["contract"]).expanduser().resolve(strict=True),
            Path(config["dataset"]["manifest"]).expanduser().resolve(strict=True),
            Path(config["dataset"]["train_stats"]).expanduser().resolve(strict=True),
        )
        if reference_paths is None:
            reference_paths = paths
        elif paths != reference_paths:
            raise RandomSeedProtocolError(
                "All repeated experiments must use the same dataset contract, "
                "manifest, and train statistics"
            )
    if reference_paths is None:
        raise RandomSeedProtocolError("No trainable model configurations were found")
    contract_path, manifest_path, stats_path = reference_paths
    contract = DatasetContract.load(contract_path)
    contract.verify_fingerprints(include_normalization=True)
    if contract.manifest_path != manifest_path:
        raise RandomSeedProtocolError(
            "The configured manifest differs from the dataset contract manifest"
        )
    selected_stats = contract.payload["normalization"]["selected"]
    stats_sha256 = sha256_file(stats_path)
    if str(selected_stats["sha256"]) != stats_sha256:
        raise RandomSeedProtocolError(
            "The configured train statistics differ from the dataset contract"
        )
    counts = {
        split: int(contract.payload["sample_counts"][split])
        for split in ("train", "val", "test")
    }
    if any(counts[split] <= 0 for split in counts):
        raise RandomSeedProtocolError(
            f"Fixed split requires non-empty train/val/test counts: {counts}"
        )
    return {
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_stats": str(stats_path),
        "train_stats_sha256": stats_sha256,
        "split_counts": counts,
    }


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    k: int,
    seeds: tuple[int, ...],
    inventory: list[dict[str, str]],
    runtime_sha256: str,
    dataset_identity: Mapping[str, Any],
) -> None:
    if int(protocol.get("schema_version", 0)) != PROTOCOL_SCHEMA_VERSION:
        raise RandomSeedProtocolError(
            "The repeated-seed session uses an unsupported protocol version; start a new session."
        )
    if str(protocol.get("protocol", "")) != PROTOCOL_NAME:
        raise RandomSeedProtocolError("The repeated-seed protocol name differs")
    if int(protocol.get("k", 0)) != k or tuple(protocol.get("seeds", ())) != seeds:
        raise RandomSeedProtocolError("The repeated-seed schedule differs")
    if protocol.get("models") != inventory:
        raise RandomSeedProtocolError(
            "Resolved model configurations differ from the repeated-seed protocol"
        )
    if str(protocol.get("runtime_source_sha256", "")) != runtime_sha256:
        raise RandomSeedProtocolError(
            "Runtime Python sources changed; start a new repeated-seed session"
        )
    if protocol.get("dataset") != dataset_identity:
        raise RandomSeedProtocolError(
            "Dataset split/assets changed; start a new repeated-seed session"
        )


def _assert_inputs_unchanged(
    model_configs: tuple[Path, ...],
    *,
    expected_inventory: list[dict[str, str]],
    expected_runtime_sha256: str,
    expected_dataset_identity: Mapping[str, Any],
) -> None:
    from utils.experiment_inventory import config_inventory

    inventory = config_inventory(model_configs)
    runtime_sha256, _ = _campaign_runtime_sha256()
    dataset = _dataset_identity(model_configs)
    if inventory != expected_inventory:
        raise RandomSeedProtocolError(
            "A model configuration changed during the repeated-seed campaign"
        )
    if runtime_sha256 != expected_runtime_sha256:
        raise RandomSeedProtocolError(
            "Runtime source code changed during the repeated-seed campaign"
        )
    if dataset != expected_dataset_identity:
        raise RandomSeedProtocolError(
            "Dataset assets changed during the repeated-seed campaign"
        )


def latest_resumable_session(runs_root: Path, k: int) -> Path:
    root = runs_root / "random_seed_validation" / f"k{k}"
    candidates: list[Path] = []
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not (child / "protocol.json").is_file():
                continue
            status_path = child / "status.json"
            status = (
                _load_mapping(status_path).get("status")
                if status_path.is_file()
                else None
            )
            if status != "complete":
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"No resumable k={k} repeated-seed session exists below {root}"
        )
    return max(
        candidates,
        key=lambda path: (
            (path / "status.json").stat().st_mtime_ns
            if (path / "status.json").is_file()
            else path.stat().st_mtime_ns,
            path.name,
        ),
    )


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from utils.reporting import numerical_summary_rows

    sanitized = [
        {
            key: value
            for key, value in row.items()
            if key not in {"seed", "seed_index"}
        }
        for row in results
    ]
    return numerical_summary_rows(
        sanitized,
        group_keys=("model", "display_name", "split"),
    )


def _metric(
    summary: list[dict[str, Any]], model: str, name: str
) -> tuple[float, float]:
    row = next(
        item
        for item in summary
        if item.get("model") == model
        and item.get("split") == "test"
        and item.get("metric") == name
    )
    return float(row["mean"]), float(row["std"])


def _display_results(results: list[dict[str, Any]]) -> None:
    LOGGER.info("━" * 132)
    LOGGER.info(
        "%-32s | %10s | %8s | %8s | %8s | %9s | %10s | %12s",
        "Model",
        "Seed",
        "MAE",
        "RMSE",
        "R²",
        "Event MAE",
        "sample/s",
        "parameters",
    )
    LOGGER.info("─" * 132)
    for row in results:
        LOGGER.info(
            "%-32s | %10d | %8.4f | %8.4f | %8.4f | %9.4f | %10.2f | %12s",
            str(row.get("display_name", row["model"]))[:32],
            int(row["seed"]),
            float(row.get("pixel_micro_mae", float("nan"))),
            float(row.get("pixel_micro_rmse", float("nan"))),
            float(row.get("pixel_micro_r2", float("nan"))),
            float(row.get("event_macro_mae", float("nan"))),
            float(row.get("efficiency_samples_per_second", float("nan"))),
            f"{int(row.get('total_parameters', 0)):,}",
        )
    LOGGER.info("━" * 132)


def _display_aggregate(
    summary: list[dict[str, Any]], model_configs: tuple[Path, ...]
) -> None:
    from utils.config import load_config
    from utils.experiment_catalog import experiment_display_name, experiment_id

    LOGGER.info("Fixed-test mean ± sample standard deviation across random seeds")
    LOGGER.info("━" * 126)
    LOGGER.info(
        "%-34s | %-20s | %-20s | %-20s | %-20s",
        "Model",
        "MAE (m)",
        "RMSE (m)",
        "R²",
        "Event MAE (m)",
    )
    LOGGER.info("─" * 126)
    for config_path in model_configs:
        config = load_config(config_path)
        model = experiment_id(config)
        LOGGER.info(
            "%-34s | %8.4f ± %-8.4f | %8.4f ± %-8.4f | %8.4f ± %-8.4f | %8.4f ± %-8.4f",
            experiment_display_name(config)[:34],
            *_metric(summary, model, "pixel_micro_mae"),
            *_metric(summary, model, "pixel_micro_rmse"),
            *_metric(summary, model, "pixel_micro_r2"),
            *_metric(summary, model, "event_macro_mae"),
        )
    LOGGER.info("━" * 126)


def run_random_seed_validation(
    k: int | None,
    *,
    output: Path | None = None,
    resume: Path | None = None,
    model_ids: tuple[str, ...] | None = None,
    model_config_paths: tuple[Path, ...] | None = None,
) -> Path:
    if resume is not None and output is not None:
        raise ValueError("--resume and --output are mutually exclusive")
    if model_ids and model_config_paths:
        raise ValueError("--models and --model-config are mutually exclusive")

    from utils.config import load_config
    from utils.experiment_inventory import config_inventory
    from utils.experiment_catalog import (
        deep_learning_config_paths,
        experiment_display_name,
        experiment_id,
    )
    from utils.logging import setup_logging
    from utils.misc import atomic_write_json
    from utils.model_dispatch import configured_model_kind
    from utils.reporting import write_union_rows
    from utils.run_paths import runtime_section, started_at_run_id
    from utils.tensorboard import (
        add_metadata,
        add_scalars,
        create_summary_writer,
        flush,
    )

    main_config = load_config(PROJECT_ROOT / "configs" / "sar_topo_edge_kan.xml")
    repeated_config = runtime_section(main_config, "random_seed_validation")
    default_k = int(repeated_config.get("default_k", 5))
    selected_k = default_k if k is None else int(k)
    protocol: dict[str, Any] | None = None
    is_resuming = resume is not None
    if is_resuming:
        session = resume.expanduser().resolve(strict=True)
        if not session.is_dir():
            raise NotADirectoryError(f"Repeated-seed session is not a directory: {session}")
        protocol = _load_mapping(session / "protocol.json")
        protocol_k = int(protocol["k"])
        if k is not None and selected_k != protocol_k:
            raise ValueError(
                f"Requested k={selected_k} does not match resumed session k={protocol_k}"
            )
        selected_k = protocol_k
    if selected_k < 1:
        raise ValueError("k must be at least 1")
    seed_stride = int(repeated_config.get("seed_stride", 1009))
    seeds = seed_schedule(int(main_config["seed"]), seed_stride, selected_k)
    all_model_configs = deep_learning_config_paths(include_ablations=True)
    if model_config_paths:
        model_configs = tuple(dict.fromkeys(
            path.expanduser().resolve(strict=True) for path in model_config_paths
        ))
        if not model_configs:
            raise ValueError("--model-config requires at least one XML path")
        for path in model_configs:
            kind, _ = configured_model_kind(load_config(path))
            if kind == "traditional":
                raise ValueError(f"--model-config requires a trainable model: {path}")
    elif model_ids:
        requested = tuple(dict.fromkeys(str(value) for value in model_ids))
        by_id = {
            experiment_id(load_config(path)): path for path in all_model_configs
        }
        missing = sorted(set(requested).difference(by_id))
        if missing:
            raise ValueError(
                f"Unknown --models identifiers: {missing}; available={sorted(by_id)}"
            )
        model_configs = tuple(by_id[value] for value in requested)
    else:
        model_configs = all_model_configs
    inventory = config_inventory(model_configs)
    runtime_sha256, runtime_source_files = _campaign_runtime_sha256()
    dataset = _dataset_identity(model_configs)

    if protocol is not None:
        _validate_protocol(
            protocol,
            k=selected_k,
            seeds=seeds,
            inventory=inventory,
            runtime_sha256=runtime_sha256,
            dataset_identity=dataset,
        )
    else:
        session = (
            output.expanduser().resolve()
            if output is not None
            else Path(main_config["runs_root"])
            / "random_seed_validation"
            / f"k{selected_k}"
            / started_at_run_id(main_config)
        )
        if session.exists():
            raise FileExistsError(f"Repeated-seed output already exists: {session}")
        session.mkdir(parents=True)

    setup_logging(
        session / "random_seed_validation.log",
        show_python_warnings=bool(
            main_config["logging"].get("show_python_warnings", True)
        ),
    )
    worker_mkl_threading_layer = str(
        repeated_config.get("worker_mkl_threading_layer", "GNU")
    ).strip().upper()
    _isolated_worker_environment(worker_mkl_threading_layer)
    require_cuda = bool(repeated_config.get("require_cuda", True))
    _require_healthy_cuda(require_cuda)
    _preflight_isolated_worker(
        mkl_threading_layer=worker_mkl_threading_layer,
        require_cuda=require_cuda,
    )

    if protocol is None:
        protocol = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "k": selected_k,
            "base_seed": int(main_config["seed"]),
            "seed_stride": seed_stride,
            "seeds": list(seeds),
            "split_policy": {
                "train": "optimization",
                "val": "checkpoint_selection_and_early_stopping",
                "test": "post_training_evaluation_only",
                "repartitioned": False,
            },
            "dataset": dataset,
            "runtime_source_sha256": runtime_sha256,
            "runtime_source_files": runtime_source_files,
            "job_isolation": "one_subprocess_per_model_seed",
            "worker_mkl_threading_layer": worker_mkl_threading_layer,
            "models": inventory,
        }
        atomic_write_json(session / "protocol.json", protocol)

    LOGGER.info("━" * 88)
    LOGGER.info(
        "%s fixed-split random-seed validation | repeats=%d | models=%d | jobs=%d",
        "Resuming" if is_resuming else "Starting",
        selected_k,
        len(model_configs),
        selected_k * len(model_configs),
    )
    LOGGER.info("Seeds: %s", list(seeds))
    LOGGER.info(
        "Fixed split: train=%d | val=%d | test=%d",
        int(dataset["split_counts"]["train"]),
        int(dataset["split_counts"]["val"]),
        int(dataset["split_counts"]["test"]),
    )
    LOGGER.info("Output directory: %s", session)
    LOGGER.info("━" * 88)

    writer = create_summary_writer(
        session / "tensorboard",
        enabled=bool(main_config["logging"].get("tensorboard", False)),
        flush_seconds=int(main_config["logging"].get("tensorboard_flush_seconds", 30)),
        logger=LOGGER,
    )
    add_metadata(
        writer,
        {
            "operation": PROTOCOL_NAME,
            "k": selected_k,
            "models": len(model_configs),
            "seeds": list(seeds),
            "split": "fixed train/val/test",
        },
    )

    results: list[dict[str, Any]] = []
    total_jobs = selected_k * len(model_configs)
    visited = 0
    finished = 0
    try:
        for seed_index, seed in enumerate(seeds, start=1):
            seed_root = session / "seeds" / f"seed_{seed_index:02d}_{seed:010d}"
            for source_config in model_configs:
                _assert_inputs_unchanged(
                    model_configs,
                    expected_inventory=inventory,
                    expected_runtime_sha256=runtime_sha256,
                    expected_dataset_identity=dataset,
                )
                source = load_config(source_config)
                model_id = experiment_id(source)
                display_name = experiment_display_name(source)
                model_root = seed_root / "models" / model_id
                training_output = model_root / "train"
                visited += 1
                existing = _completed_job_result(
                    model_root,
                    model_id=model_id,
                    display_name=display_name,
                    seed=seed,
                    seed_index=seed_index,
                    source_config=source_config,
                )
                if existing is not None:
                    LOGGER.info(
                        "[%d/%d] Reusing completed %s | seed=%d (%d/%d)",
                        visited,
                        total_jobs,
                        display_name,
                        seed,
                        seed_index,
                        selected_k,
                    )
                    results.append(existing)
                    add_scalars(
                        writer,
                        existing,
                        step=seed_index,
                        prefix=f"random_seed_validation/{model_id}/test",
                    )
                    finished += 1
                    write_union_rows(session / "metrics_by_seed.csv", results)
                    atomic_write_json(
                        session / "status.json",
                        {
                            "status": "running",
                            "k": selected_k,
                            "models": len(model_configs),
                            "jobs_completed": finished,
                            "total_jobs": total_jobs,
                            "resumed": is_resuming,
                        },
                    )
                    continue

                model_kind, _ = configured_model_kind(source)
                _require_healthy_cuda(require_cuda)
                resume_checkpoint = (
                    _resume_checkpoint_or_archive(
                        model_root,
                        model_kind=model_kind,
                    )
                    if is_resuming
                    else None
                )
                overlay = write_seed_config(
                    source_config,
                    seed_root / "configs" / f"{model_id}.xml",
                    training_output=training_output,
                    session_root=session,
                    seed=seed,
                    seed_index=seed_index,
                    k=selected_k,
                    runtime_source_sha256=runtime_sha256,
                    dataset_identity=dataset,
                    resume_checkpoint=resume_checkpoint,
                )
                LOGGER.info(
                    "[%d/%d] Running isolated job %s | seed=%d (%d/%d)%s",
                    visited,
                    total_jobs,
                    display_name,
                    seed,
                    seed_index,
                    selected_k,
                    f" | resume={resume_checkpoint.name}"
                    if resume_checkpoint is not None
                    else "",
                )
                _run_isolated_job(
                    overlay=overlay,
                    training_output=training_output,
                    model_root=model_root,
                    seed=seed,
                    seed_index=seed_index,
                    source_config=source_config,
                    mkl_threading_layer=worker_mkl_threading_layer,
                )
                result = _completed_job_result(
                    model_root,
                    model_id=model_id,
                    display_name=display_name,
                    seed=seed,
                    seed_index=seed_index,
                    source_config=source_config,
                )
                if result is None:
                    raise RuntimeError(
                        f"Worker exited successfully without complete results: {model_root}"
                    )
                results.append(result)
                add_scalars(
                    writer,
                    result,
                    step=seed_index,
                    prefix=f"random_seed_validation/{model_id}/test",
                )
                flush(writer)
                write_union_rows(session / "metrics_by_seed.csv", results)
                finished += 1
                atomic_write_json(
                    session / "status.json",
                    {
                        "status": "running",
                        "k": selected_k,
                        "models": len(model_configs),
                        "jobs_completed": finished,
                        "total_jobs": total_jobs,
                        "resumed": is_resuming,
                    },
                )
    except BaseException as exc:
        write_union_rows(session / "metrics_by_seed.csv", results)
        atomic_write_json(
            session / "status.json",
            {
                "status": "failed",
                "k": selected_k,
                "models": len(model_configs),
                "jobs_completed": finished,
                "active_job_index": visited,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        if isinstance(exc, RandomSeedJobError):
            LOGGER.error("Random-seed validation stopped | %s", exc)
        else:
            LOGGER.exception("Random-seed validation stopped after an error")
        raise
    finally:
        if writer is not None:
            writer.close()

    summaries = _summary_rows(results)
    write_union_rows(session / "metrics_by_seed.csv", results)
    write_union_rows(session / "metrics_summary.csv", summaries)
    atomic_write_json(
        session / "summary.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL_NAME,
            "k": selected_k,
            "seeds": list(seeds),
            "dataset": dataset,
            "seed_results": results,
            "metric_summaries": summaries,
        },
    )
    atomic_write_json(
        session / "status.json",
        {
            "status": "complete",
            "k": selected_k,
            "models": len(model_configs),
            "jobs": total_jobs,
            "resumed": is_resuming,
        },
    )
    _display_results(results)
    _display_aggregate(summaries, model_configs)
    LOGGER.info("Complete random-seed validation report: %s", session)
    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "k",
        type=int,
        nargs="?",
        default=None,
        help=(
            "Number of random-seed repeats (default: "
            "runtime.random_seed_validation.default_k, currently 5)."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Run only the listed model identifiers. Defaults to the complete "
            "learned-model and ablation catalog."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        action="append",
        default=None,
        metavar="XML",
        help="Run only the explicitly listed trainable XML configurations.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="SESSION",
        help=(
            "Resume a session directory. Omit SESSION to select the newest "
            "unfinished session for the requested/default k."
        ),
    )
    parser.add_argument("--worker-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-cuda", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--source-config", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_preflight:
        _worker_preflight(require_cuda=bool(args.require_cuda))
        return 0
    if args.require_cuda:
        raise SystemExit("--require-cuda is only valid with --worker-preflight")
    if args.worker:
        required = {
            "--config": args.config,
            "--run-directory": args.run_directory,
            "--model-root": args.model_root,
            "--seed": args.seed,
            "--seed-index": args.seed_index,
            "--source-config": args.source_config,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"Missing required worker arguments: {', '.join(missing)}")
        _worker_run(
            args.config,
            args.run_directory,
            args.model_root,
            int(args.seed),
            int(args.seed_index),
            args.source_config,
        )
        return 0
    if any(
        value is not None
        for value in (
            args.config,
            args.run_directory,
            args.model_root,
            args.seed,
            args.seed_index,
            args.source_config,
        )
    ):
        raise SystemExit("Internal worker arguments require --worker")
    if args.resume is not None and args.output is not None:
        raise SystemExit("--resume and --output are mutually exclusive")
    if args.models and args.model_config:
        raise SystemExit("--models and --model-config are mutually exclusive")

    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "latest":
            from utils.config import load_config
            from utils.run_paths import runtime_section

            config = load_config(PROJECT_ROOT / "configs" / "sar_topo_edge_kan.xml")
            repeated = runtime_section(config, "random_seed_validation")
            selected_k = (
                int(repeated.get("default_k", 5)) if args.k is None else int(args.k)
            )
            resume_path = latest_resumable_session(
                Path(config["runs_root"]), selected_k
            )
        else:
            resume_path = Path(args.resume)
    try:
        run_random_seed_validation(
            args.k,
            output=args.output,
            resume=resume_path,
            model_ids=tuple(args.models) if args.models else None,
            model_config_paths=(
                tuple(args.model_config) if args.model_config else None
            ),
        )
    except CudaHealthError as exc:
        LOGGER.error("%s", exc)
        return 2
    except RandomSeedJobError:
        return 1
    except RandomSeedProtocolError as exc:
        LOGGER.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
