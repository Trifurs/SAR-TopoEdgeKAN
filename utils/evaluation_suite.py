"""Run-directory discovery and reproducible checkpoint evaluation helpers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from tools._neural_regression import run_evaluation_from_config as evaluate_comparator
from tools.evaluate_sar_topo_edge_kan import run_evaluation_from_config as evaluate_sar_topo_edge_kan
from utils.experiment_catalog import experiment_display_name, experiment_id
from utils.model_dispatch import configured_model_kind


REQUIRED_RUN_FILES = ("best_raw.pth", "resolved_config.json", "training_summary.json")


def load_resolved_run_config(run_directory: Path) -> dict[str, Any]:
    path = run_directory / "resolved_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read resolved training config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Resolved training config is not a mapping: {path}")
    return value


def completed_run_directories(search_root: Path) -> list[Path]:
    """Find only checkpoint runs that carry complete provenance metadata."""

    root = search_root.expanduser().resolve(strict=True)
    candidates: list[Path] = []
    if root.is_dir() and all((root / name).is_file() for name in REQUIRED_RUN_FILES):
        candidates.append(root)
    if root.is_dir():
        for checkpoint in root.rglob("best_raw.pth"):
            run = checkpoint.parent
            if all((run / name).is_file() for name in REQUIRED_RUN_FILES):
                candidates.append(run)
    return sorted(set(candidates))


def select_runs_from_roots(
    search_roots: tuple[Path, ...], *, latest_only: bool = True
) -> list[Path]:
    """Select runs across explicit roots while keeping CV outputs out of scope."""

    runs = sorted(
        {
            run
            for root in search_roots
            for run in completed_run_directories(root)
        }
    )
    if not runs:
        return []
    by_experiment: dict[str, list[Path]] = {}
    for run in runs:
        config = load_resolved_run_config(run)
        kind, _ = configured_model_kind(config)
        if kind == "traditional":
            continue
        by_experiment.setdefault(experiment_id(config), []).append(run)
    if not latest_only:
        return sorted(run for values in by_experiment.values() for run in values)
    selected: list[Path] = []
    for values in by_experiment.values():
        selected.append(
            max(
                values,
                key=lambda path: (
                    (path / "training_summary.json").stat().st_mtime_ns,
                    path.name,
                ),
            )
        )
    return sorted(selected, key=lambda path: experiment_id(load_resolved_run_config(path)))


def evaluate_deep_run(
    run_directory: Path,
    output: Path,
    *,
    split: str = "test",
    device: str | None = None,
    max_batches: int | None = None,
    save_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint using the exact config saved by its training run."""

    run = run_directory.expanduser().resolve(strict=True)
    config = deepcopy(load_resolved_run_config(run))
    kind, identifier = configured_model_kind(config)
    if kind == "traditional":
        raise ValueError(f"Traditional methods have no checkpoint run: {run}")
    selected_device = str(device or config.get("device", "auto"))
    selected_weights = "raw"
    checkpoint = run / "best_raw.pth"
    if kind == "sar_topo_edge_kan":
        summary = evaluate_sar_topo_edge_kan(
            config,
            checkpoint,
            split,
            selected_device,
            output,
            save_predictions,
            max_batches,
            selected_weights,
            None,
            None,
        )
    else:
        summary = evaluate_comparator(
            config,
            checkpoint,
            split,
            selected_device,
            output,
            identifier,
            max_batches=max_batches,
        )
    enriched = {
        **summary,
        "model": experiment_id(config),
        "display_name": experiment_display_name(config),
        "source_run": run.name,
        "run_directory": str(run),
        "split": split,
        "selected_checkpoint": str(checkpoint),
        "selected_weights": selected_weights,
    }
    from utils.misc import atomic_write_json

    atomic_write_json(output / "summary.json", enriched)
    return enriched
