#!/usr/bin/env python3
"""Test all completed model runs through one directory-driven entry point.

Usage:
    python test.py runs/sar_topo_edge_kan/train

The newest complete checkpoint for every discovered deep-learning experiment is
tested on the official test split. Traditional models are evaluated by default
and may be disabled with ``--no-traditional``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import logging
from pathlib import Path
from typing import Any

import torch

from tools._run_terrain_baseline import run_model as run_traditional_evaluation
from utils.config import load_config
from utils.evaluation_suite import (
    evaluate_deep_run,
    load_resolved_run_config,
    select_runs_from_roots,
)
from utils.experiment_catalog import (
    experiment_display_name,
    experiment_id,
    traditional_config_paths,
)
from utils.logging import setup_logging
from utils.misc import atomic_write_json
from utils.reporting import scalar_row, write_union_rows
from utils.run_paths import (
    ensure_output_is_available,
    optional_nonnegative_int,
    optional_path,
    runtime_section,
    started_at_run_id,
)
from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


LOGGER = logging.getLogger("test_suite")


def _release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _default_output(config: dict[str, Any]) -> Path:
    return Path(config["runs_root"]) / "test" / started_at_run_id(config)


def _display_results(rows: list[dict[str, Any]]) -> None:
    LOGGER.info("━" * 122)
    LOGGER.info(
        "%-34s | %9s | %9s | %9s | %9s | %10s | %12s",
        "Model",
        "MAE (m)",
        "RMSE (m)",
        "R²",
        "Event MAE",
        "sample/s",
        "parameters",
    )
    LOGGER.info("─" * 122)
    for row in rows:
        LOGGER.info(
            "%-34s | %9.4f | %9.4f | %9.4f | %9.4f | %10.2f | %12s",
            str(row.get("display_name", row.get("model", "unknown")))[:34],
            float(row.get("pixel_micro_mae", float("nan"))),
            float(row.get("pixel_micro_rmse", float("nan"))),
            float(row.get("pixel_micro_r2", float("nan"))),
            float(row.get("event_macro_mae", float("nan"))),
            float(row.get("efficiency_samples_per_second", float("nan"))),
            f"{int(row.get('total_parameters', 0)):,}",
        )
    LOGGER.info("━" * 122)


def run_suite(
    search_root: Path,
    *,
    include_traditional: bool | None = None,
    latest_only: bool | None = None,
    output: Path | None = None,
    device: str | None = None,
    max_batches: int | None = None,
    save_predictions: bool | None = None,
) -> Path:
    source = search_root.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Training-run search path is not a directory: {source}")
    # JSON run configs are intentionally loaded through the discovery helper;
    # use one source XML only to resolve the common output/logging policy.
    policy_config = load_config(
        Path(__file__).resolve().parent / "configs" / "sar_topo_edge_kan.xml"
    )
    test_config = runtime_section(policy_config, "test")
    if include_traditional is None:
        include_traditional = bool(test_config.get("include_traditional", True))
    if latest_only is None:
        latest_only = bool(test_config.get("latest_completed_only", True))
    if save_predictions is None:
        save_predictions = bool(test_config.get("save_predictions", False))
    if max_batches is None:
        max_batches = optional_nonnegative_int(
            test_config.get("max_batches"), "runtime.test.max_batches"
        )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("runtime.test.max_batches must be positive when provided")
    search_roots = [source]
    sibling_ablation = source.parent / "ablation"
    if source.name == "train" and sibling_ablation.is_dir():
        search_roots.append(sibling_ablation)
    deep_runs = select_runs_from_roots(
        tuple(search_roots), latest_only=latest_only
    )
    run_keys = Counter(
        (experiment_id(load_resolved_run_config(run)), run.name) for run in deep_runs
    )
    configured_output = optional_path(test_config.get("output"), "runtime.test.output")
    destination = (
        output.expanduser().resolve()
        if output is not None
        else configured_output
        if configured_output is not None
        else _default_output(policy_config)
    )
    allow_existing = bool(test_config.get("allow_existing_output", False))
    ensure_output_is_available(
        destination,
        allow_existing=allow_existing,
        operation="test suite",
    )
    destination.mkdir(parents=True, exist_ok=allow_existing)
    setup_logging(
        destination / "test.log",
        show_python_warnings=bool(
            policy_config["logging"].get("show_python_warnings", True)
        ),
    )
    LOGGER.info("━" * 78)
    LOGGER.info(
        "Test suite started | search roots=%s",
        ", ".join(str(path) for path in search_roots),
    )
    LOGGER.info(
        "Deep runs=%d | traditional=%s | selection=%s",
        len(deep_runs),
        "enabled" if include_traditional else "disabled",
        "latest completed run per model" if latest_only else "all completed runs",
    )
    LOGGER.info("Output directory: %s", destination)
    LOGGER.info("━" * 78)
    if not deep_runs and not include_traditional:
        raise FileNotFoundError(
            f"No completed deep-learning runs found below {source}, and traditional "
            "evaluation was disabled."
        )
    writer = create_summary_writer(
        destination / "tensorboard",
        enabled=bool(policy_config["logging"].get("tensorboard", False)),
        flush_seconds=int(
            policy_config["logging"].get("tensorboard_flush_seconds", 30)
        ),
        logger=LOGGER,
    )
    add_metadata(
        writer,
        {
            "operation": "official_test_suite",
            "source": source,
            "search_roots": ", ".join(str(path) for path in search_roots),
            "latest_only": latest_only,
            "include_traditional": include_traditional,
        },
    )
    rows: list[dict[str, Any]] = []
    try:
        for index, run in enumerate(deep_runs, start=1):
            run_config = load_resolved_run_config(run)
            model_id = experiment_id(run_config)
            output_run_id = run.name
            if run_keys[(model_id, run.name)] > 1:
                suffix = hashlib.sha256(str(run).encode("utf-8")).hexdigest()[:10]
                output_run_id = f"{run.name}-{suffix}"
            model_output = destination / "deep_learning" / model_id / output_run_id
            LOGGER.info(
                "[%d/%d] Testing %s | run=%s",
                index,
                len(deep_runs),
                experiment_display_name(run_config),
                run.name,
            )
            row = evaluate_deep_run(
                run,
                model_output,
                split="test",
                device=device,
                max_batches=max_batches,
                save_predictions=save_predictions,
            )
            rows.append(scalar_row(row))
            add_scalars(writer, row, step=0, prefix=f"test/{model_id}")
            write_union_rows(destination / "metrics_by_model.csv", rows)
            _release_model_memory()

        if include_traditional:
            for config_path in traditional_config_paths():
                config = load_config(config_path)
                method = str(config["compare"]["method"])
                method_output = destination / "traditional" / method
                LOGGER.info("Testing traditional method %s", method)
                summary = run_traditional_evaluation(
                    config,
                    "test",
                    method,
                    method_output,
                    max_batches,
                    save_predictions,
                )
                row = {
                    **summary,
                    "model": experiment_id(config),
                    "display_name": experiment_display_name(config),
                    "source_run": "deterministic",
                    "run_directory": "",
                    "split": "test",
                }
                atomic_write_json(method_output / "summary.json", row)
                rows.append(scalar_row(row))
                add_scalars(writer, row, step=0, prefix=f"test/{method}")
                write_union_rows(destination / "metrics_by_model.csv", rows)
                # The traditional runner owns a per-method log; restore the suite log.
                setup_logging(destination / "test.log")
        flush(writer)
    except Exception as exc:
        setup_logging(destination / "test.log")
        write_union_rows(destination / "metrics_by_model.csv", rows)
        atomic_write_json(
            destination / "status.json",
            {
                "status": "failed",
                "models_completed": len(rows),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        LOGGER.exception("Test suite stopped after an error")
        raise
    finally:
        if writer is not None:
            writer.close()

    rows.sort(key=lambda row: (str(row.get("model_family", "")), str(row["model"])))
    write_union_rows(destination / "metrics_by_model.csv", rows)
    atomic_write_json(
        destination / "summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "search_roots": [str(path) for path in search_roots],
            "selection": "latest_per_model" if latest_only else "all_runs",
            "traditional_included": include_traditional,
            "models_evaluated": len(rows),
            "results": rows,
        },
    )
    atomic_write_json(
        destination / "status.json",
        {"status": "complete", "models_evaluated": len(rows)},
    )
    _display_results(rows)
    LOGGER.info("Complete test report: %s", destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        type=Path,
        help="Training directory to search recursively (for example, runs/.../train).",
    )
    parser.add_argument(
        "--traditional",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also evaluate all traditional models (default: enabled).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Test every completed repeat instead of only the newest run per model.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")
    run_suite(
        args.runs,
        include_traditional=args.traditional,
        latest_only=False if args.all_runs else None,
        output=args.output,
        device=args.device,
        max_batches=args.max_batches,
        save_predictions=args.save_predictions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
