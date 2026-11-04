#!/usr/bin/env python3
"""Evaluate any configured model through one XML-driven entry point.

Usage:
    python evaluate.py configs/sar_topo_edge_kan.xml

The ``runtime.evaluation`` section selects the split, checkpoint, output, and
optional prediction export. If no checkpoint is specified, the latest completed
time-stamped training run is used (or ``runtime.evaluation.source_run``).
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping
import logging
from pathlib import Path
import time
from typing import Any

from tools._neural_regression import run_evaluation as run_learned_evaluation
from tools._run_terrain_baseline import run_model as run_traditional_evaluation
from tools.evaluate_sar_topo_edge_kan import run_evaluation as run_sar_topo_edge_kan_evaluation
from utils.config import load_config
from utils.logging import format_duration, setup_logging
from utils.model_dispatch import configured_model_kind
from utils.run_paths import (
    RuntimeConfigError,
    allow_existing_output,
    ensure_output_is_available,
    evaluation_checkpoint_path,
    evaluation_output_path,
    evaluation_split,
    optional_nonnegative_int,
    runtime_section,
    started_at_run_id,
)
from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


LOGGER = logging.getLogger("evaluation")


def _checkpoint(config: Mapping[str, Any]) -> Path:
    path = evaluation_checkpoint_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"configured evaluation checkpoint does not exist: {path}. "
            "Run python train.py <config.xml> first, or set runtime.evaluation.checkpoint."
        )
    return path


def _start_evaluation_logging(
    config: Mapping[str, Any],
    *,
    output: Path,
    model: str,
    split: str,
    checkpoint: Path,
) -> tuple[Any | None, float]:
    """Initialize a concise evaluation log and its companion event stream."""

    setup_logging(
        output / "evaluate.log",
        show_python_warnings=bool(config["logging"].get("show_python_warnings", True)),
    )
    LOGGER.info("━" * 78)
    LOGGER.info("Evaluation started | model=%s | split=%s", model, split)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Output directory: %s", output)
    LOGGER.info("━" * 78)
    writer = create_summary_writer(
        output / "tensorboard",
        enabled=bool(config["logging"].get("tensorboard", False)),
        flush_seconds=int(config["logging"].get("tensorboard_flush_seconds", 30)),
        logger=LOGGER,
    )
    add_metadata(
        writer,
        {
            "model": model,
            "split": split,
            "checkpoint": checkpoint,
            "source_run": checkpoint.parent.name,
        },
    )
    return writer, time.perf_counter()


def _finish_evaluation_logging(
    writer: Any | None,
    *,
    summary: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    """Persist scalar results and emit a compact completion record."""

    try:
        add_scalars(writer, summary, step=0, prefix="evaluation")
        add_scalars(writer, {"elapsed_seconds": elapsed_seconds}, step=0, prefix="system")
        flush(writer)
    finally:
        if writer is not None:
            writer.close()
    pixel_mae = summary.get("pixel_micro_mae")
    event_mae = summary.get("event_macro_mae")
    if isinstance(pixel_mae, (int, float)) and isinstance(event_mae, (int, float)):
        LOGGER.info(
            "Evaluation complete | elapsed=%s | pixel_micro_mae=%.5f | event_macro_mae=%.5f",
            format_duration(elapsed_seconds),
            float(pixel_mae),
            float(event_mae),
        )
    else:
        LOGGER.info("Evaluation complete | elapsed=%s", format_duration(elapsed_seconds))


def run_from_config(config_path: Path) -> dict[str, Any]:
    """Dispatch evaluation based solely on the selected XML configuration."""

    path = config_path.expanduser().resolve(strict=True)
    config = load_config(path)
    kind, identifier = configured_model_kind(config)
    evaluation = runtime_section(config, "evaluation")
    split = evaluation_split(config)
    max_batches = optional_nonnegative_int(
        evaluation.get("max_batches"), "runtime.evaluation.max_batches"
    )
    if kind == "traditional":
        output = evaluation_output_path(config, started_at_run_id(config))
        ensure_output_is_available(
            output,
            allow_existing=allow_existing_output(config, "evaluation"),
            operation="evaluation",
        )
        return run_traditional_evaluation(
            config,
            split,
            identifier,
            output,
            max_batches,
            bool(evaluation.get("save_predictions", False)),
        )

    checkpoint = _checkpoint(config)
    output = evaluation_output_path(config, checkpoint.parent.name)
    ensure_output_is_available(
        output,
        allow_existing=allow_existing_output(config, "evaluation"),
        operation="evaluation",
    )
    writer, started = _start_evaluation_logging(
        config,
        output=output,
        model=str(config.get("model", {}).get("display_name", identifier)),
        split=split,
        checkpoint=checkpoint,
    )
    if kind == "sar_topo_edge_kan":
        weights = str(evaluation.get("weights", "raw"))
        if weights != "raw":
            raise RuntimeConfigError("Only raw model weights are supported")
        validity_mask = evaluation.get("validity_mask")
        try:
            summary = run_sar_topo_edge_kan_evaluation(
                path,
                checkpoint,
                split,
                str(config.get("device", "auto")),
                output,
                bool(evaluation.get("save_predictions", False)),
                max_batches,
                weights,
                None if validity_mask is None else str(validity_mask),
                None,
            )
        except Exception:
            if writer is not None:
                writer.close()
            raise
        _finish_evaluation_logging(
            writer, summary=summary, elapsed_seconds=time.perf_counter() - started
        )
        return summary

    try:
        if str(evaluation.get("weights", "raw")) != "raw":
            raise RuntimeConfigError("learned comparator evaluation supports only raw weights")
        summary = run_learned_evaluation(
            Namespace(
                config=path,
                checkpoint=checkpoint,
                split=split,
                device=str(config.get("device", "auto")),
                num_workers=None,
                max_batches=max_batches,
                output=output,
            ),
            identifier,
        )
    except Exception:
        if writer is not None:
            writer.close()
        raise
    _finish_evaluation_logging(
        writer, summary=summary, elapsed_seconds=time.perf_counter() - started
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Model XML configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
