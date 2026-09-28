#!/usr/bin/env python3
"""Train any configured model through one XML-driven entry point.

Usage:
    python train.py configs/sar_topo_edge_kan.xml

The XML selects the model, data contract, hyperparameters, timestamped run
directory, and output policy. Traditional methods have no fitting stage and are
therefore evaluated only by the root-level ``test.py`` entry point.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any

from tools._neural_regression import run_training as run_learned_training
from tools.train_sar_topo_edge_kan import run_training as run_sar_topo_edge_kan_training
from utils.config import load_config
from utils.model_dispatch import configured_model_kind
from utils.run_paths import (
    RuntimeConfigError,
    allow_existing_output,
    ensure_output_is_available,
    optional_nonnegative_int,
    optional_path,
    runtime_section,
    started_at_run_id,
    train_output_path,
)


def _main_training_args(config_path: Path, config: Mapping[str, Any], output: Path) -> Namespace:
    train = runtime_section(config, "train")
    init_weights = str(train.get("init_weights", "raw"))
    if init_weights not in {"raw", "ema"}:
        raise RuntimeConfigError("runtime.train.init_weights must be 'raw' or 'ema'")
    init_transfer = str(train.get("init_transfer", "strict"))
    if init_transfer not in {"strict", "compatible"}:
        raise RuntimeConfigError(
            "runtime.train.init_transfer must be 'strict' or 'compatible'"
        )
    return Namespace(
        config=config_path,
        device=None,
        epochs=None,
        batch_size=None,
        num_workers=None,
        resume=optional_path(train.get("resume"), "runtime.train.resume"),
        init_checkpoint=optional_path(
            train.get("init_checkpoint"), "runtime.train.init_checkpoint"
        ),
        init_weights=init_weights,
        init_transfer=init_transfer,
        max_train_batches=optional_nonnegative_int(
            train.get("max_train_batches"), "runtime.train.max_train_batches"
        ),
        max_val_batches=optional_nonnegative_int(
            train.get("max_val_batches"), "runtime.train.max_val_batches"
        ),
        no_amp=False,
        seed=None,
        allow_fingerprint_mismatch=bool(
            train.get("allow_fingerprint_mismatch", False)
        ),
        output=output,
    )


def _learned_training_args(config_path: Path, config: Mapping[str, Any], output: Path) -> Namespace:
    train = runtime_section(config, "train")
    init_checkpoint = optional_path(
        train.get("init_checkpoint"), "runtime.train.init_checkpoint"
    )
    if init_checkpoint is not None:
        raise RuntimeConfigError(
            "learned comparison training does not implement init_checkpoint; "
            "use runtime.train.resume with a last_raw.pth checkpoint instead"
        )
    return Namespace(
        config=config_path,
        device=None,
        epochs=None,
        batch_size=None,
        num_workers=None,
        max_train_batches=optional_nonnegative_int(
            train.get("max_train_batches"), "runtime.train.max_train_batches"
        ),
        max_val_batches=optional_nonnegative_int(
            train.get("max_val_batches"), "runtime.train.max_val_batches"
        ),
        no_amp=False,
        seed=None,
        resume=optional_path(train.get("resume"), "runtime.train.resume"),
        allow_fingerprint_mismatch=bool(
            train.get("allow_fingerprint_mismatch", False)
        ),
        output=output,
    )


def run_from_config(config_path: Path) -> Path:
    """Dispatch one XML configuration without accepting model-specific CLI options."""

    path = config_path.expanduser().resolve(strict=True)
    config = load_config(path)
    kind, identifier = configured_model_kind(config)
    run_id = started_at_run_id(config)
    if kind == "traditional":
        raise RuntimeConfigError(
            "Traditional models have no training stage. Test them with "
            "`python test.py <training-runs-directory> --traditional` "
            "(traditional evaluation is enabled by default)."
        )

    train = runtime_section(config, "train")
    resume = optional_path(train.get("resume"), "runtime.train.resume")
    if resume is not None:
        configured_output = optional_path(train.get("output"), "runtime.train.output")
        output = resume.resolve().parent
        if configured_output is not None and configured_output != output:
            raise RuntimeConfigError(
                "runtime.train.output must equal the parent directory of "
                "runtime.train.resume"
            )
    else:
        output = train_output_path(config, run_id)
        ensure_output_is_available(
            output,
            allow_existing=allow_existing_output(config, "train"),
            operation="training",
        )
    if kind == "sar_topo_edge_kan":
        return run_sar_topo_edge_kan_training(_main_training_args(path, config, output))
    return run_learned_training(_learned_training_args(path, config, output), identifier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Model XML configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_from_config(args.config)
    logging.getLogger("training").info("Training complete | output=%s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
