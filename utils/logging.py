"""File/console logging and append-only metric CSV utilities."""

from __future__ import annotations

import csv
import io
import logging
import math
import warnings
from pathlib import Path
from typing import Any

from utils.misc import atomic_write_text


def setup_logging(
    log_path: Path | None = None,
    level: int = logging.INFO,
    *,
    show_python_warnings: bool = True,
) -> None:
    """Configure compact console logs and complete file logs for one operation."""

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    handlers: list[logging.Handler] = [console]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").disabled = not show_python_warnings


def configure_training_warning_filters(
    *, deterministic: bool, logger: logging.Logger
) -> None:
    """Condense one known PyTorch warning without hiding unrelated warnings.

    PyTorch currently has no bitwise-deterministic CUDA backward kernel for
    adaptive average pooling.  The project intentionally requests deterministic
    algorithms in ``warn_only`` mode, so a long framework warning otherwise
    interrupts the first epoch.  Keep all other warnings visible and record the
    reproducibility limitation once in a compact, searchable form.
    """

    if not deterministic:
        return
    warnings.filterwarnings(
        "ignore",
        message=(
            r"adaptive_avg_pool2d_backward_cuda does not have a deterministic "
            r"implementation.*"
        ),
        category=UserWarning,
    )
    logger.info(
        "Reproducibility note | deterministic mode is seed-controlled; "
        "PyTorch has no bitwise-deterministic CUDA backward for adaptive "
        "average pooling, so its known framework warning is condensed."
    )


def format_duration(seconds: float | None) -> str:
    """Format elapsed time compactly while retaining useful epoch precision."""

    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    remaining = max(0.0, float(seconds))
    hours, remaining = divmod(remaining, 3600.0)
    minutes, remaining = divmod(remaining, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{remaining:04.1f}"


def log_training_header(
    logger: logging.Logger,
    *,
    model: str,
    run_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    parameters: int | None = None,
    tensorboard_dir: Path | None = None,
) -> None:
    """Emit a stable, scannable training header instead of a raw config dump."""

    logger.info("━" * 78)
    logger.info("Training started | model=%s | device=%s", model, device)
    logger.info("Run directory: %s", run_dir)
    logger.info("Schedule: %d epochs | batch size: %d", epochs, batch_size)
    if parameters is not None:
        logger.info("Trainable parameters: %s", f"{parameters:,}")
    if tensorboard_dir is not None:
        logger.info("TensorBoard events: %s", tensorboard_dir)
    logger.info("━" * 78)


def log_epoch_summary(
    logger: logging.Logger,
    *,
    epoch: int,
    total_epochs: int,
    epoch_seconds: float,
    elapsed_seconds: float,
    eta_seconds: float | None,
    train_loss: float,
    metric_name: str | None,
    metric_value: float | None,
    best_metric: float | None,
    learning_rate: float,
    patience: int,
    patience_limit: int,
    minimum_epochs: int,
    improved: bool | None,
) -> None:
    """Print one complete epoch record with timing and early-stop state."""

    metric_text = "validation=skipped"
    if metric_name is not None and metric_value is not None:
        metric_text = f"val/{metric_name}={metric_value:.5f}"
    best_text = "best=--" if best_metric is None or not math.isfinite(best_metric) else f"best={best_metric:.5f}"
    change = "*" if improved else ""
    eligibility = ""
    if epoch < minimum_epochs:
        eligibility = f" (active from {minimum_epochs})"
    logger.info(
        "Epoch %03d/%03d | epoch=%s | train=%.5f | %s | %s%s | "
        "lr=%.2e | early-stop=%d/%d%s | elapsed=%s | eta=%s",
        epoch,
        total_epochs,
        format_duration(epoch_seconds),
        train_loss,
        metric_text,
        best_text,
        change,
        learning_rate,
        patience,
        patience_limit,
        eligibility,
        format_duration(elapsed_seconds),
        format_duration(eta_seconds),
    )


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    fieldnames = list(row)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames
            if existing_fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            if len(existing_fieldnames) != len(set(existing_fieldnames)):
                raise ValueError(f"CSV contains duplicate columns: {path}")
            # Metric aggregation may reorder keys or expose optional
            # diagnostics only in selected epochs (including after resume).
            # Preserve the established order, append genuinely new columns,
            # and let DictWriter leave unavailable values empty.
            fieldnames = existing_fieldnames + [
                key for key in row if key not in existing_fieldnames
            ]
            existing = list(reader)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing)
    writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
