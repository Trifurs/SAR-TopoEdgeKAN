"""Small, dependency-tolerant TensorBoard helpers shared by all runners."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping


def create_summary_writer(
    log_dir: Path,
    *,
    enabled: bool,
    flush_seconds: int,
    logger: logging.Logger,
) -> Any | None:
    """Create a writer only when enabled, with a clear non-fatal fallback."""

    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - depends on optional installation
        logger.warning("TensorBoard is unavailable; continuing without event logs: %s", exc)
        return None
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=max(1, int(flush_seconds)))
    logger.info("TensorBoard event stream initialized: %s", log_dir)
    return writer


def add_metadata(writer: Any | None, metadata: Mapping[str, Any]) -> None:
    """Store concise immutable run metadata in the TensorBoard Text tab."""

    if writer is None:
        return
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    writer.add_text("run/metadata", "  \n".join(lines), global_step=0)


def add_scalars(
    writer: Any | None,
    values: Mapping[str, Any],
    *,
    step: int,
    prefix: str = "",
) -> None:
    """Write finite numeric values while avoiding invalid TensorBoard records."""

    if writer is None:
        return
    normalized_prefix = prefix.strip("/")
    for name, value in values.items():
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        tag = f"{normalized_prefix}/{name}" if normalized_prefix else str(name)
        writer.add_scalar(tag, float(value), global_step=step)


def flush(writer: Any | None) -> None:
    """Make live dashboards update at epoch boundaries."""

    if writer is not None:
        writer.flush()
