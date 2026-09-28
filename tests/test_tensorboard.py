from __future__ import annotations

import logging
from pathlib import Path

from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


def test_tensorboard_writer_persists_scalar_and_metadata_events(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    writer = create_summary_writer(
        log_dir,
        enabled=True,
        flush_seconds=1,
        logger=logging.getLogger("test.tensorboard"),
    )

    assert writer is not None
    add_metadata(writer, {"model": "test_model", "seed": 1})
    add_scalars(
        writer,
        {"loss": 0.125, "non_finite": float("inf"), "ignored_flag": True},
        step=1,
        prefix="train",
    )
    flush(writer)
    writer.close()

    assert list(log_dir.glob("events.out.tfevents.*"))
