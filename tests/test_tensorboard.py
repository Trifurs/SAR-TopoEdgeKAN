from __future__ import annotations

import logging
from pathlib import Path

import torch

from tools.train_sar_topo_edge_kan import _training_diagnostic_values
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


def test_training_logs_only_measured_module_diagnostics() -> None:
    mask = torch.ones(1, 1, 2, 2)
    batch = {"masks": {"valid_depth_mask": mask}, "validity": {"output_valid": mask}}
    assert _training_diagnostic_values({}, batch) == {"canonical_positive_pixel_count": 4.0}
    values = _training_diagnostic_values({
        "sar_diagnostics": {"quality_mean": torch.tensor(0.5)},
        "graph_diagnostics": {
            "final_graph_gate_mean": torch.tensor([0.25, 0.75], requires_grad=True),
            "graph_gamma_mean": torch.tensor(float("nan")),
        },
    }, batch)
    assert values == {
        "canonical_positive_pixel_count": 4.0,
        "quality_mean": 0.5,
        "final_graph_gate_mean": 0.5,
    }
