from __future__ import annotations

import torch

from utils.efficiency import InferenceEfficiency, model_size_metrics


def test_efficiency_summary_is_stable_and_can_exclude_a_warmup_batch() -> None:
    timer = InferenceEfficiency(torch.device("cpu"), warmup_batches=1)
    timer.stop(timer.start(), samples=2, output_pixels=200)
    timer.stop(timer.start(), samples=3, output_pixels=300)
    first = timer.summary()
    second = timer.summary()
    assert first == second
    assert first["efficiency_warmup_batches_excluded"] == 1
    assert first["efficiency_batches"] == 1
    assert first["efficiency_samples"] == 3
    assert first["efficiency_output_pixels"] == 300
    assert first["efficiency_device"] == "cpu"


def test_model_size_metrics_reports_parameters_and_buffers() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    metrics = model_size_metrics(model)
    assert metrics["total_parameters"] == 24
    assert metrics["trainable_parameters"] == 24
    assert metrics["parameter_storage_mib"] > 0.0
    assert metrics["buffer_storage_mib"] > 0.0
