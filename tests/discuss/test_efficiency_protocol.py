from __future__ import annotations

import json
import os

import torch
from torch import nn

from tools.discuss.profile_efficiency import checkpoint_index, measure


def test_efficiency_selects_a_common_seed_before_selecting_the_latest_run(tmp_path):
    runs = []
    for name, seed, timestamp in (("old", 11, 1000), ("new", 11, 2000), ("other_seed", 22, 3000)):
        run = tmp_path / name
        run.mkdir()
        (run / "best_raw.pth").touch()
        (run / "resolved_config.json").write_text(json.dumps({
            "model": {"name": "sar_topo_edge_kan"}, "run_name": "discuss_full", "seed": seed,
        }), encoding="utf-8")
        summary = run / "training_summary.json"
        summary.write_text("{}", encoding="utf-8")
        os.utime(summary, ns=(timestamp, timestamp))
        runs.append(run)
    assert checkpoint_index(tmp_path, seed=11) == {"discuss_full": runs[1]}
    assert checkpoint_index(tmp_path, seed=22) == {"discuss_full": runs[2]}
    assert checkpoint_index(tmp_path, seed=33) == {}


def test_efficiency_excludes_unused_modules_and_reports_repeated_timings():
    class Example(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Conv2d(1, 1, 1)
            self.unused = nn.Linear(2, 1)

        def forward(self, inputs):
            return {"depth": self.network(inputs["valid_depth_mask"])}

    model = Example()
    result = measure(
        model, {"valid_depth_mask": torch.ones(2, 1, 8, 8)}, torch.device("cpu"),
        precision="float32", warmup=1, iterations=2, repetitions=2, stages=True, flops=False,
    )
    assert result["total_parameters"] == 5
    assert result["executed_module_parameters"] == 2
    assert result["latency_samples"] == 4
    assert len(result["raw_batch_latency_ms"]) == 4
    assert result["mean_batch_latency_ms"] > 0
    assert result["samples_per_second_mean"] > 0
    assert result["supported_flops_per_sample"] is None
    assert "network" in result["stage_latency_ms"]
    assert not model.network._forward_hooks
    assert not model.network._forward_pre_hooks
