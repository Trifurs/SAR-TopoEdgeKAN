from __future__ import annotations

import json
import os
from pathlib import Path

from utils.evaluation_suite import (
    completed_run_directories,
    select_runs_from_roots,
)


def _run(root: Path, model: str, name: str, mtime: int) -> Path:
    run = root / model / name
    run.mkdir(parents=True)
    (run / "best_raw.pth").touch()
    (run / "resolved_config.json").write_text(
        json.dumps({"model": {"name": model}, "run_name": model}),
        encoding="utf-8",
    )
    summary = run / "training_summary.json"
    summary.write_text("{}", encoding="utf-8")
    os.utime(summary, ns=(mtime, mtime))
    return run


def test_run_discovery_requires_complete_metadata_and_selects_latest_per_model(
    tmp_path: Path,
) -> None:
    old = _run(tmp_path, "sar_topo_edge_kan", "old", 1_000)
    new = _run(tmp_path, "sar_topo_edge_kan", "new", 2_000)
    comparator = _run(tmp_path, "unet_depth_regression", "one", 1_500)
    incomplete = tmp_path / "broken"
    incomplete.mkdir()
    (incomplete / "best_raw.pth").touch()

    assert completed_run_directories(tmp_path) == sorted([old, new, comparator])
    assert select_runs_from_roots((tmp_path,), latest_only=True) == [new, comparator]
    assert select_runs_from_roots((tmp_path,), latest_only=False) == sorted([old, new, comparator])


def test_multiple_explicit_roots_include_train_and_ablation_without_other_branches(
    tmp_path: Path,
) -> None:
    main = _run(tmp_path / "train", "sar_topo_edge_kan", "main", 1_000)
    ablation = _run(tmp_path / "ablation", "sar_topo_edge_kan", "abl", 2_000)
    config = json.loads((ablation / "resolved_config.json").read_text(encoding="utf-8"))
    config["ablation"] = {"variant_id": "wo_tae_kan"}
    (ablation / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    _run(tmp_path / "other_campaign", "sar_topo_edge_kan", "unrelated", 3_000)

    selected = select_runs_from_roots(
        (tmp_path / "train", tmp_path / "ablation"), latest_only=True
    )
    assert selected == [main, ablation]
