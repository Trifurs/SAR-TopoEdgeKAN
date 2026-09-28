from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import evaluate
import pytest
import train

from utils.config import load_config
from utils.model_dispatch import configured_model_kind
from utils.run_paths import (
    evaluation_checkpoint_path,
    evaluation_output_path,
    started_at_run_id,
    train_output_path,
)


def _configured(path: str, tmp_path: Path) -> dict:
    config = load_config(path)
    config["runtime"]["train"]["output"] = tmp_path / "train_output"
    config["runtime"]["evaluation"]["output"] = tmp_path / "evaluation_output"
    return config


def test_common_runtime_resolves_model_and_ablation_paths() -> None:
    main = load_config("configs/sar_topo_edge_kan.xml")
    assert configured_model_kind(main) == ("sar_topo_edge_kan", "sar_topo_edge_kan")
    run_id = started_at_run_id(main, datetime(2026, 9, 8, 15, 30, 45, 123456))
    assert run_id == "20260908-153045-123456"
    assert train_output_path(main, run_id).as_posix().endswith(
        "train/sar_topo_edge_kan/20260908-153045-123456"
    )
    assert evaluation_output_path(main, run_id).as_posix().endswith(
        "evaluate/sar_topo_edge_kan/val/20260908-153045-123456"
    )
    main["runtime"]["evaluation"]["source_run"] = run_id
    assert evaluation_checkpoint_path(main).as_posix().endswith(
        "train/sar_topo_edge_kan/20260908-153045-123456/best_raw.pth"
    )

    ablation = load_config("configs/ablation/sar_topo_edge_kan_wo_tae_kan.xml")
    assert train_output_path(ablation, run_id).as_posix().endswith(
        "ablation/wo_tae_kan/20260908-153045-123456"
    )


def test_evaluation_defaults_to_the_newest_completed_timestamped_run(
    tmp_path: Path,
) -> None:
    config = load_config("configs/sar_topo_edge_kan.xml")
    config["runs_root"] = tmp_path
    older = train_output_path(config, "20260908-090000-000000")
    newer = train_output_path(config, "20260908-100000-000000")
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    old_checkpoint = older / "best_raw.pth"
    new_checkpoint = newer / "best_raw.pth"
    old_summary = older / "training_summary.json"
    new_summary = newer / "training_summary.json"
    old_checkpoint.touch()
    new_checkpoint.touch()
    old_summary.touch()
    new_summary.touch()
    os.utime(old_summary, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new_summary, ns=(2_000_000_000, 2_000_000_000))

    assert evaluation_checkpoint_path(config) == new_checkpoint


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("configs/sar_topo_edge_kan.xml", ("sar_topo_edge_kan", "sar_topo_edge_kan")),
        (
            "configs/compare/deep_learning/dlsim_attention_unet.xml",
            ("learned_comparator", "dlsim_attention_unet"),
        ),
        (
            "configs/compare/deep_learning/dlsim_linknet.xml",
            ("learned_comparator", "dlsim_linknet"),
        ),
        (
            "configs/compare/deep_learning/unet_depth_regression.xml",
            ("learned_comparator", "unet_depth_regression"),
        ),
        (
            "configs/compare/deep_learning/resnet18_depth_regression.xml",
            ("learned_comparator", "resnet18_depth_regression"),
        ),
        (
            "configs/compare/deep_learning/unetplusplus_depth_regression.xml",
            ("learned_comparator", "unetplusplus_depth_regression"),
        ),
        ("configs/compare/traditional/fwdet_v2.xml", ("traditional", "fwdet_v2")),
        ("configs/compare/traditional/tsa.xml", ("traditional", "tsa")),
        ("configs/compare/traditional/fldepth.xml", ("traditional", "fldepth")),
    ],
)
def test_every_model_configuration_has_one_unified_dispatch_kind(
    path: str, expected: tuple[str, str]
) -> None:
    config = load_config(path)
    assert configured_model_kind(config) == expected
    assert config["runtime"]["run_id_format"] == "%Y%m%d-%H%M%S-%f"
    assert config["training"]["epochs"] > 0
    assert config["logging"]["progress_bar"] is True
    assert config["optimizer"]["beta1"] == 0.9
    assert config["optimizer"]["beta2"] == 0.999


def test_unified_train_dispatches_main_without_cli_overrides(monkeypatch, tmp_path: Path) -> None:
    config = _configured("configs/sar_topo_edge_kan.xml", tmp_path)
    monkeypatch.setattr(train, "load_config", lambda _: config)
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return args.output

    monkeypatch.setattr(train, "run_sar_topo_edge_kan_training", fake_run)
    result = train.run_from_config(Path("configs/sar_topo_edge_kan.xml"))
    assert result == tmp_path / "train_output"
    assert captured["args"].device is None
    assert captured["args"].epochs is None
    assert captured["args"].batch_size is None
    assert captured["args"].output == tmp_path / "train_output"


def test_unified_train_rejects_traditional_models(monkeypatch, tmp_path: Path) -> None:
    config = _configured("configs/compare/traditional/fwdet_v2.xml", tmp_path)
    monkeypatch.setattr(train, "load_config", lambda _: config)
    with pytest.raises(ValueError, match="test.py"):
        train.run_from_config(Path("configs/compare/traditional/fwdet_v2.xml"))


def test_unified_train_dispatches_learned_comparator_without_cli_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    config = _configured(
        "configs/compare/deep_learning/dlsim_attention_unet.xml", tmp_path
    )
    monkeypatch.setattr(train, "load_config", lambda _: config)
    captured = {}

    def fake_run(args, identifier):
        captured["args"] = args
        captured["identifier"] = identifier
        return args.output

    monkeypatch.setattr(train, "run_learned_training", fake_run)
    result = train.run_from_config(
        Path("configs/compare/deep_learning/dlsim_attention_unet.xml")
    )
    assert result == tmp_path / "train_output"
    assert captured["identifier"] == "dlsim_attention_unet"
    assert captured["args"].device is None
    assert captured["args"].epochs is None
    assert captured["args"].batch_size is None
    assert captured["args"].num_workers is None
    assert captured["args"].output == tmp_path / "train_output"


def test_unified_train_passes_learned_comparator_resume_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "train_output" / "last_raw.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    config = _configured("configs/compare/deep_learning/dlsim_linknet.xml", tmp_path)
    config["runtime"]["train"]["resume"] = checkpoint
    monkeypatch.setattr(train, "load_config", lambda _: config)
    captured = {}

    def fake_run(args, identifier):
        captured["args"] = args
        captured["identifier"] = identifier
        return args.output

    monkeypatch.setattr(train, "run_learned_training", fake_run)
    result = train.run_from_config(
        Path("configs/compare/deep_learning/dlsim_linknet.xml")
    )
    assert result == checkpoint.parent
    assert captured["identifier"] == "dlsim_linknet"
    assert captured["args"].resume == checkpoint


def test_unified_evaluate_resolves_configured_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "trained" / "best_raw.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    config = _configured("configs/sar_topo_edge_kan.xml", tmp_path)
    config["runtime"]["evaluation"]["checkpoint"] = checkpoint
    monkeypatch.setattr(evaluate, "load_config", lambda _: config)
    captured = {}

    def fake_run(*args):
        captured["args"] = args
        return {"pixel_micro_mae": 0.0}

    monkeypatch.setattr(evaluate, "run_sar_topo_edge_kan_evaluation", fake_run)
    result = evaluate.run_from_config(Path("configs/sar_topo_edge_kan.xml"))
    assert result == {"pixel_micro_mae": 0.0}
    assert captured["args"][1] == checkpoint
    assert captured["args"][2] == "val"
    assert captured["args"][4] == tmp_path / "evaluation_output"
