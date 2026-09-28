from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.config import load_config
from utils.experiment_inventory import resolved_config_sha256
from validate_random_seeds import (
    EVALUATION_COMPLETION_FILES,
    PROTOCOL_NAME,
    PROTOCOL_SCHEMA_VERSION,
    TRAINING_COMPLETION_FILES,
    RandomSeedProtocolError,
    _completed_job_result,
    _isolated_worker_environment,
    _resume_checkpoint_or_archive,
    _summary_rows,
    _validate_protocol,
    latest_resumable_session,
    seed_schedule,
    write_seed_config,
)


def test_seed_schedule_is_deterministic_unique_and_bounded() -> None:
    assert seed_schedule(20_260_908, 1009, 5) == (
        20_260_908,
        20_261_917,
        20_262_926,
        20_263_935,
        20_264_944,
    )
    with pytest.raises(ValueError, match="at least 1"):
        seed_schedule(1, 1, 0)
    with pytest.raises(ValueError, match="stride must be positive"):
        seed_schedule(1, 0, 2)
    with pytest.raises(ValueError, match="NumPy compatibility"):
        seed_schedule(2**32 - 1, 1, 2)


def test_seed_overlay_preserves_dataset_and_records_fixed_split(tmp_path: Path) -> None:
    source = Path("configs/sar_topo_edge_kan.xml").resolve()
    source_config = load_config(source)
    destination = tmp_path / "seed.xml"
    training_output = tmp_path / "models" / "sar_topo_edge_kan" / "train"
    write_seed_config(
        source,
        destination,
        training_output=training_output,
        session_root=tmp_path / "session",
        seed=12345,
        seed_index=2,
        k=5,
        runtime_source_sha256="unit-test-runtime",
        dataset_identity={
            "contract_sha256": "unit-test-dataset",
            "manifest_sha256": "unit-test-dataset",
            "train_stats_sha256": "unit-test-dataset",
        },
    )
    config = load_config(destination)
    assert config["seed"] == 12345
    assert config["dataset"] == source_config["dataset"]
    assert config["runtime"]["train"]["output"] == training_output.resolve()
    assert config["runtime"]["train"]["resume"] is None
    assert config["runtime"]["evaluation"]["split"] == "test"
    assert config["loss"]["frozen_depth_weights_artifact"] == (
        training_output / "frozen_depth_weights.json"
    ).resolve()
    assert config["random_seed_validation"] == {
        "protocol": PROTOCOL_NAME,
        "k": 5,
        "seed_index": 2,
        "seed": 12345,
        "fixed_split": True,
        "source_config": source,
        "source_config_sha256": resolved_config_sha256(source),
        "runtime_source_sha256": "unit-test-runtime",
        "dataset_contract_sha256": "unit-test-dataset",
        "dataset_manifest_sha256": "unit-test-dataset",
        "train_stats_sha256": "unit-test-dataset",
    }


def test_seed_overlay_can_resume_family_specific_checkpoint(tmp_path: Path) -> None:
    source = Path("configs/compare/deep_learning/dlsim_linknet.xml").resolve()
    checkpoint = tmp_path / "model" / "train" / "last_raw.pth"
    destination = tmp_path / "seed.xml"
    write_seed_config(
        source,
        destination,
        training_output=checkpoint.parent,
        session_root=tmp_path / "session",
        seed=7,
        seed_index=1,
        k=3,
        runtime_source_sha256="unit-test-runtime",
        dataset_identity={
            "contract_sha256": "unit-test-dataset",
            "manifest_sha256": "unit-test-dataset",
            "train_stats_sha256": "unit-test-dataset",
        },
        resume_checkpoint=checkpoint,
    )
    config = load_config(destination)
    assert config["runtime"]["train"]["resume"] == checkpoint.resolve()


def test_completed_seed_job_requires_training_and_test_products(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    for name in TRAINING_COMPLETION_FILES:
        path = model_root / "train" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for name in EVALUATION_COMPLETION_FILES:
        path = model_root / "test" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pixel_micro_mae": 0.2} if name == "summary.json" else {}
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = _completed_job_result(
        model_root,
        model_id="example",
        display_name="Example",
        seed=11,
        seed_index=2,
        source_config=Path("example.xml"),
    )
    assert result is not None
    assert result["seed"] == 11
    assert result["seed_index"] == 2
    assert result["split"] == "test"
    (model_root / "test" / "metrics_by_event.csv").unlink()
    assert (
        _completed_job_result(
            model_root,
            model_id="example",
            display_name="Example",
            seed=11,
            seed_index=2,
            source_config=Path("example.xml"),
        )
        is None
    )


@pytest.mark.parametrize(
    ("model_kind", "checkpoint_name"),
    [("sar_topo_edge_kan", "last.pth"), ("learned_comparator", "last_raw.pth")],
)
def test_partial_seed_training_resumes_from_family_checkpoint(
    tmp_path: Path, model_kind: str, checkpoint_name: str
) -> None:
    model_root = tmp_path / model_kind
    checkpoint = model_root / "train" / checkpoint_name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert (
        _resume_checkpoint_or_archive(model_root, model_kind=model_kind) == checkpoint
    )


def test_latest_seed_resume_ignores_complete_sessions(tmp_path: Path) -> None:
    root = tmp_path / "random_seed_validation" / "k5"
    failed = root / "20260915-120000-000001"
    complete = root / "20260915-130000-000001"
    for path, status in ((failed, "failed"), (complete, "complete")):
        path.mkdir(parents=True)
        (path / "protocol.json").write_text("{}", encoding="utf-8")
        (path / "status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
    assert latest_resumable_session(tmp_path, 5) == failed


def test_protocol_rejects_runtime_or_seed_schedule_changes() -> None:
    inventory = [{"model": "example", "resolved_config_sha256": "abc"}]
    dataset = {"manifest_sha256": "dataset"}
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "k": 2,
        "seeds": [3, 5],
        "models": inventory,
        "runtime_source_sha256": "runtime",
        "dataset": dataset,
    }
    _validate_protocol(
        protocol,
        k=2,
        seeds=(3, 5),
        inventory=inventory,
        runtime_sha256="runtime",
        dataset_identity=dataset,
    )
    with pytest.raises(RandomSeedProtocolError, match="schedule differs"):
        _validate_protocol(
            protocol,
            k=2,
            seeds=(3, 7),
            inventory=inventory,
            runtime_sha256="runtime",
            dataset_identity=dataset,
        )
    with pytest.raises(RandomSeedProtocolError, match="Runtime Python"):
        _validate_protocol(
            protocol,
            k=2,
            seeds=(3, 5),
            inventory=inventory,
            runtime_sha256="changed",
            dataset_identity=dataset,
        )


def test_summary_excludes_seed_identifiers_from_metric_rows() -> None:
    rows = [
        {
            "model": "example",
            "display_name": "Example",
            "split": "test",
            "seed": 3,
            "seed_index": 1,
            "pixel_micro_mae": 0.2,
        },
        {
            "model": "example",
            "display_name": "Example",
            "split": "test",
            "seed": 5,
            "seed_index": 2,
            "pixel_micro_mae": 0.4,
        },
    ]
    summary = _summary_rows(rows)
    assert {row["metric"] for row in summary} == {"pixel_micro_mae"}
    assert summary[0]["mean"] == pytest.approx(0.3)
    assert summary[0]["std"] == pytest.approx(2**0.5 / 10)


def test_seed_worker_forces_gnu_mkl_layer(monkeypatch) -> None:
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")
    environment = _isolated_worker_environment("gnu")
    assert environment["MKL_THREADING_LAYER"] == "GNU"
    with pytest.raises(ValueError, match="must be GNU"):
        _isolated_worker_environment("INTEL")
