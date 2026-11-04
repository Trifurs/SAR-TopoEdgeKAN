from __future__ import annotations

import torch
import pytest

from tools.evaluate_sar_topo_edge_kan import dataset_fingerprint
from utils.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from utils.config import jsonable_config
from utils.registry import build_model


def test_checkpoint_round_trip_is_fingerprint_strict(production_config, tmp_path) -> None:
    model = build_model(production_config)
    fingerprint = dataset_fingerprint(production_config)
    path = tmp_path / "model.pth"
    save_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        best_metric=1.0,
        resolved_config=jsonable_config(production_config),
        dataset_fingerprint=fingerprint,
        training_context={"epochs": 1},
    )
    restored = build_model(production_config)
    checkpoint = load_checkpoint(
        path,
        restored,
        expected_fingerprint=fingerprint,
        expected_training_identity_sha256=checkpoint_identity(
            production_config, fingerprint
        ),
    )
    assert checkpoint["output_semantics"] == "conditional_positive"
    source = next(model.parameters()).detach()
    target = next(restored.parameters()).detach()
    assert torch.equal(source, target)
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_checkpoint(path, restored, expected_fingerprint={"contract_sha256": "changed"})
    with pytest.raises(CheckpointError, match="Training identity"):
        load_checkpoint(path, restored, expected_training_identity_sha256="changed")


def checkpoint_identity(config, fingerprint):
    from utils.checkpoint import training_identity_sha256

    return training_identity_sha256(
        jsonable_config(config), fingerprint, training_context={"epochs": 1}
    )
