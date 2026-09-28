from __future__ import annotations

import torch

from tools.evaluate_sar_topo_edge_kan import dataset_fingerprint
from utils.checkpoint import initialize_from_checkpoint, load_checkpoint, save_checkpoint
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


def checkpoint_identity(config, fingerprint):
    from utils.checkpoint import training_identity_sha256

    return training_identity_sha256(
        jsonable_config(config), fingerprint, training_context={"epochs": 1}
    )


def test_compatible_initialization_marks_shape_changes_as_new(
    production_config, tmp_path
) -> None:
    source_model = build_model(production_config)
    fingerprint = dataset_fingerprint(production_config)
    path = tmp_path / "source.pth"
    save_checkpoint(
        path,
        source_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        best_metric=1.0,
        resolved_config=jsonable_config(production_config),
        dataset_fingerprint=fingerprint,
    )
    tensor_name = "heads.depth_head.trunk.3.bias"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model"][tensor_name] = torch.zeros(2)
    torch.save(payload, path)
    target_model = build_model(production_config)

    _, report = initialize_from_checkpoint(
        path,
        target_model,
        transfer="compatible",
        expected_fingerprint=fingerprint,
    )

    assert tensor_name in report["new_tensors"]
    assert tensor_name in report["unused_source_tensors"]
