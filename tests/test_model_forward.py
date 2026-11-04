from __future__ import annotations

import torch
from torch.utils.data import default_collate

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from utils.registry import build_model


def test_model_forward_on_audited_sample(production_config) -> None:
    contract = DatasetContract.load(production_config["dataset"]["contract"])
    spec = ModelInputSpec.from_config(production_config)
    dataset = FloodDepthDataset(
        production_config["dataset"]["contract"],
        production_config["dataset"]["train_stats"],
        "train",
        band_spec=resolve_band_spec(production_config, contract),
        input_spec=spec,
        minimum_event_band_fraction=float(
            production_config["dataset"]["minimum_event_band_fraction"]
        ),
        s1_qa_names=production_config["dataset"].get("model_s1_qa_names"),
    )
    batch = default_collate([dataset[0]])
    model = build_model(production_config).eval()
    assert "valid_depth_mask" not in prepare_model_inputs(batch, spec)
    model_inputs = prepare_model_inputs(
        batch,
        spec,
        include_valid_depth_mask=model.known_flood_range_enabled,
    )
    assert torch.equal(
        model_inputs["valid_depth_mask"], batch["masks"]["valid_depth_mask"]
    )
    with torch.no_grad():
        outputs = model(model_inputs)
    assert outputs["depth"].shape == batch["label"].shape
    assert torch.isfinite(outputs["depth"]).all()
    support = batch["masks"]["valid_depth_mask"] > 0.5
    assert torch.all(outputs["depth"][support] > 0)
    assert torch.all(outputs["depth"][~support] == 0)
