from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compare.common._depth_regression import comparison_input_channels
from compare.common.comparison_factory import build_comparison_model
from utils.config import load_config


CONFIGS = (
    "dlsim_attention_unet.xml",
    "dlsim_linknet.xml",
    "unet_depth_regression.xml",
    "resnet18_depth_regression.xml",
    "unetplusplus_depth_regression.xml",
)


@pytest.mark.parametrize("filename", CONFIGS)
def test_learned_comparison_model_is_range_conditioned(filename: str) -> None:
    config = load_config(Path("configs/compare/deep_learning") / filename)
    model = build_comparison_model(config).eval()
    channels = comparison_input_channels(config["model"]["input_schema"])
    inputs = torch.randn(1, channels, 64, 64)
    flood_range = torch.zeros(1, 1, 64, 64)
    flood_range[..., 8:-8, 8:-8] = 1.0
    with torch.no_grad():
        outputs = model(inputs, flood_range)
    assert outputs["depth"].shape == flood_range.shape
    assert torch.isfinite(outputs["depth"]).all()
    assert torch.isfinite(outputs["conditional_depth"]).all()
    assert torch.all(outputs["conditional_depth"] > 0)
    assert torch.count_nonzero(outputs["depth"] * (1.0 - flood_range)) == 0


def test_learned_comparison_source_record_covers_every_model() -> None:
    text = Path("docs/COMPARISON_SOURCES.md").read_text(encoding="utf-8")
    for filename in CONFIGS:
        identifier = load_config(
            Path("configs/compare/deep_learning") / filename
        )["model"]["name"]
        assert identifier in text
