from __future__ import annotations

import pytest
import torch

from utils.spatial_tta import resolve_spatial_tta, spatial_tta_forward


def test_spatial_tta_keeps_identity_first_and_removes_duplicates() -> None:
    assert resolve_spatial_tta(("horizontal", "identity", "horizontal")) == (
        "identity",
        "horizontal",
    )


def test_spatial_tta_flips_nested_inputs_and_restores_dense_outputs() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    outputs = spatial_tta_forward(
        lambda inputs: {
            "depth": inputs["nested"]["image"],
            "conditional_depth": inputs["nested"]["image"],
        },
        {"nested": {"image": value}, "metadata": ("unchanged",)},
        ("horizontal", "vertical"),
    )
    assert torch.equal(outputs["depth"], value)
    assert outputs["spatial_tta_transforms"] == (
        "identity",
        "horizontal",
        "vertical",
    )


def test_spatial_tta_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        resolve_spatial_tta(("diagonal",))
