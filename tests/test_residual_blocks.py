"""Shape and gradient contracts for configurable SAR-TopoEdgeKAN residual blocks."""

from __future__ import annotations

import pytest
import torch

from models.efficient_blocks import (
    EfficientResidualBlock,
    InvertedResidualBlock,
    SpatialResidualBlock,
    residual_path_scale,
    residual_block,
)
from models.s1_hydrology_backbone import (
    HydrologyContext,
    JointSARHydrologyEncoder,
    S1FeatureAlignment,
)


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    (
        ("efficient", EfficientResidualBlock),
        ("inverted", InvertedResidualBlock),
        ("spatial", SpatialResidualBlock),
    ),
)
def test_residual_block_preserves_shape_and_gradient(kind, expected_type):
    module = residual_block(kind, channels=16, dropout=0.0, groups=8)
    value = torch.randn(2, 16, 17, 19, requires_grad=True)
    output = module(value)

    assert isinstance(module, expected_type)
    assert output.shape == value.shape
    output.mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_residual_block_rejects_unknown_kind():
    with pytest.raises(ValueError, match="efficient.*inverted.*spatial"):
        residual_block("unknown", channels=16, dropout=0.0, groups=8)


def test_residual_path_scale_is_shared_per_sample_and_expectation_preserving():
    reference = torch.ones(64, 8, 4, 4)
    torch.manual_seed(17)
    scale = residual_path_scale(reference, 0.5, training=True)

    assert scale.shape == (64, 1, 1, 1)
    assert set(scale.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(
        residual_path_scale(reference, 0.5, training=False),
        torch.ones(64, 1, 1, 1),
    )


def test_joint_sar_encoder_preserves_multiscale_contract():
    encoder = JointSARHydrologyEncoder(
        state_channels=2,
        change_channels=3,
        channels=(8, 16, 24, 32),
        dropout=0.0,
        groups=8,
        block_kind="spatial",
        known_flood_range_enabled=True,
    )
    shape = (2, 1, 32, 32)
    valid = torch.ones(shape)
    outputs, diagnostics = encoder(
        torch.randn(2, 2, 32, 32),
        torch.randn(2, 2, 32, 32),
        torch.randn(2, 3, 32, 32),
        valid,
        torch.ones(shape),
        {
            "s1_t1": valid,
            "s1_t2": valid,
            "s1_change": valid,
        },
    )

    assert [tuple(value.shape) for value in outputs] == [
        (2, 8, 32, 32),
        (2, 16, 16, 16),
        (2, 24, 8, 8),
        (2, 32, 4, 4),
    ]
    assert len(diagnostics["quality_gates"]) == 4
    assert len(diagnostics["change_evidence"]) == 4
    sum(value.mean() for value in outputs).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


def test_joint_sar_encoder_event_validity_gates_every_output_scale():
    encoder = JointSARHydrologyEncoder(
        state_channels=2,
        change_channels=3,
        channels=(8, 16, 24, 32),
        dropout=0.0,
        groups=8,
        block_kind="inverted",
        known_flood_range_enabled=False,
    ).eval()
    generator = torch.Generator().manual_seed(31)
    shape = (2, 1, 32, 32)
    valid = torch.ones(shape)
    pre = torch.randn(2, 2, 32, 32, generator=generator)
    event = torch.randn(2, 2, 32, 32, generator=generator)
    change = torch.randn(2, 3, 32, 32, generator=generator)
    event_missing = torch.zeros_like(valid)
    outputs, diagnostics = encoder(
        pre,
        event,
        change,
        valid,
        None,
        {"s1_t1": valid, "s1_t2": event_missing, "s1_change": valid},
    )

    assert all(torch.count_nonzero(value) == 0 for value in outputs)
    assert diagnostics["quality_mean"].item() == 0.0
    assert not any("reliability" in name for name, _ in encoder.named_parameters())


def test_hydrology_context_signed_residual_can_learn_below_identity():
    context = HydrologyContext(
        channels=8,
        groups=4,
        dropout=0.0,
        residual_init=0.0,
        residual_max=0.5,
        residual_parameterization="signed",
    ).eval()
    with torch.no_grad():
        context.gamma.fill_(-0.1)
    value = torch.randn(2, 8, 9, 11)
    output = context(value)

    assert context.residual.item() == pytest.approx(-0.1)
    assert not torch.equal(output, value)


def test_s1_feature_alignment_projects_each_scale_and_preserves_gradients():
    alignment = S1FeatureAlignment(channels=(8,), groups=4).eval()
    sar = [torch.randn(2, 8, 8, 8, requires_grad=True)]
    one = torch.ones(2, 1, 8, 8)
    outputs, diagnostics = alignment(sar, one)

    assert outputs[0].shape == sar[0].shape
    assert diagnostics["sar_gates"][0].shape == (2, 1, 8, 8)
    assert diagnostics["sar_gate_mean"].item() == pytest.approx(1.0)
    assert not hasattr(alignment, "terrain_gate")
    assert not hasattr(alignment, "refine")
    outputs[0].square().mean().backward()
    assert sar[0].grad is not None
    assert torch.isfinite(sar[0].grad).all()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in alignment.projection.parameters()
    )
