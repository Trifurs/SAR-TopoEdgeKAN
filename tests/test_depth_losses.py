from __future__ import annotations

import pytest
import torch

from losses.composite_loss import CompositeFloodDepthLoss
from losses.depth_losses import (
    depth_bin_macro_root_mean_square,
    event_depth_hierarchical_macro_root_mean_square,
    event_macro_root_mean_square,
    event_mean_bias_loss,
)
from utils.config import load_config


def test_event_mean_bias_loss_gives_events_equal_weight() -> None:
    prediction = torch.tensor([[[[3.0, 3.0]]], [[[1.0, 1.0]]]])
    target = torch.tensor([[[[1.0, 1.0]]], [[[2.0, 2.0]]]])
    mask = torch.ones_like(prediction, dtype=torch.bool)

    loss = event_mean_bias_loss(
        prediction,
        target,
        mask,
        ["large-positive-bias", "negative-bias"],
        beta=1.0,
    )

    # smooth-L1(2)=1.5 and smooth-L1(-1)=0.5; both events receive one vote.
    assert float(loss) == pytest.approx(1.0)


def test_event_mean_bias_loss_groups_repeated_event_rasters() -> None:
    prediction = torch.tensor([[[[2.0]]], [[[0.0]]], [[[4.0]]]])
    target = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)

    grouped = event_mean_bias_loss(
        prediction,
        target,
        mask,
        ["same", "same", "other"],
        beta=1.0,
    )

    # The first event's +1 and -1 biases cancel; the other event has bias +3.
    assert float(grouped) == pytest.approx(1.25)


def test_event_mean_bias_loss_rejects_nonpositive_beta() -> None:
    value = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="beta"):
        event_mean_bias_loss(value, value, torch.ones_like(value), beta=0.0)


def test_event_macro_rmse_gives_each_event_one_vote() -> None:
    error = torch.tensor([[[[3.0, 4.0]]], [[[1.0, 1.0]]]])
    mask = torch.ones_like(error, dtype=torch.bool)

    loss = event_macro_root_mean_square(error, mask, ["large", "small"])

    # sqrt(mean(3^2, 4^2))=sqrt(12.5), then average with the unit-RMSE event.
    expected = (torch.sqrt(torch.tensor(12.5)) + 1.0) / 2.0
    assert torch.isclose(loss, expected, atol=1.0e-6)


def test_depth_bin_macro_rmse_gives_rare_tail_one_vote() -> None:
    error = torch.tensor([[[[1.0, 1.0, 4.0]]]])
    target = torch.tensor([[[[0.1, 0.2, 2.0]]]])
    mask = torch.ones_like(error, dtype=torch.bool)

    loss = depth_bin_macro_root_mean_square(
        error,
        target,
        mask,
        [0.1, 0.5, 3.0],
    )

    # Shallow RMSE is one and the single deep cell RMSE is four.
    assert float(loss) == pytest.approx(2.5)


def test_hierarchical_rmse_is_symmetric_and_has_finite_gradients() -> None:
    error = torch.tensor(
        [[[[1.0, -2.0, 3.0]]], [[[1.0, 2.0, -3.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[[0.1, 0.4, 2.0]]], [[[0.1, 0.4, 2.0]]]])
    mask = torch.ones_like(error, dtype=torch.bool)

    loss = event_depth_hierarchical_macro_root_mean_square(
        error,
        target,
        mask,
        ["event-a", "event-b"],
        [0.1, 0.3, 0.5, 3.0],
        [0.1, 0.3, 0.5, 1.0, 3.0],
    )
    mirrored = event_depth_hierarchical_macro_root_mean_square(
        -error,
        target,
        mask,
        ["event-a", "event-b"],
        [0.1, 0.3, 0.5, 3.0],
        [0.1, 0.3, 0.5, 1.0, 3.0],
    )

    assert torch.isclose(loss, mirrored)
    loss.backward()
    assert error.grad is not None
    assert torch.isfinite(error.grad).all()


@pytest.mark.parametrize("dtype, mode, expected", [
    (torch.float32, "full", 4.9181904792785645),
    (torch.float32, "mixed", 4.713714122772217),
    (torch.float32, "empty", 2.0000000233721948e-7),
    (torch.float64, "full", 4.979876766016875),
    (torch.float64, "mixed", 4.57823530261963),
    (torch.float64, "empty", 2e-7),
])
def test_shared_objective_preserves_the_production_value_and_mask(dtype, mode, expected) -> None:
    config = load_config("configs/sar_topo_edge_kan.xml")["loss"]
    criterion = CompositeFloodDepthLoss(
        config, [.1, .23, .49, 1., 2., 5., 12.], [.1, .23, .49, 1., 12.]
    )
    generator = torch.Generator().manual_seed(341)
    label = .1 + torch.rand(3, 1, 5, 7, generator=generator, dtype=dtype) * 8
    prediction = (.1 + torch.rand(3, 1, 5, 7, generator=generator, dtype=dtype) * 9).requires_grad_()
    valid = torch.ones_like(label, dtype=torch.bool)
    output_valid = torch.ones_like(valid)
    if mode == "mixed":
        valid[..., 0, :] = False
        output_valid[..., 2, ::2] = False
    elif mode == "empty":
        valid.zero_()
    batch = {
        "label": label,
        "masks": {"valid_depth_mask": valid},
        "validity": {"output_valid": output_valid},
        "metadata": {"source_event_id": ["event-a", "event-a", "event-b"]},
    }
    # A depth tensor, supervision masks and event identities suffice; the loss
    # has no dependency on model-specific features or observation metadata.
    loss, components = criterion({"depth": prediction}, batch)
    assert torch.equal(loss, loss.new_tensor(expected))
    assert set(components) == {
        "depth", "depth_linear", "depth_log", "depth_bias", "total",
        "metric_pixel_mae", "metric_pixel_rmse", "metric_event_mae",
        "metric_event_rmse", "metric_event_hierarchical_mae",
        "metric_depth_bin_rmse", "metric_event_hierarchical_rmse",
        "valid_depth_mask_pixels", "output_valid_pixels",
        "positive_supervision_pixels", "positive_excluded_by_output_valid_pixels",
        "positive_excluded_by_output_valid_fraction",
    }
    positive = valid & output_valid
    assert components["positive_supervision_pixels"] == positive.sum()
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[~positive]) == 0
    if mode == "empty":
        assert torch.count_nonzero(prediction.grad) == 0
    else:
        assert torch.count_nonzero(prediction.grad[positive]) > 0


def test_shared_objective_rejects_an_unsupported_objective() -> None:
    config = load_config("configs/sar_topo_edge_kan.xml")["loss"]
    with pytest.raises(ValueError, match="task-adaptive"):
        CompositeFloodDepthLoss(dict(config, objective_mode="unsupported"))
