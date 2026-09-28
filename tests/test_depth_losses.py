from __future__ import annotations

import pytest
import torch

from losses.depth_losses import (
    depth_bin_macro_root_mean_square,
    event_depth_hierarchical_macro_root_mean_square,
    event_macro_root_mean_square,
    event_mean_bias_loss,
)


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
