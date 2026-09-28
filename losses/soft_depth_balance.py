"""Train-only soft depth balancing helper."""

from __future__ import annotations

from collections.abc import Sequence

import torch

def soft_depth_balance_weights(
    target: torch.Tensor,
    positive: torch.Tensor,
    train_bins: Sequence[float],
    minimum: float = 0.5,
    maximum: float = 3.0,
    alpha: float = 0.5,
    tau: float = 10.0,
    train_bin_counts: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return a static train-derived curve without minibatch normalization.

    Calls with positive pixels are rejected rather than deriving a target weight
    from the current minibatch. Production training supplies a
    ``FrozenSoftDepthBalance`` directly to the task-adaptive objective.
    """

    if alpha < 0.0 or tau < 0.0:
        raise ValueError("soft_depth_balance alpha and tau must be nonnegative")
    selected = positive > 0.5
    if not bool(torch.any(selected)):
        return torch.ones_like(target)
    raise ValueError(
        "soft-depth balance requires a FrozenSoftDepthBalance derived from the "
        "complete training split"
    )
