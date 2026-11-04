"""Masked pixel, event and depth-stratified reductions used by the shared objective."""


from __future__ import annotations


from collections.abc import Sequence


import torch
import torch.nn.functional as F


def masked_micro_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average every valid element equally, independent of sample/event identity."""

    if values.shape != mask.shape:
        mask = torch.broadcast_to(mask, values.shape)
    selected = mask.to(dtype=torch.bool)
    if not torch.any(selected):
        return values.sum() * 0.0
    return values[selected].mean()



def event_macro_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Average pixels within each batch-present event, then average events."""

    if values.shape != mask.shape:
        mask = torch.broadcast_to(mask, values.shape)
    mask = mask.to(dtype=torch.bool)
    batch_size = values.shape[0]
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [index for index, item in enumerate(event_ids) if str(item) == event]
        event_values = values[indices]
        event_mask = mask[indices]
        if torch.any(event_mask):
            event_losses.append(event_values[event_mask].mean())
    if not event_losses:
        return values.sum() * 0.0
    return torch.stack(event_losses).mean()


def event_macro_root_mean_square(
    error: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Average event-level RMSE while retaining a stable zero-error gradient.

    Pixel-micro RMSE can still be dominated by events containing many tiles.
    Computing RMSE inside each batch-present event first gives rare events a
    direct tail-sensitive signal. The square root is clamped only for its
    derivative; exactly zero error is explicitly returned as zero.
    """

    if error.shape != mask.shape:
        mask = torch.broadcast_to(mask, error.shape)
    selected = mask.to(dtype=torch.bool)
    batch_size = error.shape[0]
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [
            index for index, item in enumerate(event_ids) if str(item) == event
        ]
        event_mask = selected[indices]
        if torch.any(event_mask):
            mean_square = error[indices][event_mask].square().mean()
            event_losses.append(
                torch.where(
                    mean_square > 0.0,
                    mean_square.clamp_min(1.0e-12).sqrt(),
                    mean_square,
                )
            )
    if not event_losses:
        return error.sum() * 0.0
    return torch.stack(event_losses).mean()



def event_mean_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None = None,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Penalize the signed mean-depth error of each batch-present event.

    Pixel-micro regression can almost ignore a rare event with a small flooded
    area even when its entire depth field is shifted by several metres.  This
    calibration term first averages the signed error within each event and then
    gives every event one vote.  Passing ``event_ids=None`` intentionally treats
    every raster in the batch as an independent group, which is useful when an
    event-interleaved sampler places at most one raster from an event in a batch.
    """

    if beta <= 0.0:
        raise ValueError("event mean-bias beta must be positive")
    if prediction.shape != target.shape:
        target = torch.broadcast_to(target, prediction.shape)
    if prediction.shape != mask.shape:
        mask = torch.broadcast_to(mask, prediction.shape)
    selected = mask.to(dtype=torch.bool)
    batch_size = prediction.shape[0]
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    signed_error = prediction - target
    losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [
            index for index, item in enumerate(event_ids) if str(item) == event
        ]
        event_mask = selected[indices]
        if torch.any(event_mask):
            mean_error = signed_error[indices][event_mask].mean()
            losses.append(
                F.smooth_l1_loss(
                    mean_error,
                    torch.zeros_like(mean_error),
                    beta=float(beta),
                )
            )
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()



def event_depth_bin_macro_mean(
    values: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    train_depth_bins: Sequence[float],
) -> torch.Tensor:
    """Average strata within each event, then average events with equal weight.

    The first and last frozen train-depth values are observed extrema; only the
    internal values define strata. Boundary values belong to the shallower stratum,
    matching ``numpy.digitize(..., right=True)`` in evaluation.
    """

    if values.shape != mask.shape:
        mask = torch.broadcast_to(mask, values.shape)
    if target.shape != values.shape:
        target = torch.broadcast_to(target, values.shape)
    mask = mask.to(dtype=torch.bool)
    boundaries = sorted(set(float(value) for value in train_depth_bins))
    internal = boundaries[1:-1] if len(boundaries) > 2 else []
    if not internal:
        return event_macro_masked_mean(values, mask, event_ids)

    batch_size = values.shape[0]
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    bin_boundaries = target.new_tensor(internal)
    # right=False assigns an exact boundary (for example 0.23 m) to the
    # shallower bin: (-inf, 0.23], (0.23, 0.48], (0.48, inf).
    bin_index = torch.bucketize(target, bin_boundaries, right=False)
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [index for index, item in enumerate(event_ids) if str(item) == event]
        event_values = values[indices]
        event_mask = mask[indices]
        event_bins = bin_index[indices]
        cell_losses: list[torch.Tensor] = []
        for depth_bin in range(len(internal) + 1):
            selected = event_mask & (event_bins == depth_bin)
            if torch.any(selected):
                cell_losses.append(event_values[selected].mean())
        if cell_losses:
            event_losses.append(torch.stack(cell_losses).mean())
    if not event_losses:
        return values.sum() * 0.0
    return torch.stack(event_losses).mean()



def depth_bin_macro_root_mean_square(
    error: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    train_depth_bins: Sequence[float],
) -> torch.Tensor:
    """Macro-average RMSE across frozen train-depth strata.

    The objective treats over- and under-prediction symmetrically. Each
    non-empty train-defined stratum receives one vote, so rare deep-water cells
    retain a squared-error gradient without dominating through raw pixel count.
    """

    if error.shape != mask.shape:
        mask = torch.broadcast_to(mask, error.shape)
    if target.shape != error.shape:
        target = torch.broadcast_to(target, error.shape)
    selected_mask = mask.to(dtype=torch.bool)
    boundaries = sorted(set(float(value) for value in train_depth_bins))
    internal = boundaries[1:-1] if len(boundaries) > 2 else []
    if not internal:
        mean_square = masked_micro_mean(error.square(), selected_mask)
        return torch.where(
            mean_square > 0.0,
            mean_square.clamp_min(1.0e-12).sqrt(),
            mean_square,
        )
    bin_index = torch.bucketize(target, target.new_tensor(internal), right=False)
    bin_rmse: list[torch.Tensor] = []
    for depth_bin in range(len(internal) + 1):
        selected = selected_mask & (bin_index == depth_bin)
        if torch.any(selected):
            mean_square = error[selected].square().mean()
            bin_rmse.append(
                torch.where(
                    mean_square > 0.0,
                    mean_square.clamp_min(1.0e-12).sqrt(),
                    mean_square,
                )
            )
    if not bin_rmse:
        return error.sum() * 0.0
    return torch.stack(bin_rmse).mean()



def event_depth_hierarchical_macro_root_mean_square(
    error: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    primary_depth_bins: Sequence[float],
    refined_depth_bins: Sequence[float],
) -> torch.Tensor:
    """Return event/depth-hierarchical RMSE with train-only boundaries."""

    hierarchical_mse = event_depth_hierarchical_macro_mean(
        error.square(),
        target,
        mask,
        event_ids,
        primary_depth_bins,
        refined_depth_bins,
    )
    return torch.where(
        hierarchical_mse > 0.0,
        hierarchical_mse.clamp_min(1.0e-12).sqrt(),
        hierarchical_mse,
    )



def event_depth_hierarchical_macro_mean(
    values: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    primary_depth_bins: Sequence[float],
    refined_depth_bins: Sequence[float],
) -> torch.Tensor:
    """Macro-average primary depth regimes, refining only within each regime.

    Primary regimes retain equal influence. The broad deep regime is split
    into train-only refined strata, whose means are averaged before the primary
    regime receives its one-vote weight. Thus adding tail resolution cannot
    silently make deep water occupy four times the weight of a primary regime.
    """

    if values.shape != mask.shape:
        mask = torch.broadcast_to(mask, values.shape)
    if target.shape != values.shape:
        target = torch.broadcast_to(target, values.shape)
    mask = mask.to(dtype=torch.bool)
    primary_edges = sorted(set(float(value) for value in primary_depth_bins))
    refined_edges = sorted(set(float(value) for value in refined_depth_bins))
    primary_internal = primary_edges[1:-1] if len(primary_edges) > 2 else []
    refined_internal = refined_edges[1:-1] if len(refined_edges) > 2 else []
    if not refined_internal or primary_edges == refined_edges:
        return event_depth_bin_macro_mean(
            values, target, mask, event_ids, refined_depth_bins
        )

    batch_size = values.shape[0]
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    primary_index = torch.bucketize(
        target, target.new_tensor(primary_internal), right=False
    )
    refined_index = torch.bucketize(
        target, target.new_tensor(refined_internal), right=False
    )
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [index for index, item in enumerate(event_ids) if str(item) == event]
        event_values = values[indices]
        event_mask = mask[indices]
        event_primary = primary_index[indices]
        event_refined = refined_index[indices]
        primary_losses: list[torch.Tensor] = []
        for primary_bin in range(len(primary_internal) + 1):
            primary_selected = event_mask & (event_primary == primary_bin)
            if not torch.any(primary_selected):
                continue
            refined_losses: list[torch.Tensor] = []
            for refined_bin in range(len(refined_internal) + 1):
                selected = primary_selected & (event_refined == refined_bin)
                if torch.any(selected):
                    refined_losses.append(event_values[selected].mean())
            primary_losses.append(torch.stack(refined_losses).mean())
        if primary_losses:
            event_losses.append(torch.stack(primary_losses).mean())
    if not event_losses:
        return values.sum() * 0.0
    return torch.stack(event_losses).mean()
