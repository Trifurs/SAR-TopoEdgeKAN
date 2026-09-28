"""Positive-depth and heteroscedastic losses with configurable aggregation."""

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


def tail_underprediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    tail_threshold_m: float,
    tail_margin_m: float = 0.0,
    beta_m: float = 0.25,
    aggregation: str = "pixel_micro",
) -> torch.Tensor:
    """Penalize only robust underprediction above a train-defined depth threshold."""

    if tail_threshold_m < 0 or tail_margin_m < 0 or beta_m <= 0:
        raise ValueError("invalid tail loss parameters")
    target = torch.broadcast_to(target, prediction.shape) if target.shape != prediction.shape else target
    positive_mask = torch.broadcast_to(positive_mask, prediction.shape) if positive_mask.shape != prediction.shape else positive_mask
    tail = positive_mask.bool() & (target >= float(tail_threshold_m))
    error = F.relu(target - prediction - float(tail_margin_m))
    penalty = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=beta_m)
    if aggregation == "pixel_micro":
        return masked_micro_mean(penalty, tail)
    if aggregation == "sample_macro":
        values = [masked_micro_mean(penalty[index:index + 1], tail[index:index + 1]) for index in range(prediction.shape[0]) if torch.any(tail[index])]
        return torch.stack(values).mean() if values else prediction.sum() * 0.0
    raise ValueError("aggregation must be pixel_micro or sample_macro")


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


def depth_bin_macro_mean(
    values: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    train_depth_bins: Sequence[float],
) -> torch.Tensor:
    """Average train-defined depth strata globally without using event identity."""

    if values.shape != mask.shape:
        mask = torch.broadcast_to(mask, values.shape)
    if target.shape != values.shape:
        target = torch.broadcast_to(target, values.shape)
    mask = mask.to(dtype=torch.bool)
    boundaries = sorted(set(float(value) for value in train_depth_bins))
    internal = boundaries[1:-1] if len(boundaries) > 2 else []
    if not internal:
        return masked_micro_mean(values, mask)
    bin_index = torch.bucketize(target, target.new_tensor(internal), right=False)
    bin_losses = [
        values[selected].mean()
        for depth_bin in range(len(internal) + 1)
        if torch.any(selected := mask & (bin_index == depth_bin))
    ]
    if not bin_losses:
        return values.sum() * 0.0
    return torch.stack(bin_losses).mean()


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


def sample_depth_bin_macro_mean(
    values: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    train_depth_bins: Sequence[float],
) -> torch.Tensor:
    """Average depth strata inside each raster, then average rasters equally.

    This reduction deliberately uses the batch dimension only.  It never reads a
    source-event identifier, so two rasters remain separate even if their metadata
    happen to name the same event.
    """

    batch_size = values.shape[0]
    sample_ids = [str(index) for index in range(batch_size)]
    return event_depth_bin_macro_mean(
        values,
        target,
        mask,
        sample_ids,
        train_depth_bins,
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

    For subset150 this preserves equal shallow/mid/deep influence. The broad deep
    regime is then split into train-only tail strata, whose means are averaged before
    the deep regime receives its one-third weight. Thus adding tail resolution cannot
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


def _smooth_absolute(value: torch.Tensor, beta: float) -> torch.Tensor:
    """Differentiable absolute penalty with a metre-valued transition width."""

    if beta <= 0:
        raise ValueError("bias beta must be positive")
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=beta)


def event_depth_hierarchical_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    primary_depth_bins: Sequence[float],
    refined_depth_bins: Sequence[float],
    beta: float,
) -> torch.Tensor:
    """Penalize signed cell-mean errors before event/depth macro averaging.

    A pixelwise robust loss can remain small while an entire event or depth regime is
    shifted in one direction. Flood-depth products are especially vulnerable to this
    failure because the valid positive labels are long-tailed and event clustered.
    This term first computes the signed mean error of each refined depth cell, applies
    a robust absolute penalty to that mean, averages cells inside their original
    shallow/mid/deep regime, and finally macro-averages regimes and events.
    """

    if prediction.shape != target.shape:
        target = torch.broadcast_to(target, prediction.shape)
    if prediction.shape != mask.shape:
        mask = torch.broadcast_to(mask, prediction.shape)
    mask = mask.to(dtype=torch.bool)
    primary_edges = sorted(set(float(value) for value in primary_depth_bins))
    refined_edges = sorted(set(float(value) for value in refined_depth_bins))
    primary_internal = primary_edges[1:-1] if len(primary_edges) > 2 else []
    refined_internal = refined_edges[1:-1] if len(refined_edges) > 2 else []

    batch_size = prediction.shape[0]
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    error = prediction - target
    primary_index = torch.bucketize(
        target, target.new_tensor(primary_internal), right=False
    )
    refined_index = torch.bucketize(
        target, target.new_tensor(refined_internal), right=False
    )
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [index for index, item in enumerate(event_ids) if str(item) == event]
        event_error = error[indices]
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
                    refined_losses.append(
                        _smooth_absolute(event_error[selected].mean(), beta)
                    )
            primary_losses.append(torch.stack(refined_losses).mean())
        if primary_losses:
            event_losses.append(torch.stack(primary_losses).mean())
    if not event_losses:
        return prediction.sum() * 0.0
    return torch.stack(event_losses).mean()


def positive_depth_losses(
    positive_depth: torch.Tensor,
    final_depth: torch.Tensor | None,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    lambda_log: float,
    lambda_final: float,
    train_depth_bins: Sequence[float] | None = None,
    primary_depth_bins: Sequence[float] | None = None,
    bias_beta: float = 0.1,
    underprediction_factor: float = 1.0,
    underprediction_min_depth_m: float = 0.0,
    aggregation_mode: str = "auto",
    linear_loss: str = "smooth_l1",
    depth_huber_beta_m: float = 1.0,
    log_depth_huber_beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    if underprediction_factor < 1.0:
        raise ValueError("underprediction_factor must be at least one")
    if underprediction_min_depth_m < 0.0:
        raise ValueError("underprediction_min_depth_m must be nonnegative")
    if linear_loss == "smooth_l1":
        linear_pixels = F.smooth_l1_loss(
            positive_depth, target, reduction="none", beta=depth_huber_beta_m
        )
    elif linear_loss == "l1":
        linear_pixels = (positive_depth - target).abs()
    else:
        raise ValueError(
            "linear_loss must be 'smooth_l1' or 'l1', "
            f"received {linear_loss!r}"
        )
    if underprediction_factor != 1.0:
        asymmetric = (
            (positive_depth.detach() < target)
            & (target >= underprediction_min_depth_m)
        )
        linear_pixels = linear_pixels * torch.where(
            asymmetric,
            linear_pixels.new_tensor(underprediction_factor),
            linear_pixels.new_tensor(1.0),
        )
    log_pixels = F.smooth_l1_loss(
        torch.log1p(positive_depth), torch.log1p(target.clamp_min(0.0)),
        reduction="none", beta=log_depth_huber_beta
    )
    valid_aggregation_modes = {
        "auto",
        "pixel_micro",
        "depth_bin_macro",
        "sample_depth_bin",
        "event_macro",
        "event_depth_bin",
        "event_depth_hierarchical",
    }
    if aggregation_mode not in valid_aggregation_modes:
        raise ValueError(
            f"Unknown aggregation_mode {aggregation_mode!r}; expected one of "
            f"{sorted(valid_aggregation_modes)}"
        )
    resolved_aggregation = aggregation_mode
    if resolved_aggregation == "auto":
        if train_depth_bins and primary_depth_bins:
            resolved_aggregation = "event_depth_hierarchical"
        elif train_depth_bins:
            resolved_aggregation = "event_depth_bin"
        else:
            resolved_aggregation = "event_macro"

    if resolved_aggregation == "pixel_micro":
        def reducer(pixels: torch.Tensor) -> torch.Tensor:
            return masked_micro_mean(pixels, positive_mask)
    elif resolved_aggregation == "depth_bin_macro":
        if not train_depth_bins:
            raise ValueError("depth_bin_macro aggregation requires train_depth_bins")

        def reducer(pixels: torch.Tensor) -> torch.Tensor:
            return depth_bin_macro_mean(
                pixels, target, positive_mask, train_depth_bins
            )
    elif resolved_aggregation == "sample_depth_bin":
        if not train_depth_bins:
            raise ValueError("sample_depth_bin aggregation requires train_depth_bins")

        def reducer(pixels: torch.Tensor) -> torch.Tensor:
            return sample_depth_bin_macro_mean(
                pixels, target, positive_mask, train_depth_bins
            )
    elif resolved_aggregation == "event_depth_hierarchical":
        if not train_depth_bins or not primary_depth_bins:
            raise ValueError(
                "event_depth_hierarchical aggregation requires both depth-bin lists"
            )

        def reducer(pixels: torch.Tensor) -> torch.Tensor:
            return event_depth_hierarchical_macro_mean(
                pixels,
                target,
                positive_mask,
                event_ids,
                primary_depth_bins,
                train_depth_bins,
            )
    elif resolved_aggregation == "event_depth_bin":
        if not train_depth_bins:
            raise ValueError("event_depth_bin aggregation requires train_depth_bins")

        def reducer(pixels: torch.Tensor) -> torch.Tensor:
            return event_depth_bin_macro_mean(
                pixels, target, positive_mask, event_ids, train_depth_bins
            )
    else:
        def reducer(pixels: torch.Tensor) -> torch.Tensor:
            return event_macro_masked_mean(pixels, positive_mask, event_ids)
    linear = reducer(linear_pixels)
    logarithmic = reducer(log_pixels)
    if final_depth is not None and lambda_final != 0.0:
        if linear_loss == "l1":
            final_pixels = (final_depth - target).abs()
        else:
            final_pixels = F.smooth_l1_loss(
                final_depth, target, reduction="none", beta=depth_huber_beta_m
            )
        final = reducer(final_pixels)
    else:
        final = linear.detach() * 0.0
    combined = linear + lambda_log * logarithmic + lambda_final * final
    if resolved_aggregation in {"pixel_micro", "depth_bin_macro"}:
        error = positive_depth - target
        selected = positive_mask.to(dtype=torch.bool)
        bias = (
            _smooth_absolute(error[selected].mean(), bias_beta)
            if torch.any(selected)
            else positive_depth.sum() * 0.0
        )
    elif resolved_aggregation == "sample_depth_bin":
        error = positive_depth - target
        sample_biases = []
        for index in range(error.shape[0]):
            selected = positive_mask[index].to(dtype=torch.bool)
            if torch.any(selected):
                sample_biases.append(
                    _smooth_absolute(error[index][selected].mean(), bias_beta)
                )
        bias = (
            torch.stack(sample_biases).mean()
            if sample_biases
            else positive_depth.sum() * 0.0
        )
    elif train_depth_bins and primary_depth_bins:
        bias = event_depth_hierarchical_bias_loss(
            positive_depth,
            target,
            positive_mask,
            event_ids,
            primary_depth_bins,
            train_depth_bins,
            bias_beta,
        )
    else:
        # Preserve useful behavior for external callers without frozen strata.
        error = positive_depth - target
        batch_size = error.shape[0]
        resolved_events = (
            [str(value) for value in event_ids]
            if event_ids is not None and len(event_ids) == batch_size
            else [str(index) for index in range(batch_size)]
        )
        event_biases: list[torch.Tensor] = []
        for event in dict.fromkeys(resolved_events):
            indices = [
                index for index, item in enumerate(resolved_events) if item == event
            ]
            selected = positive_mask[indices].to(dtype=torch.bool)
            if torch.any(selected):
                event_biases.append(
                    _smooth_absolute(error[indices][selected].mean(), bias_beta)
                )
        bias = (
            torch.stack(event_biases).mean()
            if event_biases
            else positive_depth.sum() * 0.0
        )
    return {
        "depth": combined,
        "depth_linear": linear,
        "depth_log": logarithmic,
        "depth_final": final,
        "depth_bias": bias,
    }


def event_depth_exceedance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    train_depth_bins: Sequence[float],
    temperature_m: float,
    aggregation_mode: str = "event_macro",
) -> torch.Tensor:
    """Preserve ordered exceedance decisions at frozen train-depth boundaries.

    The continuous prediction is converted to a soft exceedance logit at every
    internal train-only boundary. Thresholds and source events receive equal weight,
    so the rare deep tail contributes without replacing the metre-valued regressor or
    treating unlabeled pixels as dry negatives.
    """

    if temperature_m <= 0:
        raise ValueError("temperature_m must be positive")
    if aggregation_mode not in {
        "auto",
        "pixel_micro",
        "depth_bin_macro",
        "sample_depth_bin",
        "event_macro",
        "event_depth_bin",
        "event_depth_hierarchical",
    }:
        raise ValueError(f"Unknown aggregation_mode {aggregation_mode!r}")
    boundaries = sorted(set(float(value) for value in train_depth_bins))[1:-1]
    if not boundaries:
        return prediction.sum() * 0.0
    threshold_losses: list[torch.Tensor] = []
    for boundary in boundaries:
        logits = (prediction - boundary) / temperature_m
        expected = (target > boundary).to(dtype=prediction.dtype)
        pixels = F.binary_cross_entropy_with_logits(
            logits, expected, reduction="none"
        )
        if aggregation_mode == "pixel_micro":
            threshold_losses.append(masked_micro_mean(pixels, positive_mask))
        elif aggregation_mode == "depth_bin_macro":
            threshold_losses.append(
                depth_bin_macro_mean(
                    pixels, target, positive_mask, train_depth_bins
                )
            )
        elif aggregation_mode == "sample_depth_bin":
            threshold_losses.append(
                event_macro_masked_mean(pixels, positive_mask, None)
            )
        elif aggregation_mode in {"auto", "event_macro"}:
            threshold_losses.append(
                event_macro_masked_mean(pixels, positive_mask, event_ids)
            )
        else:
            # Exceedance thresholds are already balanced explicitly. Event/depth
            # modes use event-macro reduction.
            threshold_losses.append(
                event_macro_masked_mean(pixels, positive_mask, event_ids)
            )
    return torch.stack(threshold_losses).mean()


def laplace_nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    positive_mask: torch.Tensor,
    event_ids: Sequence[str] | None,
    train_depth_bins: Sequence[float] | None = None,
    primary_depth_bins: Sequence[float] | None = None,
    aggregation_mode: str = "auto",
) -> torch.Tensor:
    stable_scale = scale.clamp(1e-4, 20.0)
    pixels = (prediction - target).abs() / stable_scale + torch.log(stable_scale)
    valid_aggregation_modes = {
        "auto",
        "pixel_micro",
        "depth_bin_macro",
        "sample_depth_bin",
        "event_macro",
        "event_depth_bin",
        "event_depth_hierarchical",
    }
    if aggregation_mode not in valid_aggregation_modes:
        raise ValueError(
            f"Unknown aggregation_mode {aggregation_mode!r}; expected one of "
            f"{sorted(valid_aggregation_modes)}"
        )
    resolved_aggregation = aggregation_mode
    if resolved_aggregation == "auto":
        if train_depth_bins and primary_depth_bins:
            resolved_aggregation = "event_depth_hierarchical"
        elif train_depth_bins:
            resolved_aggregation = "event_depth_bin"
        else:
            resolved_aggregation = "event_macro"
    if resolved_aggregation == "pixel_micro":
        return masked_micro_mean(pixels, positive_mask)
    if resolved_aggregation == "depth_bin_macro":
        if not train_depth_bins:
            raise ValueError("depth_bin_macro aggregation requires train_depth_bins")
        return depth_bin_macro_mean(
            pixels, target, positive_mask, train_depth_bins
        )
    if resolved_aggregation == "sample_depth_bin":
        if not train_depth_bins:
            raise ValueError("sample_depth_bin aggregation requires train_depth_bins")
        return sample_depth_bin_macro_mean(
            pixels, target, positive_mask, train_depth_bins
        )
    if resolved_aggregation == "event_depth_hierarchical" and not (
        train_depth_bins and primary_depth_bins
    ):
        raise ValueError(
            "event_depth_hierarchical aggregation requires both depth-bin lists"
        )
    if resolved_aggregation == "event_depth_bin" and not train_depth_bins:
        raise ValueError("event_depth_bin aggregation requires train_depth_bins")
    if resolved_aggregation == "event_macro":
        return event_macro_masked_mean(pixels, positive_mask, event_ids)
    if resolved_aggregation == "event_depth_hierarchical":
        return event_depth_hierarchical_macro_mean(
            pixels,
            target,
            positive_mask,
            event_ids,
            primary_depth_bins,
            train_depth_bins,
        )
    return event_depth_bin_macro_mean(
        pixels, target, positive_mask, event_ids, train_depth_bins
    )
