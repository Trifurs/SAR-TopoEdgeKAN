from __future__ import annotations

import numpy as np
import pytest

from metrics.aggregator import EvaluationAggregator


def test_evaluation_keeps_pixel_and_event_weighting_distinct() -> None:
    aggregator = EvaluationAggregator([0.1, 1.5, 4.0])
    aggregator.add(
        "a1", "event-a", np.array([1.0, 3.0, 9.0, 999.0]),
        np.array([1.0, 1.0, 3.0, 0.0]), np.array([True, True, True, False]),
    )
    aggregator.add(
        "b1", "event-b", np.array([2.0, 4.0]),
        np.array([1.0, 2.0]), np.ones(2, dtype=bool),
    )
    aggregator.add(
        "a2", "event-a", np.array([5.0]),
        np.array([1.0]), np.ones(1, dtype=bool),
    )
    summary, samples, events, bins = aggregator.summarize()
    assert summary["pixel_micro_mae"] == pytest.approx(2.5)
    assert summary["pixel_micro_rmse"] == pytest.approx(np.sqrt(61.0 / 6.0))
    assert summary["event_macro_mae"] == pytest.approx(2.25)
    assert summary["event_macro_rmse"] == pytest.approx((np.sqrt(14.0) + np.sqrt(2.5)) / 2.0)
    assert summary["balanced_composite_error_m"] == pytest.approx(0.25 * (
        summary["pixel_micro_mae"] + summary["pixel_micro_rmse"]
        + summary["event_macro_mae"] + summary["event_depth_hierarchical_macro_mae"]
    ))
    assert len(samples) == 3 and len(events) == 2 and len(bins) == 2
    assert sum(row["pixels"] for row in bins) == 6


def test_evaluation_rejects_an_empty_run() -> None:
    with pytest.raises(RuntimeError, match="No evaluation records"):
        EvaluationAggregator([0.1, 1.0, 4.0]).summarize()
