from __future__ import annotations

import pytest
import torch

from datasets.samplers import event_weights


def test_tempered_event_weights_interpolate_uniform_and_fully_balanced() -> None:
    event_ids = ["large", "large", "large", "large", "small"]
    uniform = event_weights(event_ids, 0.0)
    tempered = event_weights(event_ids, 0.5)
    balanced = event_weights(event_ids, 1.0)

    assert torch.allclose(uniform, torch.ones_like(uniform))
    assert tempered[-1] / tempered[0] == pytest.approx(2.0)
    assert balanced[-1] / balanced[0] == pytest.approx(4.0)


@pytest.mark.parametrize("power", [-0.01, 1.01, float("nan")])
def test_event_balance_power_rejects_invalid_values(power: float) -> None:
    with pytest.raises(ValueError, match="balance power"):
        event_weights(["event"], power)
