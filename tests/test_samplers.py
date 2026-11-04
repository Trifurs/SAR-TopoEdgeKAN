from __future__ import annotations

import pytest
import torch

from datasets.samplers import BalancedRemainderBatchSampler, event_weights, make_event_balanced_sampler
from datasets.transforms import SynchronousAugment


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


def test_active_replacement_sampler_preserves_seeded_draws_across_epochs() -> None:
    event_ids = ["large"] * 13 + ["medium"] * 3 + ["small"]
    sampler = make_event_balanced_sampler(event_ids, seed=20260908, balance_power=0.25)
    expected_generator = torch.Generator().manual_seed(20260908)
    weights = torch.tensor([13 ** -0.25] * 13 + [3 ** -0.25] * 3 + [1.0], dtype=torch.double)
    for _ in range(3):
        expected = torch.multinomial(weights, len(event_ids), replacement=True, generator=expected_generator).tolist()
        batches = list(BalancedRemainderBatchSampler(sampler, batch_size=12))
        assert [len(batch) for batch in batches] == [9, 8]
        assert [index for batch in batches for index in batch] == expected


def test_geometric_augmentation_preserves_alignment_and_rng() -> None:
    source = torch.arange(35).reshape(1, 5, 7)
    torch.manual_seed(42)
    expected = []
    for _ in range(5):
        value = source
        if torch.rand(()) < 0.5:
            value = torch.flip(value, dims=(-1,))
        if torch.rand(()) < 0.5:
            value = torch.flip(value, dims=(-2,))
        if torch.rand(()) < 0.25:
            value = torch.rot90(value, int(torch.randint(1, 4, ()).item()), dims=(-2, -1))
        expected.append(value)
    after_expected = torch.rand(4)
    torch.manual_seed(42)
    augment = SynchronousAugment(0.5, 0.5, 0.25)
    for value in expected:
        sample = augment({"sar": source, "validity": {"support": source + 10}})
        assert torch.equal(sample["sar"], value)
        assert torch.equal(sample["validity"]["support"], value + 10)
    assert torch.equal(torch.rand(4), after_expected)
