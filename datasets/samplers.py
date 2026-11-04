"""Tempered event-balanced sampling with stable remainder batches."""

from __future__ import annotations

from collections import Counter
import logging
import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import BatchSampler, Sampler, WeightedRandomSampler


LOGGER = logging.getLogger(__name__)


def event_weights(
    event_ids: Sequence[str], balance_power: float = 1.0
) -> torch.Tensor:
    """Return per-sample weights proportional to ``event_count ** -power``.

    ``power=0`` is ordinary uniform sampling, while ``power=1`` gives every
    source event the same expected mass.  Fractional powers provide a stable
    compromise for highly imbalanced event catalogues without repeatedly
    presenting singleton events as often as large multi-patch events.
    """

    if not math.isfinite(balance_power) or not 0.0 <= balance_power <= 1.0:
        raise ValueError("event balance power must lie in [0, 1]")
    if not event_ids or any(not event for event in event_ids):
        LOGGER.warning("source_event_id is missing; falling back to uniform sampling")
        return torch.ones(len(event_ids), dtype=torch.double)
    counts = Counter(event_ids)
    return torch.tensor(
        [float(counts[event]) ** -float(balance_power) for event in event_ids],
        dtype=torch.double,
    )


def make_event_balanced_sampler(
    event_ids: Sequence[str], seed: int, balance_power: float = 1.0
) -> WeightedRandomSampler:
    """Return a tempered event-balanced sampler with replacement."""

    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        event_weights(event_ids, balance_power),
        len(event_ids),
        replacement=True,
        generator=generator,
    )


class BalancedRemainderBatchSampler(BatchSampler):
    """Batch a sampler without creating a tiny final remainder batch.

    The number of batches is ``ceil(n / batch_size)`` (unless ``drop_last`` is
    requested), and indices are partitioned into sizes differing by at most one.
    This keeps every sample while avoiding a singleton for ordinary datasets.
    """

    def __init__(
        self, sampler: Sampler[int], batch_size: int, drop_last: bool = False
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(iter(self.sampler))
        if self.drop_last:
            count = len(indices) // self.batch_size
            usable = indices[: count * self.batch_size]
            sizes = [self.batch_size] * count
        else:
            count = math.ceil(len(indices) / self.batch_size)
            if count == 0:
                return
            base, remainder = divmod(len(indices), count)
            sizes = [base + (index < remainder) for index in range(count)]
            usable = indices
        offset = 0
        for size in sizes:
            yield usable[offset : offset + size]
            offset += size

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)
