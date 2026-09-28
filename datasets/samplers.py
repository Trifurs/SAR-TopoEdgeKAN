"""Event-aware samplers with an exact, without-replacement training epoch."""

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


def _event_interleaved_order(
    event_ids: Sequence[str], generator: torch.Generator
) -> list[int]:
    """Return every sample index once while spreading repeated events apart."""

    if not event_ids:
        return []
    groups: dict[str, list[int]] = {}
    for index, event_id in enumerate(event_ids):
        key = str(event_id) if event_id else f"__missing_event_{index}"
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[position] for position in permutation]

    event_names = list(groups)
    order: list[int] = []
    while event_names:
        permutation = torch.randperm(len(event_names), generator=generator).tolist()
        shuffled_events = [event_names[position] for position in permutation]
        remaining: list[str] = []
        for event_name in shuffled_events:
            order.append(groups[event_name].pop())
            if groups[event_name]:
                remaining.append(event_name)
        event_names = remaining
    return order


class EventEpochSampler(Sampler[int]):
    """Visit every training sample exactly once per deterministic epoch."""

    def __init__(self, event_ids: Sequence[str], seed: int) -> None:
        self.event_ids = tuple(str(value) for value in event_ids)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(_event_interleaved_order(self.event_ids, generator))

    def __len__(self) -> int:
        return len(self.event_ids)


class DistributedEventBalancedSampler(Sampler[int]):
    """Draw one global weighted epoch deterministically and shard it across ranks."""

    def __init__(
        self,
        event_ids: Sequence[str],
        num_replicas: int,
        rank: int,
        seed: int,
        balance_power: float = 1.0,
    ) -> None:
        self.weights = event_weights(event_ids, balance_power)
        self.dataset_size = len(event_ids)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = int(math.ceil(self.dataset_size / num_replicas))
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


class DistributedEventEpochSampler(Sampler[int]):
    """Shard one global without-replacement event-aware order across DDP ranks.

    As with PyTorch's ``DistributedSampler``, at most ``world_size - 1`` padded
    indices are repeated so every rank executes the same number of optimizer steps.
    No padding occurs in the single-process training path used for subset150.
    """

    def __init__(
        self, event_ids: Sequence[str], num_replicas: int, rank: int, seed: int
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas})")
        self.event_ids = tuple(str(value) for value in event_ids)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.event_ids) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = _event_interleaved_order(self.event_ids, generator)
        if order and len(order) < self.total_size:
            order.extend(order[: self.total_size - len(order)])
        return iter(order[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


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
