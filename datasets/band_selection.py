"""Contract-resolved model input band selections.

The dataset contract describes every available raster band.  :class:`BandSpec`
describes the smaller, ordered view consumed by one model run without mutating the
contract or relying on positional channel assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec


MODEL_BAND_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain", "s1_conditioning")


@dataclass(frozen=True)
class SelectedBands:
    names: tuple[str, ...]
    indexes: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "indexes": list(self.indexes), "channels": len(self.names)}


@dataclass(frozen=True)
class BandSpec:
    """An immutable, ordered selection resolved against exact descriptions."""

    groups: Mapping[str, SelectedBands]
    uses_default_bands: bool = False

    @classmethod
    def resolve(
        cls,
        contract: DatasetContract,
        configured: Mapping[str, Sequence[str]] | None,
        required_groups: Sequence[str] | None = None,
    ) -> "BandSpec":
        required = tuple(required_groups or ("s1_t1", "s1_t2", "s1_change", "terrain"))
        if configured is None:
            return cls(
                {
                    group: SelectedBands(
                        tuple(str(name) for name in contract.group(group)["band_descriptions"]),
                        tuple(range(len(contract.group(group)["band_descriptions"]))),
                    )
                    for group in required
                } | {"s1_conditioning": SelectedBands((), ())},
                uses_default_bands=True,
            )
        unknown_groups = set(configured).difference(MODEL_BAND_GROUPS)
        if unknown_groups:
            raise ValueError(f"Unknown model band groups: {sorted(unknown_groups)}")
        resolved: dict[str, SelectedBands] = {}
        for group in required:
            requested = tuple(str(value) for value in configured.get(group, ()))
            if len(requested) != len(set(requested)):
                raise ValueError(f"Duplicate band names in model_bands.{group}: {requested}")
            available = tuple(str(value) for value in contract.group(group)["band_descriptions"])
            missing = [name for name in requested if name not in available]
            if missing:
                raise ValueError(
                    f"Unknown band(s) for {group}: {missing}; available={list(available)}"
                )
            resolved[group] = SelectedBands(
                requested, tuple(available.index(name) for name in requested)
            )
        if "s1_t1" in resolved and "s1_t2" in resolved:
            cls._validate_temporal_pair(resolved["s1_t1"].names, resolved["s1_t2"].names, "S1")
        conditioning = tuple(str(value) for value in configured.get("s1_conditioning", ()))
        if len(conditioning) != len(set(conditioning)):
            raise ValueError(f"Duplicate s1_conditioning names: {conditioning}")
        temporal_available = {
            str(name): (group, index)
            for group in ("s1_t1", "s1_t2")
            for index, name in enumerate(contract.group(group)["band_descriptions"])
        }
        missing_conditioning = [name for name in conditioning if name not in temporal_available]
        if missing_conditioning:
            raise ValueError(
                "Unknown S1 conditioning band(s): "
                f"{missing_conditioning}; available={sorted(temporal_available)}"
            )
        # Conditioning indexes are relative to its source group and are serialized
        # mainly for audit.  ``conditioning_sources`` is the authoritative mapping.
        resolved["s1_conditioning"] = SelectedBands(
            conditioning, tuple(temporal_available[name][1] for name in conditioning)
        )
        empty_required = [group for group in required if not resolved[group].names]
        if empty_required:
            raise ValueError(f"Model band groups cannot be empty: {empty_required}")
        return cls(resolved, uses_default_bands=False)

    @staticmethod
    def _temporal_base(name: str) -> str:
        """Normalize only the temporal marker, retaining band semantics."""

        value = str(name)
        for marker in ("_pre_", "_event_"):
            value = value.replace(marker, "_")
        if value.endswith("_pre") or value.endswith("_event"):
            value = value.rsplit("_", 1)[0]
        return value

    @classmethod
    def _validate_temporal_pair(
        cls, pre: Sequence[str], event: Sequence[str], modality: str
    ) -> None:
        if len(pre) != len(event):
            raise ValueError(
                f"{modality} T1/T2 band counts differ: {len(pre)} != {len(event)}"
            )
        pre_base = tuple(cls._temporal_base(name) for name in pre)
        event_base = tuple(cls._temporal_base(name) for name in event)
        if pre_base != event_base:
            raise ValueError(
                f"{modality} T1/T2 semantic/order mismatch: "
                f"T1={list(pre)} T2={list(event)}"
            )

    def names(self, group: str) -> tuple[str, ...]:
        return self.groups[group].names

    def indexes(self, group: str) -> tuple[int, ...]:
        return self.groups[group].indexes

    def channels(self, group: str) -> int:
        return len(self.names(group))

    def conditioning_sources(self, contract: DatasetContract) -> tuple[tuple[str, int], ...]:
        result: list[tuple[str, int]] = []
        for name in self.names("s1_conditioning") if "s1_conditioning" in self.groups else ():
            for group in ("s1_t1", "s1_t2"):
                descriptions = list(contract.group(group)["band_descriptions"])
                if name in descriptions:
                    result.append((group, descriptions.index(name)))
                    break
        return tuple(result)

    def read_indexes(self, contract: DatasetContract, group: str) -> tuple[int, ...]:
        """Return zero-based indexes needed from a continuous raster group."""

        indexes = list(self.indexes(group))
        for source_group, source_index in self.conditioning_sources(contract):
            if source_group == group and source_index not in indexes:
                indexes.append(source_index)
        return tuple(indexes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uses_default_bands": self.uses_default_bands,
            "groups": {group: selected.as_dict() for group, selected in self.groups.items()},
            "channel_counts": {group: len(selected.names) for group, selected in self.groups.items()},
        }


def resolve_band_spec(config: Mapping[str, Any], contract: DatasetContract) -> BandSpec:
    dataset = config.get("dataset", config)
    configured = dataset.get("model_bands") if isinstance(dataset, Mapping) else None
    if configured is not None and not isinstance(configured, Mapping):
        raise ValueError("dataset.model_bands must be a mapping")
    input_spec = ModelInputSpec.from_config(config)
    contract.validate_input_groups(input_spec.active_groups)
    return BandSpec.resolve(contract, configured, input_spec.continuous_groups)
