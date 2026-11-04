"""Contract-resolved model input band selections.

The dataset contract describes the audited raster view. :class:`BandSpec`
resolves the ordered channels consumed by one model run without mutating the
contract or admitting unmodeled source bands into temporal SAR states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec


MODEL_BAND_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain")
S1_POLARIZATION_BANDS = {
    "s1_t1": ("VV_pre_db", "VH_pre_db"),
    "s1_t2": ("VV_event_db", "VH_event_db"),
}


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
        uses_default_bands = configured is None
        if configured is None:
            configured = {
                group: S1_POLARIZATION_BANDS.get(
                    group, tuple(str(name) for name in contract.group(group)["band_descriptions"])
                )
                for group in required
            }
        unknown_groups = set(configured).difference(MODEL_BAND_GROUPS)
        if unknown_groups:
            raise ValueError(f"Unknown model band groups: {sorted(unknown_groups)}")
        resolved: dict[str, SelectedBands] = {}
        for group in required:
            requested = tuple(str(value) for value in configured.get(group, ()))
            if len(requested) != len(set(requested)):
                raise ValueError(f"Duplicate band names in model_bands.{group}: {requested}")
            if group in S1_POLARIZATION_BANDS:
                unsupported = set(requested).difference(S1_POLARIZATION_BANDS[group])
                if unsupported:
                    raise ValueError(
                        f"Only VV/VH polarization bands are supported for {group}: {sorted(unsupported)}"
                    )
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
        empty_required = [group for group in required if not resolved[group].names]
        if empty_required:
            raise ValueError(f"Model band groups cannot be empty: {empty_required}")
        return cls(resolved, uses_default_bands=uses_default_bands)

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

    def read_indexes(self, contract: DatasetContract, group: str) -> tuple[int, ...]:
        """Return zero-based indexes needed from a continuous raster group."""

        return self.indexes(group)

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
