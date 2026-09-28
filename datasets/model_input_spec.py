"""Input contract for the production SAR-and-terrain pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


ACTIVE_GROUPS = (
    "label",
    "masks",
    "s1_t1",
    "s1_t2",
    "s1_change",
    "s1_qa",
    "terrain",
)
CONTINUOUS_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain")


@dataclass(frozen=True)
class ModelInputSpec:
    """Serializable whitelist of raster groups consumed by the model."""

    mode: str = "s1_terrain"
    active_groups: tuple[str, ...] = ACTIVE_GROUPS

    @classmethod
    def from_mode(cls, mode: str | None = None) -> "ModelInputSpec":
        if mode not in {None, "s1_terrain"}:
            raise ValueError("The production pipeline accepts only 's1_terrain'")
        return cls()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ModelInputSpec":
        dataset = config.get("dataset", config)
        if not isinstance(dataset, Mapping):
            raise ValueError("dataset configuration must be a mapping")
        if bool(dataset.get("sentinel2_enabled", False)):
            raise ValueError("Sentinel-2 is disabled for the production S1/terrain contract")
        return cls.from_mode(dataset.get("input_mode"))

    @property
    def is_s1_only(self) -> bool:
        return True

    @property
    def continuous_groups(self) -> tuple[str, ...]:
        return CONTINUOUS_GROUPS

    @property
    def qa_groups(self) -> tuple[str, ...]:
        return ("s1_qa",)

    def requires(self, group: str) -> bool:
        return group in self.active_groups

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active_groups": list(self.active_groups),
            "continuous_groups": list(self.continuous_groups),
            "qa_groups": list(self.qa_groups),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def active_groups_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(list(self.active_groups), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
