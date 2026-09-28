"""Reliability channels shared by the SAR model and dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


RELIABILITY_NAMES = (
    "s1_event_observation_count_z",
    "s1_event_day_z",
    "s1_available",
    "dem_available",
    "event_duration_log_scaled",
    "s1_day_missing",
)


@dataclass(frozen=True)
class ReliabilitySpec:
    names: tuple[str, ...] = RELIABILITY_NAMES

    @classmethod
    def from_mode(cls, mode: str | None = None) -> "ReliabilitySpec":
        if mode not in {None, "s1_terrain"}:
            raise ValueError("The production pipeline accepts only the SAR reliability schema")
        return cls()

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Reliability channel {name!r} is not active") from exc

    def as_dict(self) -> dict[str, object]:
        return {"names": list(self.names), "channels": len(self.names)}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()
