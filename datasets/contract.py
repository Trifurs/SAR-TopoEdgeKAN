"""Hash-aware data-contract utilities for the production input pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_GROUPS = (
    "label",
    "masks",
    "terrain",
    "s1_t1",
    "s1_t2",
    "s1_change",
    "s1_qa",
)
MODEL_CONTINUOUS_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain")


class ContractError(RuntimeError):
    """Raised when the audited dataset contract cannot be trusted."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve a path and reject traversal outside the dataset root."""

    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"Path escapes dataset root: {path}") from exc
    return resolved


@dataclass(frozen=True)
class DatasetContract:
    """Validated runtime view of an audited flood-depth dataset."""

    path: Path
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "DatasetContract":
        contract_path = Path(path).expanduser().resolve(strict=True)
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"Cannot read dataset contract {contract_path}: {exc}") from exc
        if payload.get("status") != "ready":
            raise ContractError(f"Dataset contract is not ready: {payload.get('status')!r}")
        groups = payload.get("raster_groups")
        if not isinstance(groups, dict):
            raise ContractError("Contract has no raster_groups mapping")
        missing = set(REQUIRED_GROUPS).difference(groups)
        if missing:
            raise ContractError(f"Contract is missing groups: {sorted(missing)}")
        return cls(contract_path, payload)

    def validate_input_groups(self, active_groups: tuple[str, ...]) -> None:
        missing = set(active_groups).difference(self.payload.get("raster_groups", {}))
        if missing:
            raise ContractError(f"Contract is missing active input groups: {sorted(missing)}")

    @property
    def dataset_root(self) -> Path:
        return Path(str(self.payload["dataset_root"])).expanduser().resolve(strict=True)

    @property
    def manifest_path(self) -> Path:
        manifest = self.payload["manifest"]
        explicit = manifest.get("path")
        if explicit is not None:
            path = Path(str(explicit)).expanduser()
            if not path.is_absolute():
                path = self.path.parent / path
            return path.resolve(strict=True)
        relative = Path(str(manifest["relative_path"]))
        return ensure_within(self.dataset_root / relative, self.dataset_root)

    @property
    def hash(self) -> str:
        return sha256_file(self.path)

    def group(self, name: str) -> Mapping[str, Any]:
        try:
            return self.payload["raster_groups"][name]
        except KeyError as exc:
            raise ContractError(f"Unknown raster group: {name}") from exc

    def band_index(self, group: str, description: str) -> int:
        """Return the zero-based index resolved from audited band descriptions."""

        descriptions = list(self.group(group)["band_descriptions"])
        try:
            return descriptions.index(description)
        except ValueError as exc:
            raise ContractError(
                f"Band {description!r} is absent from {group}: {descriptions}"
            ) from exc

    @property
    def main_input_channels(self) -> int:
        return sum(int(self.group(name)["band_count"]) for name in MODEL_CONTINUOUS_GROUPS)

    def verify_fingerprints(self, include_normalization: bool = True) -> None:
        """Fail closed when audited source files or statistics no longer match."""

        observed_manifest = sha256_file(self.manifest_path)
        expected_manifest = str(self.payload["manifest"]["sha256"])
        if observed_manifest != expected_manifest:
            raise ContractError(
                "Manifest fingerprint changed: "
                f"expected {expected_manifest}, observed {observed_manifest}"
            )
        for relative, expected_hash in self.payload.get("key_file_sha256", {}).items():
            source = ensure_within(self.dataset_root / relative, self.dataset_root)
            observed_hash = sha256_file(source)
            if observed_hash != expected_hash:
                raise ContractError(
                    f"Key data file changed ({relative}): expected {expected_hash}, "
                    f"observed {observed_hash}"
                )
        if include_normalization:
            selected = self.payload.get("normalization", {}).get("selected")
            if not isinstance(selected, Mapping) or not selected.get("sha256"):
                raise ContractError("Contract has no selected normalization fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)
