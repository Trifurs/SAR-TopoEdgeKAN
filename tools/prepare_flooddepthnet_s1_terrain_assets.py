#!/usr/bin/env python3
"""Build audited S1/terrain assets for the complete FloodDepthNet release.

The full release ships a rich manifest that also records Sentinel-2 paths.  This
utility deliberately registers and opens only the S1, S1-QA, DEM, label, and
mask columns required by the ``s1_terrain`` input contract.  It validates every
listed active raster, calculates train-only robust normalization statistics, and
binds both outputs to the immutable full-data manifest.

The percentile estimates use deterministic, uniform per-band reservoirs.  Means,
standard deviations, minima, maxima, validity counts, and split counts are exact
over the scanned rasters.  The reservoir size is recorded in the resulting
statistics artifact and can be increased without changing the training code.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio

from datasets.contract import sha256_file
from utils.misc import atomic_write_json


FULL_DATASET_ROOT = Path(
    "/media/whu/0d7bb559-7b14-4875-843f-08befb3ca56b/myData/FloodDepthNet"
)
MANIFEST_RELATIVE_PATH = Path("metadata/training_manifest.csv")
READY_MARKER_RELATIVE_PATH = Path("REBALANCED_READY.json")
CORE_MANIFEST_RELATIVE_PATH = Path("metadata/training_core_s1_dem.csv")
SPLIT_COMPONENTS_RELATIVE_PATH = Path("metadata/split_components.csv")

SCOPE = "train split only; valid pixels only; val/test excluded"
EXPECTED_SPLITS = ("train", "val", "test")
CONTINUOUS_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain")
QA_TRANSFORMS = {
    "event_observation_count": "log1p(max(x,0))",
    "selected_event_day_offset": "clip(x/event_duration_days,0,1); x<0 is missing",
}
REQUIRED_MASKS = (
    "valid_depth_mask",
    "DEM_valid_mask",
    "slope_valid_mask",
    "S1_event_composite_valid_mask",
)
# These are the only event bands selected by configs/base/datasets/
# flooddepthnet_s1_terrain.xml.  Keeping the list here makes the canonical
# output-validity accounting auditable and prevents accidental S2 expansion.
SELECTED_EVENT_BANDS = ("VV_event_db", "VH_event_db")


@dataclass(frozen=True)
class GroupSpec:
    """One active raster family in the S1/terrain data contract."""

    path_column: str
    role: str
    bands: tuple[str, ...] | None = None


GROUP_SPECS = {
    "label": GroupSpec("label_path", "target"),
    "masks": GroupSpec("masks_path", "mask_only"),
    "s1_t1": GroupSpec("s1_t1_path", "model_input", ("VV_pre_db", "VH_pre_db")),
    "s1_t2": GroupSpec("s1_t2_path", "model_input", ("VV_event_db", "VH_event_db")),
    "s1_change": GroupSpec("s1_change_path", "model_input"),
    "s1_qa": GroupSpec("s1_qa_path", "quality_only"),
    "terrain": GroupSpec("dem_path", "model_input"),
}


class DatasetPreparationError(RuntimeError):
    """Raised when the full-data S1/terrain contract cannot be trusted."""


def _masked_validity(values: np.ndarray, raster_masks: np.ndarray, nodata: float | None) -> np.ndarray:
    """Mirror the runtime loader's finite/nodata validity semantics."""

    valid = raster_masks > 0
    valid &= np.isfinite(values)
    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(values)
        else:
            valid &= values != nodata
    return valid


class UniformReservoir:
    """A deterministic uniform sample that can be merged one raster at a time."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.values = np.empty(self.capacity, dtype=np.float32)
        self.size = 0
        self.seen = 0

    def update(self, values: np.ndarray) -> None:
        incoming = np.asarray(values, dtype=np.float32).reshape(-1)
        incoming_count = int(incoming.size)
        if incoming_count == 0:
            return
        if self.seen < self.capacity:
            existing = self.values[: self.size]
            merged = np.concatenate((existing, incoming))
            if merged.size <= self.capacity:
                self.values[: merged.size] = merged
                self.size = int(merged.size)
            else:
                choice = self.rng.choice(merged.size, size=self.capacity, replace=False)
                self.values[:] = merged[choice]
                self.size = self.capacity
            self.seen += incoming_count
            return

        selected_from_incoming = int(
            self.rng.hypergeometric(
                ngood=incoming_count,
                nbad=self.seen,
                nsample=self.capacity,
            )
        )
        if selected_from_incoming:
            incoming_indexes = self.rng.choice(
                incoming_count, size=selected_from_incoming, replace=False
            )
            reservoir_indexes = self.rng.choice(
                self.capacity, size=selected_from_incoming, replace=False
            )
            self.values[reservoir_indexes] = incoming[incoming_indexes]
        self.seen += incoming_count

    def quantiles(self, quantiles: Iterable[float]) -> list[float]:
        if self.size == 0:
            raise DatasetPreparationError("cannot compute quantiles from an empty reservoir")
        return [
            float(value)
            for value in np.quantile(self.values[: self.size], list(quantiles))
        ]


@dataclass
class ValueAccumulator:
    """Exact moments plus a deterministic robust-percentile reservoir."""

    reservoir_capacity: int
    seed: int
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    reservoir: UniformReservoir = field(init=False)

    def __post_init__(self) -> None:
        self.reservoir = UniformReservoir(self.reservoir_capacity, self.seed)

    def update(self, values: np.ndarray) -> None:
        selected = np.asarray(values, dtype=np.float64).reshape(-1)
        if selected.size == 0:
            return
        if not np.isfinite(selected).all():
            raise DatasetPreparationError("a normalization accumulator received non-finite values")
        batch_count = int(selected.size)
        batch_mean = float(selected.mean())
        batch_m2 = float(np.square(selected - batch_mean).sum(dtype=np.float64))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
            self.mean += delta * batch_count / total
            self.count = total
        self.minimum = min(self.minimum, float(selected.min()))
        self.maximum = max(self.maximum, float(selected.max()))
        self.reservoir.update(selected.astype(np.float32, copy=False))

    def summary(self, band: str, transform: str) -> dict[str, float | int | str]:
        if self.count <= 0:
            raise DatasetPreparationError(f"no valid train values were found for {band}")
        p005, p01, p99, p995 = self.reservoir.quantiles((0.005, 0.01, 0.99, 0.995))
        return {
            "band": band,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": math.sqrt(max(self.m2 / self.count, 0.0)),
            "p0.5": p005,
            "p1": p01,
            "p99": p99,
            "p99.5": p995,
            "transform": transform,
        }


@dataclass
class RasterGroupAudit:
    """Structural and aggregate audit state for one active raster group."""

    name: str
    spec: GroupSpec
    reservoir_capacity: int
    seed: int
    descriptions: tuple[str, ...] | None = None
    dtypes: tuple[str, ...] | None = None
    width: int | None = None
    height: int | None = None
    nodata: float | None = None
    resolution: tuple[float, float] | None = None
    crs_counts: dict[str, int] = field(default_factory=dict)
    value_counts: list[int] = field(default_factory=list)
    invalid_counts: list[int] = field(default_factory=list)
    nonfinite_counts: list[int] = field(default_factory=list)
    minima: list[float] = field(default_factory=list)
    maxima: list[float] = field(default_factory=list)

    def observe(
        self,
        path: Path,
        sample_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        with rasterio.open(path) as raster:
            available = tuple(str(item) for item in raster.descriptions)
            descriptions = self.spec.bands or available
            missing = set(descriptions).difference(available)
            if missing:
                raise DatasetPreparationError(
                    f"{self.name} is missing required bands for {sample_id}: {sorted(missing)}"
                )
            if self.spec.bands is not None and available[:len(descriptions)] != descriptions:
                raise DatasetPreparationError(
                    f"{self.name} selected bands must match the declared leading raster view for {sample_id}"
                )
            indexes = [available.index(name) + 1 for name in descriptions]
            values = raster.read(indexes=indexes, masked=False)
            raster_masks = raster.read_masks(indexes=indexes)
            valid = _masked_validity(values, raster_masks, raster.nodata)
            if any(not item or item == "None" for item in descriptions):
                raise DatasetPreparationError(
                    f"{self.name} has unnamed bands in sample {sample_id}: {path}"
                )
            dtypes = tuple(str(raster.dtypes[index - 1]) for index in indexes)
            nodata = None if raster.nodata is None else float(raster.nodata)
            resolution = tuple(float(item) for item in raster.res)
            crs = raster.crs.to_string() if raster.crs else None
            shape = (int(raster.height), int(raster.width))
        if self.descriptions is None:
            self.descriptions = descriptions
            self.dtypes = dtypes
            self.height, self.width = shape
            self.nodata = nodata
            self.resolution = resolution
            bands = len(descriptions)
            self.value_counts = [0] * bands
            self.invalid_counts = [0] * bands
            self.nonfinite_counts = [0] * bands
            self.minima = [math.inf] * bands
            self.maxima = [-math.inf] * bands
        elif (
            descriptions != self.descriptions
            or dtypes != self.dtypes
            or shape != (self.height, self.width)
            or nodata != self.nodata
            or resolution != self.resolution
        ):
            raise DatasetPreparationError(
                f"{self.name} raster contract mismatch for {sample_id}: {path}"
            )
        self.crs_counts[str(crs)] = self.crs_counts.get(str(crs), 0) + 1
        for band_index in range(values.shape[0]):
            band_values = values[band_index]
            band_valid = valid[band_index]
            valid_values = band_values[band_valid]
            self.value_counts[band_index] += int(valid_values.size)
            self.invalid_counts[band_index] += int(band_values.size - valid_values.size)
            self.nonfinite_counts[band_index] += int((~np.isfinite(band_values)).sum())
            if valid_values.size:
                self.minima[band_index] = min(
                    self.minima[band_index], float(valid_values.min())
                )
                self.maxima[band_index] = max(
                    self.maxima[band_index], float(valid_values.max())
                )
        return values, valid

    def contract_payload(self) -> dict[str, Any]:
        if self.descriptions is None or self.dtypes is None:
            raise DatasetPreparationError(f"no samples were observed for {self.name}")
        if len(self.crs_counts) != 1:
            raise DatasetPreparationError(
                f"{self.name} contains multiple coordinate reference systems: {self.crs_counts}"
            )
        aggregate_values = []
        for index, description in enumerate(self.descriptions):
            aggregate_values.append(
                {
                    "description": description,
                    "invalid_pixels": self.invalid_counts[index],
                    "maximum": self.maxima[index],
                    "minimum": self.minima[index],
                    "nonfinite_pixels": self.nonfinite_counts[index],
                    "valid_pixels": self.value_counts[index],
                }
            )
        return {
            "aggregate_values": aggregate_values,
            "band_count": len(self.descriptions),
            "band_descriptions": list(self.descriptions),
            "crs": next(iter(self.crs_counts)),
            "dtypes": list(self.dtypes),
            "height": self.height,
            "nodata": self.nodata,
            "observed_crs_counts": dict(sorted(self.crs_counts.items())),
            "path_column": self.spec.path_column,
            "resolution": list(self.resolution or ()),
            "role": self.spec.role,
            "width": self.width,
        }


def _duration_days(row: Mapping[str, str]) -> int:
    return max(
        1,
        (date.fromisoformat(str(row["event_end"])) - date.fromisoformat(str(row["event_start"]))).days
        + 1,
    )


def _new_accumulator_map(
    group_audits: Mapping[str, RasterGroupAudit], reservoir_capacity: int, seed: int
) -> dict[str, list[ValueAccumulator]]:
    result: dict[str, list[ValueAccumulator]] = {}
    for group_index, group in enumerate(CONTINUOUS_GROUPS):
        audit = group_audits[group]
        if audit.descriptions is None:
            raise DatasetPreparationError(f"cannot initialize statistics before observing {group}")
        result[group] = [
            ValueAccumulator(reservoir_capacity, seed + 10_000 * group_index + band_index)
            for band_index, _ in enumerate(audit.descriptions)
        ]
    return result


def _read_manifest(
    dataset_root: Path, manifest_path: Path | None = None
) -> tuple[list[dict[str, str]], list[str]]:
    manifest_path = (
        dataset_root / MANIFEST_RELATIVE_PATH
        if manifest_path is None
        else manifest_path.expanduser().resolve(strict=True)
    )
    if not manifest_path.is_file():
        raise DatasetPreparationError(f"missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetPreparationError("manifest has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    required = {"sample_id", "split", "event_start", "event_end", *(
        spec.path_column for spec in GROUP_SPECS.values()
    )}
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise DatasetPreparationError(
            "The complete FloodDepthNet manifest does not expose required S1/terrain "
            f"columns: {missing}"
        )
    if not rows:
        raise DatasetPreparationError("manifest is empty")
    seen_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            raise DatasetPreparationError(f"missing or duplicate sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        if row.get("split") not in EXPECTED_SPLITS:
            raise DatasetPreparationError(f"unsupported split for {sample_id}: {row.get('split')!r}")
    return rows, fieldnames


def _load_ready_marker(dataset_root: Path, manifest_sha256: str) -> dict[str, Any]:
    marker_path = dataset_root / READY_MARKER_RELATIVE_PATH
    if not marker_path.is_file():
        raise DatasetPreparationError(f"missing release readiness marker: {marker_path}")
    import json

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if str(marker.get("status", "")).lower().startswith("ready") is False:
        raise DatasetPreparationError(f"release readiness marker is not ready: {marker.get('status')!r}")
    if str(marker.get("training_manifest")) != str(MANIFEST_RELATIVE_PATH):
        raise DatasetPreparationError("readiness marker names a different training manifest")
    if str(marker.get("training_manifest_sha256")) != manifest_sha256:
        raise DatasetPreparationError("readiness marker/manifest SHA-256 mismatch")
    return marker


def _selected_indexes(descriptions: tuple[str, ...], names: tuple[str, ...], group: str) -> list[int]:
    missing = [name for name in names if name not in descriptions]
    if missing:
        raise DatasetPreparationError(
            f"{group} does not provide required selected bands: {missing}; available={list(descriptions)}"
        )
    return [descriptions.index(name) for name in names]


def _dataset_key_hashes(dataset_root: Path) -> dict[str, str]:
    relative_paths = (
        MANIFEST_RELATIVE_PATH,
        CORE_MANIFEST_RELATIVE_PATH,
        SPLIT_COMPONENTS_RELATIVE_PATH,
        READY_MARKER_RELATIVE_PATH,
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = dataset_root / relative
        if not path.is_file():
            raise DatasetPreparationError(f"required provenance file is missing: {path}")
        result[relative.as_posix()] = sha256_file(path)
    return result


def _progress(row_index: int, total_rows: int, sample_id: str) -> None:
    if row_index == 1 or row_index % 100 == 0 or row_index == total_rows:
        print(
            f"audited {row_index:,}/{total_rows:,} active-modality samples "
            f"({sample_id})",
            flush=True,
        )


def build_assets(
    dataset_root: Path,
    output_directory: Path,
    reservoir_capacity: int,
    seed: int,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    """Audit a release split manifest and write bound contract/statistics assets.

    An optional fixed-split manifest may be supplied. The immutable
    release manifest/readiness marker remain fingerprinted as source provenance,
    while normalization and sample counts bind to the selected manifest.
    """

    root = dataset_root.expanduser().resolve(strict=True)
    source_manifest_path = root / MANIFEST_RELATIVE_PATH
    source_manifest_sha256 = sha256_file(source_manifest_path)
    ready_marker = _load_ready_marker(root, source_manifest_sha256)
    active_manifest_path = (
        source_manifest_path
        if manifest_path is None
        else manifest_path.expanduser().resolve(strict=True)
    )
    rows, fieldnames = _read_manifest(root, active_manifest_path)
    manifest_sha256 = sha256_file(active_manifest_path)
    key_hashes = _dataset_key_hashes(root)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in EXPECTED_SPLITS}
    expected_counts = ready_marker.get("split_audit", {}).get("patches_by_split", {})
    if manifest_path is None and expected_counts and {
        split: int(expected_counts.get(split, -1)) for split in EXPECTED_SPLITS
    } != split_counts:
        raise DatasetPreparationError(
            f"manifest split counts {split_counts} disagree with readiness marker {expected_counts}"
        )
    if any(name.startswith("s2_") for name in GROUP_SPECS):
        raise AssertionError("the S1/terrain contract must never register Sentinel-2 groups")
    if not any(name.startswith("s2_") for name in fieldnames):
        raise DatasetPreparationError("expected full-release manifest S2 metadata was not found")

    group_audits = {
        name: RasterGroupAudit(name, spec, reservoir_capacity, seed + 1_000 * index)
        for index, (name, spec) in enumerate(GROUP_SPECS.items())
    }
    continuous_stats: dict[str, list[ValueAccumulator]] | None = None
    qa_stats: dict[str, ValueAccumulator] | None = None
    depth_stats: ValueAccumulator | None = None
    canonical_depth_stats: ValueAccumulator | None = None
    eligible_output_pixels = 0
    canonical_positive_pixels = 0

    for row_index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for group_name, group_spec in GROUP_SPECS.items():
            relative = Path(str(row[group_spec.path_column]))
            if relative.is_absolute() or ".." in relative.parts:
                raise DatasetPreparationError(
                    f"unsafe {group_name} path for {sample_id}: {relative}"
                )
            path = (root / relative).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise DatasetPreparationError(
                    f"{group_name} path escapes dataset root for {sample_id}: {path}"
                ) from exc
            loaded[group_name] = group_audits[group_name].observe(path, sample_id)

        # All active files must share the label grid.  This is also checked by the
        # runtime loader sample by sample, but auditing it once here catches a bad
        # release before an expensive formal training run begins.
        label_audit = group_audits["label"]
        for name, audit in group_audits.items():
            if (audit.height, audit.width, audit.resolution, audit.crs_counts) != (
                label_audit.height,
                label_audit.width,
                label_audit.resolution,
                label_audit.crs_counts,
            ):
                raise DatasetPreparationError(
                    f"grid mismatch between label and {name} at {sample_id}"
                )

        if continuous_stats is None:
            continuous_stats = _new_accumulator_map(group_audits, reservoir_capacity, seed)
            qa_descriptions = group_audits["s1_qa"].descriptions
            if qa_descriptions is None:
                raise DatasetPreparationError("S1 QA descriptions were not observed")
            missing_qa = set(QA_TRANSFORMS).difference(qa_descriptions)
            if missing_qa:
                raise DatasetPreparationError(f"S1 QA is missing required bands: {sorted(missing_qa)}")
            qa_stats = {
                name: ValueAccumulator(reservoir_capacity, seed + 70_000 + index)
                for index, name in enumerate(QA_TRANSFORMS)
            }
            depth_stats = ValueAccumulator(reservoir_capacity, seed + 80_000)
            canonical_depth_stats = ValueAccumulator(reservoir_capacity, seed + 90_000)

        if row["split"] == "train":
            assert continuous_stats is not None
            assert qa_stats is not None
            assert depth_stats is not None
            assert canonical_depth_stats is not None
            for group in CONTINUOUS_GROUPS:
                values, valid = loaded[group]
                for band_index, accumulator in enumerate(continuous_stats[group]):
                    accumulator.update(values[band_index][valid[band_index]])

            qa_values, qa_valid = loaded["s1_qa"]
            qa_descriptions = group_audits["s1_qa"].descriptions
            assert qa_descriptions is not None
            duration = _duration_days(row)
            count_index = qa_descriptions.index("event_observation_count")
            count_values = np.log1p(np.clip(qa_values[count_index], 0.0, None))
            qa_stats["event_observation_count"].update(
                count_values[qa_valid[count_index]]
            )
            day_index = qa_descriptions.index("selected_event_day_offset")
            day_valid = qa_valid[day_index] & (qa_values[day_index] >= 0.0)
            day_values = np.clip(qa_values[day_index] / float(duration), 0.0, 1.0)
            qa_stats["selected_event_day_offset"].update(day_values[day_valid])

            label_values, label_valid = loaded["label"]
            masks_values, _ = loaded["masks"]
            mask_descriptions = group_audits["masks"].descriptions
            if mask_descriptions is None:
                raise DatasetPreparationError("mask descriptions were not observed")
            missing_masks = set(REQUIRED_MASKS).difference(mask_descriptions)
            if missing_masks:
                raise DatasetPreparationError(f"masks are missing required bands: {sorted(missing_masks)}")
            masks = {
                name: masks_values[mask_descriptions.index(name)] > 0
                for name in REQUIRED_MASKS
            }
            declared_positive = masks["valid_depth_mask"]
            if not np.array_equal(label_valid[0], declared_positive):
                raise DatasetPreparationError(
                    f"label validity differs from valid_depth_mask for {sample_id}"
                )
            train_depth_values = label_values[0][declared_positive]
            if train_depth_values.size == 0 or np.any(train_depth_values <= 0.0):
                raise DatasetPreparationError(
                    f"invalid positive depths in train sample {sample_id}"
                )
            depth_stats.update(train_depth_values)

            event_descriptions = group_audits["s1_t2"].descriptions
            if event_descriptions is None:
                raise DatasetPreparationError("event-period S1 descriptions were not observed")
            event_indexes = _selected_indexes(
                event_descriptions, SELECTED_EVENT_BANDS, "s1_t2"
            )
            _, event_valid = loaded["s1_t2"]
            _, terrain_valid = loaded["terrain"]
            event_fraction = event_valid[event_indexes].astype(np.float32).mean(axis=0)
            terrain_is_valid = np.logical_and.reduce(terrain_valid, axis=0)
            output_valid = (
                masks["S1_event_composite_valid_mask"]
                & (event_fraction >= 1.0)
                & masks["DEM_valid_mask"]
                & masks["slope_valid_mask"]
                & terrain_is_valid
            )
            canonical_positive = declared_positive & output_valid
            eligible_output_pixels += int(output_valid.sum())
            canonical_positive_pixels += int(canonical_positive.sum())
            canonical_depth_stats.update(label_values[0][canonical_positive])
        _progress(row_index, len(rows), sample_id)

    if (
        continuous_stats is None
        or qa_stats is None
        or depth_stats is None
        or canonical_depth_stats is None
    ):
        raise DatasetPreparationError("no training rows were available for statistics")
    if canonical_positive_pixels <= 0 or eligible_output_pixels <= 0:
        raise DatasetPreparationError("canonical train supervision has no eligible pixels")
    # The interior quantiles are reservoir estimates, but the two endpoints must
    # be exact.  Runtime stratification and the frozen tail-weight calibration
    # require their bins to cover every observed train depth, including rare
    # extremes that a finite reservoir might not retain.
    depth_quantiles = [
        depth_stats.minimum,
        *depth_stats.reservoir.quantiles((0.25, 0.5, 0.75)),
        depth_stats.maximum,
    ]
    depth_edges = sorted({float(value) for value in depth_quantiles})
    if len(depth_edges) < 2:
        raise DatasetPreparationError("train-depth stratification needs at least two distinct values")

    groups_payload = {
        group: [
            accumulator.summary(str(description), "identity")
            for description, accumulator in zip(group_audits[group].descriptions or (), accumulators)
        ]
        for group, accumulators in continuous_stats.items()
    }
    qa_payload = {
        "s1_qa": [
            qa_stats[name].summary(name, QA_TRANSFORMS[name]) for name in QA_TRANSFORMS
        ]
    }
    train_depth_payload = depth_stats.summary("depth_m", "identity")
    train_depth_payload["stratification_quantiles"] = depth_quantiles
    train_depth_payload["stratification_bin_edges"] = depth_edges
    train_depth_payload["canonical_output_valid_positive_pixels"] = canonical_positive_pixels
    train_depth_payload["canonical_output_valid_depth_summary"] = canonical_depth_stats.summary(
        "depth_m", "identity"
    )
    raw_positive_fraction = canonical_positive_pixels / eligible_output_pixels
    stats_payload: dict[str, Any] = {
        "groups": groups_payload,
        "manifest_sha256": manifest_sha256,
        "positive_prior": {
            "clip_bounds": [0.01, 0.5],
            "eligible_pixels": eligible_output_pixels,
            "method": (
                "observed train valid-depth fraction among canonical S1/terrain "
                "output-valid pixels; conservative proxy, not a true flood prior"
            ),
            "mode": "auto",
            "positive_pixels": canonical_positive_pixels,
            "raw_fraction": raw_positive_fraction,
            "value": float(np.clip(raw_positive_fraction, 0.01, 0.5)),
        },
        "qa_groups": qa_payload,
        "quantile_method": {
            "kind": "deterministic_uniform_reservoir",
            "seed": int(seed),
            "max_samples_per_band": int(reservoir_capacity),
            "exact_fields": ["count", "minimum", "maximum", "mean", "std"],
        },
        "scope": SCOPE,
        "train_depth": train_depth_payload,
    }
    output = output_directory.expanduser().resolve()
    stats_path = output / "train_stats.json"
    contract_path = output / "dataset_contract.json"
    atomic_write_json(stats_path, stats_payload)
    contract_payload = {
        "dataset_root": str(root),
        "key_file_sha256": key_hashes,
        "manifest": (
            {
                "relative_path": MANIFEST_RELATIVE_PATH.as_posix(),
                "sha256": manifest_sha256,
            }
            if manifest_path is None
            else {
                "path": str(active_manifest_path),
                "sha256": manifest_sha256,
                "source_relative_path": MANIFEST_RELATIVE_PATH.as_posix(),
                "source_sha256": source_manifest_sha256,
            }
        ),
        "modality_policy": {
            "active_input_mode": "s1_terrain",
            "active_raster_groups": list(GROUP_SPECS),
            "sentinel2": "manifest metadata only; never registered or read",
        },
        "normalization": {
            "selected": {
                "scope": SCOPE,
                "sha256": sha256_file(stats_path),
            }
        },
        "raster_groups": {
            name: audit.contract_payload() for name, audit in group_audits.items()
        },
        "sample_counts": split_counts,
        "status": "ready",
    }
    atomic_write_json(contract_path, contract_payload)
    print(f"wrote contract: {contract_path}")
    print(f"wrote train statistics: {stats_path}")
    print(f"split counts: {split_counts}")
    print(
        "S2 policy: metadata only; no Sentinel-2 group was registered or opened.",
        flush=True,
    )
    return contract_path, stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=FULL_DATASET_ROOT)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "assets" / "flooddepthnet_s1_terrain",
    )
    parser.add_argument(
        "--reservoir-capacity",
        type=int,
        default=1_000_000,
        help="Uniform train-value samples retained per normalized band.",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional alternate fixed train/val/test manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_assets(
        args.dataset_root,
        args.output_directory,
        args.reservoir_capacity,
        args.seed,
        args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
