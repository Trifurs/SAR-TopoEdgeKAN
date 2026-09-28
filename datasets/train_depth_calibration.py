"""Train-only canonical depth calibration for SAR-and-terrain inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio

from datasets.band_selection import BandSpec
from datasets.contract import DatasetContract, ensure_within


class TrainDepthCalibrationError(RuntimeError):
    """Raised when train-only depth calibration cannot preserve the data contract."""


@dataclass(frozen=True)
class TrainDepthScan:
    """Train-only canonical positive targets and a compact provenance summary."""

    depths_m: np.ndarray
    sample_count: int
    canonical_positive_pixels: int

    def summary(self) -> dict[str, Any]:
        if self.depths_m.size == 0:
            raise TrainDepthCalibrationError("no canonical positive train pixels were collected")
        values = self.depths_m.astype(np.float64, copy=False)
        return {
            "split": "train",
            "supervision_mask": "valid_depth_mask_and_output_valid",
            "sample_count": self.sample_count,
            "canonical_positive_pixels": self.canonical_positive_pixels,
            "depth_min_m": float(values.min()),
            "depth_max_m": float(values.max()),
            "depth_mean_m": float(values.mean()),
            "depth_median_m": float(np.median(values)),
        }


def _manifest_rows(contract: DatasetContract) -> list[dict[str, str]]:
    with contract.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
    expected = int(contract.payload["sample_counts"]["train"])
    if len(rows) != expected:
        raise TrainDepthCalibrationError(
            f"train manifest count changed: contract={expected}, manifest={len(rows)}"
        )
    return rows


def _path(contract: DatasetContract, row: Mapping[str, str], group: str) -> Path:
    column = str(contract.group(group)["path_column"])
    return ensure_within(contract.dataset_root / str(row[column]), contract.dataset_root)


def _read_data_and_valid(
    path: Path,
    indexes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as dataset:
        values = dataset.read(indexes=indexes, masked=False)
        valid = dataset.read_masks(indexes=indexes) > 0
        valid &= np.isfinite(values)
        if dataset.nodata is not None:
            if np.isnan(dataset.nodata):
                valid &= ~np.isnan(values)
            else:
                valid &= values != dataset.nodata
    return values, valid


def _read_masks(path: Path, indexes: list[int]) -> np.ndarray:
    values, _ = _read_data_and_valid(path, indexes)
    return values > 0


def collect_canonical_train_depths(
    contract: DatasetContract,
    band_spec: BandSpec,
    *,
    minimum_event_band_fraction: float,
) -> TrainDepthScan:
    """Collect train targets under the same S1-only canonical output mask.

    This intentionally opens only labels, masks, selected event-period SAR bands,
    and terrain.
    """

    if not 0.0 <= float(minimum_event_band_fraction) <= 1.0:
        raise TrainDepthCalibrationError("minimum_event_band_fraction must lie in [0, 1]")
    rows = _manifest_rows(contract)
    mask_names = list(contract.group("masks")["band_descriptions"])
    required_mask_names = (
        "valid_depth_mask",
        "DEM_valid_mask",
        "slope_valid_mask",
        "S1_event_composite_valid_mask",
    )
    missing_masks = set(required_mask_names).difference(mask_names)
    if missing_masks:
        raise TrainDepthCalibrationError(f"contract masks are missing {sorted(missing_masks)}")
    mask_indexes = {name: mask_names.index(name) + 1 for name in required_mask_names}
    s1_indexes = [index + 1 for index in band_spec.indexes("s1_t2")]
    if not s1_indexes:
        raise TrainDepthCalibrationError("S1 output validity requires selected event-period bands")
    chunks: list[np.ndarray] = []
    for row in rows:
        label, label_valid = _read_data_and_valid(_path(contract, row, "label"), [1])
        mask_values = _read_masks(
            _path(contract, row, "masks"), list(mask_indexes.values())
        )
        mask_by_name = {
            name: mask_values[index]
            for index, name in enumerate(mask_indexes)
        }
        _, event_valid_bands = _read_data_and_valid(
            _path(contract, row, "s1_t2"), s1_indexes
        )
        _, terrain_valid_bands = _read_data_and_valid(
            _path(contract, row, "terrain"), None
        )
        label_valid_2d = label_valid[0]
        declared_positive = mask_by_name["valid_depth_mask"]
        if not np.array_equal(label_valid_2d, declared_positive):
            raise TrainDepthCalibrationError(
                f"label validity differs from valid_depth_mask for {row['sample_id']}"
            )
        event_fraction = event_valid_bands.astype(np.float32).mean(axis=0)
        event_support = (
            mask_by_name["S1_event_composite_valid_mask"]
            & (event_fraction >= float(minimum_event_band_fraction))
        )
        terrain_valid = np.logical_and.reduce(terrain_valid_bands, axis=0)
        output_valid = (
            event_support
            & mask_by_name["DEM_valid_mask"]
            & mask_by_name["slope_valid_mask"]
            & terrain_valid
        )
        selected = declared_positive & output_valid
        values = label[0][selected]
        if values.size:
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise TrainDepthCalibrationError(
                    f"canonical positive depths are invalid for {row['sample_id']}"
                )
            chunks.append(values.astype(np.float32, copy=False))
    if not chunks:
        raise TrainDepthCalibrationError("train split has no canonical positive depths")
    depths = np.concatenate(chunks)
    return TrainDepthScan(
        depths_m=depths,
        sample_count=len(rows),
        canonical_positive_pixels=int(depths.size),
    )
