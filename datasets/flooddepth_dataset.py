"""Lazy, strict loader for the audited event-aggregated flood-depth subset."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from datasets.contract import DatasetContract, ensure_within
from datasets.band_selection import BandSpec
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, reliability_spec_for_mode
from datasets.supervision_masks import s1_output_valid_mask


class DatasetIntegrityError(RuntimeError):
    """Raised instead of silently training on an inconsistent raster sample."""


class FloodDepthDataset(Dataset[dict[str, Any]]):
    """Return structured modalities, targets, masks, validity, and provenance.

    Label-derived masks are returned for supervision and evaluation.  The
    model-input adapter exposes ``valid_depth_mask`` only when the caller opts
    into the documented known-flood-range benchmark protocol.  Invalid target
    values are filled with zero for tensor safety.
    """

    def __init__(
        self,
        contract_path: str | Path,
        stats_path: str | Path,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        verify_fingerprints: bool = True,
        band_spec: BandSpec | None = None,
        input_spec: ModelInputSpec | None = None,
        minimum_event_band_fraction: float = 1.0,
        s1_qa_names: Sequence[str] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.contract = DatasetContract.load(contract_path)
        if verify_fingerprints:
            self.contract.verify_fingerprints(include_normalization=True)
        self.normalizer = RobustNormalizer(Path(stats_path), self.contract)
        self.split = split
        self.transform = transform
        self.input_spec = input_spec or ModelInputSpec.from_mode()
        self.minimum_event_band_fraction = float(minimum_event_band_fraction)
        if not 0.0 <= self.minimum_event_band_fraction <= 1.0:
            raise ValueError("minimum_event_band_fraction must lie in [0, 1]")
        self.contract.validate_input_groups(self.input_spec.active_groups)
        available_qa_names = tuple(
            str(value)
            for value in self.contract.group("s1_qa")["band_descriptions"]
        )
        self._required_reliability_qa_names = (
            "event_observation_count",
            "selected_event_day_offset",
        )
        if s1_qa_names is None:
            self.s1_qa_names: tuple[str, ...] | None = None
            self._s1_qa_read_names = available_qa_names
        else:
            selected = tuple(str(value) for value in s1_qa_names)
            if len(selected) != len(set(selected)):
                raise ValueError("s1_qa_names must be a unique sequence")
            unsupported = set(selected).difference(
                {"event_observation_count", "selected_event_day_offset"}
            )
            if unsupported:
                raise ValueError(
                    "s1_qa_names may contain only modeled spatial QA channels: "
                    f"{sorted(unsupported)}"
                )
            missing_qa = set(selected).difference(available_qa_names)
            if missing_qa:
                raise ValueError(
                    f"Requested S1 QA channels are absent from the contract: {sorted(missing_qa)}"
                )
            self.s1_qa_names = selected
            # Even when no QA tensor is exposed to the model, these two train-safe
            # source bands are required to construct the single reliability tensor.
            self._s1_qa_read_names = tuple(
                dict.fromkeys((*selected, *self._required_reliability_qa_names))
            )
        self.band_spec = band_spec or BandSpec.resolve(
            self.contract, None, self.input_spec.continuous_groups
        )
        self.reliability_spec = reliability_spec_for_mode(self.input_spec.mode)
        with self.contract.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        expected = int(self.contract.payload["sample_counts"][split])
        if len(self.rows) != expected:
            raise DatasetIntegrityError(
                f"Split {split} count changed: contract={expected}, manifest={len(self.rows)}"
            )
        self.event_ids = [row.get("source_event_id", "") for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: dict[str, str], group: str) -> Path:
        column = str(self.contract.group(group)["path_column"])
        return ensure_within(self.contract.dataset_root / row[column], self.contract.dataset_root)

    def _read(
        self, row: dict[str, str], group: str, indexes: tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        path = self._path(row, group)
        expected = self.contract.group(group)
        with rasterio.open(path) as dataset:
            descriptions = list(dataset.descriptions)
            expected_descriptions = list(expected["band_descriptions"])
            if descriptions[: len(expected_descriptions)] != expected_descriptions:
                raise DatasetIntegrityError(
                    f"Band descriptions changed for {path}: "
                    f"{descriptions[:len(expected_descriptions)]} != {expected_descriptions}"
                )
            rasterio_indexes = (
                [index + 1 for index in indexes]
                if indexes is not None
                else list(range(1, len(expected_descriptions) + 1))
            )
            array = dataset.read(indexes=rasterio_indexes, masked=False)
            masks = dataset.read_masks(indexes=rasterio_indexes) > 0
            masks &= np.isfinite(array)
            if dataset.nodata is not None:
                if np.isnan(dataset.nodata):
                    masks &= ~np.isnan(array)
                else:
                    masks &= array != dataset.nodata
            metadata = {
                "path": str(path),
                "width": dataset.width,
                "height": dataset.height,
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "transform": tuple(float(value) for value in dataset.transform),
                "resolution": tuple(float(value) for value in dataset.res),
                "nodata": dataset.nodata,
                "grid": (
                    dataset.width,
                    dataset.height,
                    dataset.crs.to_wkt() if dataset.crs else None,
                    *tuple(float(value) for value in dataset.transform),
                ),
            }
        return array, masks, metadata

    @staticmethod
    def _duration(row: dict[str, str]) -> int:
        return max(
            1,
            (date.fromisoformat(row["event_end"]) - date.fromisoformat(row["event_start"])).days
            + 1,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays: dict[str, np.ndarray] = {}
        validity: dict[str, np.ndarray] = {}
        metadata_by_group: dict[str, dict[str, Any]] = {}
        read_band_counts: dict[str, int] = {}
        reference_grid: tuple[Any, ...] | None = None
        for group in self.input_spec.active_groups:
            selected_indexes = (
                self.band_spec.read_indexes(self.contract, group)
                if group in self.input_spec.continuous_groups and group != "terrain"
                else None
            )
            if group == "s1_qa" and self.s1_qa_names is not None:
                contract_names = tuple(
                    str(value)
                    for value in self.contract.group("s1_qa")["band_descriptions"]
                )
                selected_indexes = tuple(
                    contract_names.index(name) for name in self._s1_qa_read_names
                )
            array, valid, metadata = self._read(row, group, selected_indexes)
            read_band_counts[group] = int(array.shape[0])
            if reference_grid is None:
                reference_grid = metadata["grid"]
            elif metadata["grid"] != reference_grid:
                raise DatasetIntegrityError(
                    f"Sample {row['sample_id']} has a grid mismatch in {group}"
                )
            arrays[group] = array
            validity[group] = valid
            metadata_by_group[group] = metadata

        mask_names = list(self.contract.group("masks")["band_descriptions"])
        masks_np = {
            name: (arrays["masks"][band_index] > 0)[None]
            for band_index, name in enumerate(mask_names)
        }
        label_raster_valid = validity["label"][0]
        positive = masks_np["valid_depth_mask"][0]
        if not np.array_equal(label_raster_valid, positive):
            raise DatasetIntegrityError(
                f"Label validity differs from valid_depth_mask for {row['sample_id']}"
            )
        label = np.where(positive, arrays["label"][0], 0.0).astype(np.float32)[None]

        continuous: dict[str, np.ndarray] = {}
        branch_valid_fractions: dict[str, np.ndarray] = {}
        for group in self.input_spec.continuous_groups:
            descriptions = list(self.band_spec.names(group))
            read_indexes = self.band_spec.read_indexes(self.contract, group)
            positions = [read_indexes.index(index) for index in self.band_spec.indexes(group)]
            selected_array = arrays[group][positions]
            selected_validity = validity[group][positions]
            branch_valid_fractions[group] = selected_validity.astype(np.float32).mean(
                axis=0, keepdims=True
            )
            continuous[group] = self.normalizer.continuous(
                group, descriptions, selected_array, selected_validity
            )
        terrain_raw = np.where(validity["terrain"], arrays["terrain"], 0.0).astype(np.float32)
        duration = self._duration(row)
        s1_read_descriptions = list(self._s1_qa_read_names)
        s1_model_descriptions = list(
            self.s1_qa_names
            if self.s1_qa_names is not None
            else self.contract.group("s1_qa")["band_descriptions"]
        )
        s1_qa = np.zeros(
            (len(s1_model_descriptions), *arrays["s1_qa"].shape[-2:]),
            dtype=np.float32,
        )

        def qa(group: str, descriptions: list[str], name: str, day: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            position = descriptions.index(name)
            raw = arrays[group][position].astype(np.float32)
            valid = validity[group][position].copy()
            missing = (~valid) | ((raw < 0) if day else False)
            if day:
                transformed = np.clip(raw / float(duration), 0.0, 1.0)
                valid &= raw >= 0
                raw_normalized = np.where(valid, transformed, 0.0).astype(np.float32)
            else:
                transformed = np.log1p(np.clip(raw, 0.0, None))
                raw_normalized = np.where(valid, transformed, 0.0).astype(np.float32)
            normalized = self.normalizer.qa_feature(group, name, transformed, valid)
            return normalized, raw_normalized, missing.astype(np.float32)

        if (
            "event_observation_count" not in s1_read_descriptions
            or "selected_event_day_offset" not in s1_read_descriptions
        ):
            raise DatasetIntegrityError(
                "S1 depth inputs require event_observation_count and "
                "selected_event_day_offset QA channels"
            )
        s1_obs, _, _ = qa("s1_qa", s1_read_descriptions, "event_observation_count")
        s1_day, s1_day_raw, s1_missing = qa(
            "s1_qa", s1_read_descriptions, "selected_event_day_offset", day=True
        )
        if "event_observation_count" in s1_model_descriptions:
            s1_qa[s1_model_descriptions.index("event_observation_count")] = s1_obs
        if "selected_event_day_offset" in s1_model_descriptions:
            s1_qa[s1_model_descriptions.index("selected_event_day_offset")] = s1_day

        # Semantic validity and GeoTIFF/per-band validity must both hold. This avoids
        # treating a tensor-safety zero inserted at raster nodata as a real observation.
        terrain_raster_valid = np.logical_and.reduce(validity["terrain"], axis=0)
        # Modality availability is a stable semantic mask from the contract.  It
        # deliberately does not change when a model band is removed; branch-level
        # fractions above carry the selected raster validity into each encoder.
        s1_valid = (
            masks_np["S1_event_composite_valid_mask"][0]
        ).astype(np.float32)
        dem_valid = (
            masks_np["DEM_valid_mask"][0]
            & masks_np["slope_valid_mask"][0]
            & terrain_raster_valid
        ).astype(np.float32)
        duration_feature = np.full_like(
            s1_valid, np.clip(np.log1p(duration) / np.log(32.0), 0.0, 2.0), dtype=np.float32
        )
        # The event-band fraction is calculated after BandSpec selection, so it is
        # a true fraction of the model's selected VV/VH event channels and is
        # invariant to their ordering.  S1-only output support is intentionally
        # stricter than semantic availability alone.
        s1_event_fraction = branch_valid_fractions["s1_t2"]
        s1_event_support_tensor, output_valid_tensor = s1_output_valid_mask(
            torch.from_numpy(s1_valid[None]),
            torch.from_numpy(s1_event_fraction),
            torch.from_numpy(dem_valid[None]),
            self.minimum_event_band_fraction,
        )
        s1_event_support = s1_event_support_tensor.numpy()[0].astype(np.float32)
        reliability = np.stack(
            [s1_obs, s1_day, s1_valid, dem_valid, duration_feature, s1_missing], axis=0
        ).astype(np.float32)
        output_valid = output_valid_tensor.numpy()[0].astype(np.float32)

        exposed_mask_names = mask_names
        sample: dict[str, Any] = {
            group: torch.from_numpy(continuous[group])
            for group in self.input_spec.continuous_groups
        }
        sample.update({
            "s1_qa": torch.from_numpy(s1_qa),
            "terrain_raw": torch.from_numpy(terrain_raw),
            "label": torch.from_numpy(label),
            "masks": {
                name: torch.from_numpy(masks_np[name].astype(np.float32))
                for name in exposed_mask_names
            },
            "validity": {
                "s1_valid": torch.from_numpy(s1_valid[None]),
                "dem_valid": torch.from_numpy(dem_valid[None]),
                # Explicit availability aliases make the semantic/physical
                # validity contract unambiguous to augmentation and diagnostics.
                "s1_available": torch.from_numpy(s1_valid[None]),
                "dem_available": torch.from_numpy(dem_valid[None]),
                "s1_event_support": torch.from_numpy(s1_event_support[None]),
                "output_valid": torch.from_numpy(output_valid[None]),
                "s1_t1_valid_fraction": torch.from_numpy(branch_valid_fractions["s1_t1"]),
                "s1_t2_valid_fraction": torch.from_numpy(branch_valid_fractions["s1_t2"]),
                "s1_event_band_fraction": torch.from_numpy(s1_event_fraction),
                "s1_change_valid_fraction": torch.from_numpy(branch_valid_fractions["s1_change"]),
                "terrain_valid_fraction": torch.from_numpy(
                    branch_valid_fractions["terrain"]
                ),
                "dem_valid_fraction": torch.from_numpy(
                    validity["terrain"].astype(np.float32).mean(axis=0, keepdims=True)
                ),
            },
            "reliability": torch.from_numpy(reliability),
            "reliability_names": self.reliability_spec.names,
            "metadata": {
                "sample_id": row["sample_id"],
                "source_event_id": row["source_event_id"],
                "event_chain_id": row["event_chain_id"],
                "event_start": row["event_start"],
                "event_end": row["event_end"],
                "event_duration_days": duration,
                "sample_origin": row.get("sample_origin", ""),
                "split": self.split,
                "crs": metadata_by_group["label"]["crs"],
                "transform": metadata_by_group["label"]["transform"],
                "resolution": metadata_by_group["label"]["resolution"],
                "width": metadata_by_group["label"]["width"],
                "height": metadata_by_group["label"]["height"],
                "label_path": metadata_by_group["label"]["path"],
                "reliability_names": self.reliability_spec.names,
                "model_input_spec": self.input_spec.as_dict(),
                "resolved_model_bands": self.band_spec.as_dict(),
                "model_s1_qa_names": tuple(s1_model_descriptions),
                "io_profile": {
                    "opened_files": len(metadata_by_group),
                    "read_bands": int(sum(read_band_counts.values())),
                    "read_band_counts": dict(read_band_counts),
                },
                "branch_valid_fractions": {
                    key: value.mean().item() for key, value in branch_valid_fractions.items()
                },
                "minimum_event_band_fraction": self.minimum_event_band_fraction,
            },
        })
        sample["validity"]["s1_available"] = torch.from_numpy(s1_valid[None])
        sample["validity"]["dem_available"] = torch.from_numpy(dem_valid[None])
        sample["validity"]["s1_valid"] = torch.from_numpy(s1_valid[None])
        return self.transform(sample) if self.transform is not None else sample


MODEL_INPUT_KEYS = (
    "s1_t1",
    "s1_t2",
    "s1_change",
    "terrain",
    "terrain_raw",
)


def prepare_model_inputs(
    batch: dict[str, Any],
    input_spec: ModelInputSpec | None = None,
    *,
    include_valid_depth_mask: bool = False,
    include_reliability: bool = True,
) -> dict[str, Any]:
    """Whitelist inputs before calling ``model.forward``.

    ``valid_depth_mask`` is label-derived and remains excluded by default. It
    is admitted only for the explicit known-range benchmark used by every
    comparison model in this repository. ``reliability`` is independently
    optional so a physical ``w/o RCP`` model never receives that tensor.
    """

    spec = input_spec or ModelInputSpec.from_mode()
    required = (*MODEL_INPUT_KEYS, *(("reliability",) if include_reliability else ()))
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing model inputs: {missing}")
    validity = batch.get("validity")
    if not isinstance(validity, dict):
        raise KeyError("Batch has no validity mapping")
    result = {key: batch[key] for key in MODEL_INPUT_KEYS}
    if include_reliability:
        result["reliability"] = batch["reliability"]
    result.update({
        "s1_valid": validity["s1_valid"],
        "s1_event_support": validity["s1_event_support"],
        "dem_valid": validity["dem_valid"],
        "branch_validity": {
            group: validity[f"{group}_valid_fraction"]
            for group in spec.continuous_groups
        },
    })
    if include_reliability:
        result["reliability_names"] = reliability_spec_for_mode(spec.mode).names
    if include_valid_depth_mask:
        masks = batch.get("masks")
        if not isinstance(masks, dict) or "valid_depth_mask" not in masks:
            raise KeyError(
                "Known-range inference requires masks.valid_depth_mask"
            )
        result["valid_depth_mask"] = masks["valid_depth_mask"]
    return result
