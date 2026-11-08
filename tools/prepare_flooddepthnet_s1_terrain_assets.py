#!/usr/bin/env python3
"""Verify the published FloodDepthNet package and bind its frozen assets.

This command audits the extracted ``FloodDepthNet.zip`` files. It preserves the
published fixed split and copies the supplied training statistics byte-for-byte;
it does not estimate normalization values or create a new split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from utils.misc import atomic_write_json


MANIFEST_RELATIVE_PATH = "metadata/training_manifest.csv"
INFO_RELATIVE_PATH = "metadata/dataset_info.json"
STATS_RELATIVE_PATH = "metadata/train_stats.json"
SCOPE = "train split only; valid pixels only; val/test excluded"
EXPECTED_SAMPLE_COUNTS = {"train": 4084, "val": 584, "test": 1167}
RELEASE_METADATA_SHA256 = {
    MANIFEST_RELATIVE_PATH: "aa999968cc259296ba86fec0d7c39f120e0cead4ae587d907ddd1178d2a32601",
    INFO_RELATIVE_PATH: "7423fd95fbe3bf29b073671a4dac9bf9645aa3a27ebbedac83809348292d8ff2",
    STATS_RELATIVE_PATH: "b63cf9c5e01a83e680eb0c358b260632aa6f7e84ebb5a045f5159a6b54c2e2db",
    "SHA256SUMS": "2e5a9ed7ffc578ae751e006cf9b40eeaa0e5ca6e775b07cafad950ba65dbc000",
}


@dataclass(frozen=True)
class GroupSpec:
    path_column: str
    directory: str
    role: str
    bands: tuple[str, ...]


GROUP_SPECS = {
    "label": GroupSpec("label_path", "label", "target", ("depth_m",)),
    "masks": GroupSpec("masks_path", "masks", "mask_only", (
        "valid_depth_mask", "DEM_valid_mask", "slope_valid_mask",
        "S1_event_composite_valid_mask",
    )),
    "s1_t1": GroupSpec("s1_t1_path", "S1/T1", "model_input", ("VV_pre_db", "VH_pre_db")),
    "s1_t2": GroupSpec("s1_t2_path", "S1/T2", "model_input", ("VV_event_db", "VH_event_db")),
    "s1_change": GroupSpec("s1_change_path", "S1/change", "model_input", (
        "VV_delta_db", "VH_delta_db", "anomaly_raw",
    )),
    "s1_qa": GroupSpec("s1_qa_path", "S1/QA", "quality_only", (
        "event_observation_count", "selected_event_day_offset",
    )),
    "terrain": GroupSpec("dem_path", "DEM", "model_input", ("elevation_m_DSM", "slope_deg")),
}
CONTINUOUS_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain")
QA_TRANSFORMS = {
    "event_observation_count": "log1p(max(x,0))",
    "selected_event_day_offset": "clip(x/event_duration_days,0,1); x<0 is missing",
}


class DatasetPreparationError(RuntimeError):
    """The extracted files differ from the published experiment dataset."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts
            or "\\" in relative or path.as_posix() != relative):
        raise DatasetPreparationError(f"Unsafe relative dataset path: {relative!r}")
    try:
        source = (root / relative).resolve(strict=True)
        source.relative_to(root)
    except (OSError, ValueError) as exc:
        raise DatasetPreparationError(f"Missing or escaping dataset file: {relative}") from exc
    if not source.is_file():
        raise DatasetPreparationError(f"Not a regular dataset file: {relative}")
    return source


def _verified_bytes(root: Path, relative: str, expected: str) -> bytes:
    data = _source(root, relative).read_bytes()
    if _sha256(data) != expected:
        raise DatasetPreparationError(f"SHA-256 mismatch: {relative}")
    return data


def _checksums(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise DatasetPreparationError("Malformed SHA256SUMS entry")
        digest, relative = match.groups()
        if relative in result:
            raise DatasetPreparationError(f"Duplicate checksum path: {relative}")
        result[relative] = digest
    return result


def _metadata(root: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, list[dict[str, str]], dict[str, str]]:
    verified = {
        relative: _verified_bytes(root, relative, expected)
        for relative, expected in RELEASE_METADATA_SHA256.items()
    }
    sums = _checksums(verified["SHA256SUMS"])
    for relative in (MANIFEST_RELATIVE_PATH, INFO_RELATIVE_PATH, STATS_RELATIVE_PATH):
        if sums.get(relative) != RELEASE_METADATA_SHA256[relative]:
            raise DatasetPreparationError(f"Release checksum index disagrees with {relative}")
    try:
        info = json.loads(verified[INFO_RELATIVE_PATH])
        stats = json.loads(verified[STATS_RELATIVE_PATH])
        reader = csv.DictReader(io.StringIO(verified[MANIFEST_RELATIVE_PATH].decode("utf-8-sig")))
        rows = list(reader)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DatasetPreparationError("Invalid published metadata encoding") from exc
    fields = set(reader.fieldnames or ())
    required = {"sample_id", "filename", "split", "source_event_id", "event_chain_id", "event_start", "event_end"}
    for spec in GROUP_SPECS.values():
        required.update((spec.path_column, spec.path_column.removesuffix("_path") + "_sha256"))
    if not required.issubset(fields):
        raise DatasetPreparationError(f"Manifest missing fields: {sorted(required - fields)}")
    if info.get("name") != "FloodDepthNet" or info.get("sample_counts") != EXPECTED_SAMPLE_COUNTS:
        raise DatasetPreparationError("Published name or fixed sample counts changed")
    if (info.get("shape_pixels") != [256, 256] or info.get("pixel_size_m") != [20, 20]
            or info.get("crs") != "EPSG:27704"):
        raise DatasetPreparationError("Published raster grid definition changed")
    if info.get("manifest") != MANIFEST_RELATIVE_PATH:
        raise DatasetPreparationError("Dataset information names a different manifest")
    if info.get("manifest_sha256") != RELEASE_METADATA_SHA256[MANIFEST_RELATIVE_PATH]:
        raise DatasetPreparationError("Dataset information/manifest fingerprint mismatch")
    groups = info.get("raster_groups", {})
    if set(groups) != set(GROUP_SPECS):
        raise DatasetPreparationError("Published raster groups changed")
    for name, spec in GROUP_SPECS.items():
        entry = groups[name]
        expected_dtype = "uint8" if name == "masks" else "float32"
        expected_nodata = None if name == "masks" else -9999.0
        if (entry.get("path_column") != spec.path_column
                or entry.get("bands") != list(spec.bands)
                or entry.get("dtype") != expected_dtype
                or entry.get("nodata") != expected_nodata):
            raise DatasetPreparationError(f"Published band schema changed: {name}")
    _validate_stats(stats)
    if stats.get("manifest_sha256") != RELEASE_METADATA_SHA256[MANIFEST_RELATIVE_PATH]:
        raise DatasetPreparationError("Frozen statistics/manifest fingerprint mismatch")
    return info, stats, verified[STATS_RELATIVE_PATH], rows, sums


def _validate_stats(stats: dict[str, Any]) -> None:
    if stats.get("scope") != SCOPE:
        raise DatasetPreparationError("Normalization statistics do not have training-only scope")
    if set(stats.get("groups", {})) != set(CONTINUOUS_GROUPS):
        raise DatasetPreparationError("Frozen continuous-statistics groups changed")
    if set(stats.get("qa_groups", {})) != {"s1_qa"}:
        raise DatasetPreparationError("Frozen quality-statistics groups changed")
    for group in (*CONTINUOUS_GROUPS, "s1_qa"):
        entries = stats["qa_groups" if group == "s1_qa" else "groups"][group]
        if [entry.get("band") for entry in entries] != list(GROUP_SPECS[group].bands):
            raise DatasetPreparationError(f"Frozen statistics contain different bands: {group}")
        for entry in entries:
            values = [float(entry.get(key, float("nan"))) for key in ("mean", "std", "p0.5", "p99.5")]
            if not all(math.isfinite(value) for value in values) or values[1] <= 0 or values[2] > values[3]:
                raise DatasetPreparationError(f"Invalid frozen statistics: {group}/{entry['band']}")
            transform = QA_TRANSFORMS[entry["band"]] if group == "s1_qa" else "identity"
            if entry.get("transform") != transform:
                raise DatasetPreparationError(f"Frozen normalization transform changed: {group}/{entry['band']}")


def _manifest_files(rows: list[dict[str, str]], info: dict[str, Any]) -> dict[str, tuple[str, str]]:
    if Counter(row["split"] for row in rows) != EXPECTED_SAMPLE_COUNTS:
        raise DatasetPreparationError("Manifest does not preserve the fixed split counts")
    ids: set[str] = set()
    events: dict[str, set[str]] = defaultdict(set)
    files: dict[str, tuple[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in ids or row["filename"] != sample_id + ".tif":
            raise DatasetPreparationError(f"Missing or duplicate sample identity: {sample_id!r}")
        ids.add(sample_id)
        if not row["source_event_id"]:
            raise DatasetPreparationError(f"Missing source event: {sample_id}")
        events[row["source_event_id"]].add(row["split"])
        try:
            if date.fromisoformat(row["event_end"]) < date.fromisoformat(row["event_start"]):
                raise ValueError("reversed event dates")
        except ValueError as exc:
            raise DatasetPreparationError(f"Invalid event period: {sample_id}") from exc
        for group, spec in GROUP_SPECS.items():
            relative = row[spec.path_column]
            expected = f"{row['split']}/{spec.directory}/{row['filename']}"
            if relative != expected:
                raise DatasetPreparationError(f"Raster path does not match sample/split: {relative}")
            if relative in files:
                raise DatasetPreparationError(f"Duplicate raster path: {relative}")
            files[relative] = (group, row[spec.path_column.removesuffix("_path") + "_sha256"])
    if any(len(splits) != 1 for splits in events.values()):
        raise DatasetPreparationError("Source events overlap between fixed splits")
    observed_events = {split: len({row['source_event_id'] for row in rows if row['split'] == split}) for split in EXPECTED_SAMPLE_COUNTS}
    if info.get("source_event_counts") != observed_events:
        raise DatasetPreparationError("Source-event counts disagree with dataset information")
    return files


def _audit_raster(data: bytes, relative: str, group: str) -> tuple[tuple[Any, ...], np.ndarray | None]:
    spec = GROUP_SPECS[group]
    expected_dtype = "uint8" if group == "masks" else "float32"
    expected_nodata = None if group == "masks" else -9999.0
    try:
        with MemoryFile(data) as memory, memory.open() as raster:
            if (raster.count != len(spec.bands) or raster.descriptions != spec.bands
                    or raster.dtypes != (expected_dtype,) * len(spec.bands)
                    or raster.nodata != expected_nodata):
                raise DatasetPreparationError(f"Raster bands or dtype/NoData changed: {relative}")
            if (raster.width != 256 or raster.height != 256
                    or raster.crs != rasterio.crs.CRS.from_epsg(27704)
                    or raster.res != (20.0, 20.0)):
                raise DatasetPreparationError(f"Raster grid changed: {relative}")
            grid = (raster.width, raster.height, raster.crs, *tuple(raster.transform))
            values = raster.read()
            if group == "masks":
                if not np.all((values == 0) | (values == 1)):
                    raise DatasetPreparationError(f"Non-binary validity mask: {relative}")
                return grid, values[0] > 0
            if not np.isfinite(values).all():
                raise DatasetPreparationError(f"Non-finite source pixels: {relative}")
            if group == "label":
                valid = (raster.read_masks(1) > 0) & (values[0] != raster.nodata)
                if np.any(values[0][valid] <= 0):
                    raise DatasetPreparationError(f"Non-positive valid reference depth: {relative}")
                return grid, valid
            return grid, None
    except rasterio.errors.RasterioError as exc:
        raise DatasetPreparationError(f"Unreadable GeoTIFF: {relative}") from exc


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_assets(dataset_root: Path, output_directory: Path) -> tuple[Path, Path]:
    """Audit every distributed file, then bind the unchanged frozen statistics."""
    root = dataset_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DatasetPreparationError("--dataset-root must be the extracted FloodDepthNet directory")
    output = output_directory.expanduser().resolve()
    if output == root or root in output.parents:
        raise DatasetPreparationError("Asset output must be outside the immutable dataset directory")
    info, _stats, frozen_stats, rows, sums = _metadata(root)
    files = _manifest_files(rows, info)
    metadata_files = {"README.md", MANIFEST_RELATIVE_PATH, INFO_RELATIVE_PATH, STATS_RELATIVE_PATH}
    if set(sums) != set(files) | metadata_files:
        raise DatasetPreparationError("Checksum index does not describe exactly the distributed files")
    for relative in metadata_files:
        _verified_bytes(root, relative, sums[relative])
    for index, row in enumerate(rows, start=1):
        reference_grid = None
        supports = {}
        for group, spec in GROUP_SPECS.items():
            relative = row[spec.path_column]
            expected_hash = files[relative][1]
            if sums[relative] != expected_hash:
                raise DatasetPreparationError(f"Manifest/checksum-index mismatch: {relative}")
            data = _verified_bytes(root, relative, expected_hash)
            grid, support = _audit_raster(data, relative, group)
            if reference_grid is None:
                reference_grid = grid
            if grid != reference_grid:
                raise DatasetPreparationError(f"Sample raster alignment changed: {relative}")
            if support is not None:
                supports[group] = support
        if not np.array_equal(supports["label"], supports["masks"]):
            raise DatasetPreparationError(f"Reference label/valid_depth_mask mismatch: {row['sample_id']}")
        if index == 1 or index % 200 == 0 or index == len(rows):
            print(f"verified {index:,}/{len(rows):,} samples", flush=True)
    # The whole verification finishes before touching previously valid assets.
    source_hashes = dict(RELEASE_METADATA_SHA256)
    contract = {
        "status": "ready", "dataset_name": "FloodDepthNet",
        "dataset_root": Path(os.path.relpath(root, output)).as_posix(),
        "manifest": {"relative_path": MANIFEST_RELATIVE_PATH, "sha256": sums[MANIFEST_RELATIVE_PATH]},
        "key_file_sha256": source_hashes,
        "modality_policy": {"active_input_mode": "s1_terrain", "active_raster_groups": list(GROUP_SPECS)},
        "normalization": {"selected": {"scope": SCOPE, "sha256": _sha256(frozen_stats)}},
        "raster_groups": {
            group: {"path_column": spec.path_column, "role": spec.role,
                    "band_count": len(spec.bands), "band_descriptions": list(spec.bands),
                    "dtypes": ["uint8" if group == "masks" else "float32"] * len(spec.bands),
                    "nodata": None if group == "masks" else -9999.0,
                    "width": 256, "height": 256, "crs": "EPSG:27704", "resolution": [20.0, 20.0],
                    "observed_crs_counts": {"EPSG:27704": len(rows)}}
            for group, spec in GROUP_SPECS.items()
        },
        "sample_counts": dict(EXPECTED_SAMPLE_COUNTS),
    }
    stats_path, contract_path = output / "train_stats.json", output / "dataset_contract.json"
    _write_bytes(stats_path, frozen_stats)
    atomic_write_json(contract_path, contract)
    print(f"Bound {len(rows):,} fixed-split samples; frozen training statistics preserved.", flush=True)
    return contract_path, stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "FloodDepthNet",
                        help="Extracted published dataset directory.")
    parser.add_argument("--output-directory", type=Path,
                        default=PROJECT_ROOT / "assets" / "flooddepthnet_s1_terrain")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_assets(args.dataset_root, args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
