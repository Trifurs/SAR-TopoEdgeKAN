from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tools import prepare_flooddepthnet_s1_terrain_assets as prepare


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _manifest(root: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (root / prepare.MANIFEST_RELATIVE_PATH).write_text(buffer.getvalue(), encoding="utf-8")


@pytest.fixture
def published_package(tmp_path, monkeypatch):
    """Three real GeoTIFF samples exercise the release binder without large data."""
    root = tmp_path / "FloodDepthNet"
    (root / "metadata").mkdir(parents=True)
    (root / "README.md").write_text("# FloodDepthNet\n", encoding="utf-8")
    counts = {"train": 1, "val": 1, "test": 1}
    monkeypatch.setattr(prepare, "EXPECTED_SAMPLE_COUNTS", counts)
    rows = []
    for split in counts:
        row = {"sample_id": f"sample_{split}", "filename": f"sample_{split}.tif",
               "split": split, "source_event_id": f"event_{split}",
               "event_chain_id": "shared_chain", "event_start": "2020-01-01", "event_end": "2020-01-03"}
        for group, spec in prepare.GROUP_SPECS.items():
            relative = f"{split}/{spec.directory}/{row['filename']}"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            dtype = "uint8" if group == "masks" else "float32"
            values = np.full((len(spec.bands), 256, 256), 1 if group == "masks" else 2, dtype=dtype)
            if group == "label":
                values[0, 0, 0] = -9999
            elif group == "masks":
                values[0, 0, 0] = 0
            with rasterio.open(path, "w", driver="GTiff", width=256, height=256,
                               count=len(spec.bands), dtype=dtype,
                               nodata=None if group == "masks" else -9999,
                               crs="EPSG:27704", transform=from_origin(1000, 1000, 20, 20),
                               compress="deflate", photometric="MINISBLACK") as target:
                target.write(values)
                target.descriptions = spec.bands
            row[spec.path_column] = relative
            row[spec.path_column.removesuffix("_path") + "_sha256"] = _digest(path)
        rows.append(row)
    info = {
        "name": "FloodDepthNet", "sample_counts": counts, "source_event_counts": counts,
        "shape_pixels": [256, 256], "pixel_size_m": [20, 20], "crs": "EPSG:27704",
        "manifest": prepare.MANIFEST_RELATIVE_PATH,
        "raster_groups": {
            name: {"path_column": spec.path_column, "bands": list(spec.bands),
                   "dtype": "uint8" if name == "masks" else "float32",
                   "nodata": None if name == "masks" else -9999.0}
            for name, spec in prepare.GROUP_SPECS.items()
        },
    }
    def entry(band, transform="identity"):
        return {"band": band, "mean": 1.0, "std": 2.0, "p0.5": 0.0, "p99.5": 100.0,
                "transform": transform, "count": 123}
    stats = {
        "scope": prepare.SCOPE,
        "groups": {group: [entry(band) for band in prepare.GROUP_SPECS[group].bands]
                   for group in prepare.CONTINUOUS_GROUPS},
        "qa_groups": {"s1_qa": [entry(band, prepare.QA_TRANSFORMS[band])
                                  for band in prepare.GROUP_SPECS['s1_qa'].bands]},
        "train_depth": {"minimum": 0.1, "maximum": 4.0, "stratification_bin_edges": [0.1, 2.0, 4.0]},
        "positive_prior": {"value": 0.1},
    }
    def resign():
        for row in rows:
            for spec in prepare.GROUP_SPECS.values():
                row[spec.path_column.removesuffix("_path") + "_sha256"] = _digest(root / row[spec.path_column])
        _manifest(root, rows)
        manifest_sha = _digest(root / prepare.MANIFEST_RELATIVE_PATH)
        info["manifest_sha256"] = stats["manifest_sha256"] = manifest_sha
        _json(root / prepare.INFO_RELATIVE_PATH, info)
        _json(root / prepare.STATS_RELATIVE_PATH, stats)
        paths = [path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
        text = "".join(f"{_digest(path)}  {path.relative_to(root).as_posix()}\n"
                       for path in sorted(paths))
        (root / "SHA256SUMS").write_text(text, encoding="utf-8")
        monkeypatch.setattr(prepare, "RELEASE_METADATA_SHA256", {
            relative: _digest(root / relative) for relative in (
                prepare.MANIFEST_RELATIVE_PATH, prepare.INFO_RELATIVE_PATH,
                prepare.STATS_RELATIVE_PATH, "SHA256SUMS",
            )
        })
    resign()
    return root, tmp_path / "assets", rows, info, stats, resign


def test_binds_frozen_statistics_without_changing_the_published_dataset(published_package):
    from datasets.contract import DatasetContract
    from datasets.flooddepth_dataset import FloodDepthDataset
    from datasets.band_selection import BandSpec

    root, output, rows, *_ = published_package
    originals = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    contract_path, stats_path = prepare.build_assets(root, output)
    assert stats_path.read_bytes() == (root / prepare.STATS_RELATIVE_PATH).read_bytes()
    assert all(path.read_bytes() == data for path, data in originals.items())
    contract = DatasetContract.load(contract_path)
    assert not Path(contract.payload["dataset_root"]).is_absolute()
    assert contract.dataset_root == root
    contract.verify_fingerprints()
    selected = BandSpec.resolve(contract, {
        group: list(prepare.GROUP_SPECS[group].bands) for group in prepare.CONTINUOUS_GROUPS
    })
    dataset = FloodDepthDataset(contract_path, stats_path, "test", band_spec=selected, s1_qa_names=[])
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["metadata"]["sample_id"] == rows[-1]["sample_id"]
    assert sample["s1_t1"].shape == (2, 256, 256)
    assert sample["masks"]["valid_depth_mask"].sum() == 256 * 256 - 1
    assert sample["label"][0, 0, 0] == 0


def test_corrupted_file_cannot_replace_existing_verified_assets(published_package):
    root, output, rows, *_ = published_package
    prepare.build_assets(root, output)
    before = {path: path.read_bytes() for path in output.iterdir()}
    path = root / rows[0]["s1_t1_path"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(prepare.DatasetPreparationError, match="SHA-256 mismatch"):
        prepare.build_assets(root, output)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_frozen_statistics_are_bound_to_the_published_fingerprint(published_package):
    root, output, *_ = published_package
    path = root / prepare.STATS_RELATIVE_PATH
    data = json.loads(path.read_text())
    data["groups"]["s1_t1"][0]["mean"] += 1
    _json(path, data)
    with pytest.raises(prepare.DatasetPreparationError, match="SHA-256 mismatch"):
        prepare.build_assets(root, output)
    assert not output.exists()


def test_source_symlink_cannot_escape_the_dataset_root(published_package, tmp_path):
    root, output, rows, *_ = published_package
    path = root / rows[0]["s1_t1_path"]
    outside = tmp_path / "outside.tif"
    shutil.copyfile(path, outside)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(prepare.DatasetPreparationError, match="escaping dataset file"):
        prepare.build_assets(root, output)
    assert not output.exists()


def test_preparer_rejects_undistributed_raster_bands(published_package):
    root, output, rows, info, stats, resign = published_package
    path = root / rows[0]["s1_t1_path"]
    with rasterio.open(path, "r+") as source:
        source.set_band_description(2, "unpublished_auxiliary")
    resign()
    with pytest.raises(prepare.DatasetPreparationError, match="Raster bands"):
        prepare.build_assets(root, output)
    assert not output.exists()


def test_label_and_reference_support_must_agree(published_package):
    root, output, rows, info, stats, resign = published_package
    path = root / rows[0]["masks_path"]
    with rasterio.open(path, "r+") as source:
        support = source.read(1)
        support[0, 0] = 1
        source.write(support, 1)
    resign()
    with pytest.raises(prepare.DatasetPreparationError, match="label/valid_depth_mask mismatch"):
        prepare.build_assets(root, output)
    assert not output.exists()


def test_fixed_split_counts_cannot_change(published_package):
    root, output, rows, info, stats, resign = published_package
    rows[0]["split"] = "test"
    resign()
    with pytest.raises(prepare.DatasetPreparationError, match="fixed split counts"):
        prepare.build_assets(root, output)
    assert not output.exists()


def test_undeclared_modality_cannot_enter_the_release_contract(published_package):
    root, output, rows, info, stats, resign = published_package
    info["raster_groups"]["optical"] = dict(info["raster_groups"]["s1_t1"])
    resign()
    with pytest.raises(prepare.DatasetPreparationError, match="raster groups changed"):
        prepare.build_assets(root, output)
    assert not output.exists()
