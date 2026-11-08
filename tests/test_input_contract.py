from __future__ import annotations

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import RELIABILITY_NAMES, ReliabilitySpec
from utils.config import load_config


def test_production_input_contract_has_only_sar_and_terrain_groups() -> None:
    spec = ModelInputSpec.from_mode()
    assert spec.is_s1_only
    assert spec.continuous_groups == ("s1_t1", "s1_t2", "s1_change", "terrain")
    assert spec.qa_groups == ("s1_qa",)
    assert ReliabilitySpec.from_mode(spec.mode).names == RELIABILITY_NAMES


def test_nonproduction_input_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelInputSpec.from_mode("multisensor")


def test_active_full_release_configuration_explicitly_disables_sentinel2() -> None:
    config = load_config("configs/sar_topo_edge_kan.xml")
    assert config["dataset"]["name"] == "flooddepthnet_s1_terrain"
    assert config["dataset"]["sentinel2_enabled"] is False
    assert ModelInputSpec.from_config(config).is_s1_only


@pytest.fixture
def extra_band_sar_raster(tmp_path):
    path = tmp_path / "source.tif"
    names = ("VV_pre_db", "VH_pre_db", "unused_auxiliary")
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4, count=3,
        dtype="float32", crs="EPSG:32632", transform=from_origin(0, 80, 20, 20),
    ) as target:
        target.write(np.stack([np.full((4, 4), value, dtype=np.float32) for value in (1, 2, 999)]))
        target.descriptions = names
    return path, names


def test_selected_sar_view_reads_only_declared_polarization_bands(extra_band_sar_raster, monkeypatch) -> None:
    """An extra source-raster band must never enter the runtime input view."""
    path, names = extra_band_sar_raster
    tmp_path = path.parent
    contract = DatasetContract(tmp_path / "contract.json", {
        "dataset_root": str(tmp_path),
        "raster_groups": {"s1_t1": {
            "path_column": "s1_path", "band_count": 2,
            "band_descriptions": list(names[:2]),
        }},
    })
    dataset = object.__new__(FloodDepthDataset)
    dataset.contract = contract
    calls = []
    original_open = rasterio.open

    class RecordingReader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.source, name)

        def read(self, *args, **kwargs):
            calls.append(("values", tuple(kwargs["indexes"])))
            return self.source.read(*args, **kwargs)

        def read_masks(self, *args, **kwargs):
            calls.append(("validity", tuple(kwargs["indexes"])))
            return self.source.read_masks(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", lambda *args, **kwargs: RecordingReader(original_open(*args, **kwargs)))
    values, valid, _ = dataset._read({"s1_path": path.name}, "s1_t1", (0, 1))
    assert calls == [("values", (1, 2)), ("validity", (1, 2))]
    assert values.shape == valid.shape == (2, 4, 4)
    assert np.all(values[0] == 1) and np.all(values[1] == 2)
    assert valid.all()


def test_default_runtime_band_spec_is_the_declared_sar_view(production_config) -> None:
    contract = DatasetContract.load(production_config["dataset"]["contract"])
    band_spec = resolve_band_spec(production_config, contract)
    assert tuple(band_spec.groups) == ModelInputSpec.from_config(production_config).continuous_groups
    assert band_spec.names("s1_t1") == ("VV_pre_db", "VH_pre_db")
    assert band_spec.names("s1_t2") == ("VV_event_db", "VH_event_db")
    assert band_spec.read_indexes(contract, "s1_t1") == (0, 1)
    assert band_spec.read_indexes(contract, "s1_t2") == (0, 1)


@pytest.fixture
def extended_sar_contract(tmp_path):
    names = {
        "s1_t1": ["VV_pre_db", "VH_pre_db", "unused_auxiliary_pre"],
        "s1_t2": ["VV_event_db", "VH_event_db", "unused_auxiliary_event"],
        "s1_change": ["VV_delta_db", "VH_delta_db", "anomaly_raw"],
        "terrain": ["elevation_m_DSM", "slope_deg"],
    }
    return DatasetContract(tmp_path / "contract.json", {
        "raster_groups": {
            group: {"band_descriptions": bands, "band_count": len(bands)}
            for group, bands in names.items()
        },
    })


def test_default_band_selection_excludes_extra_source_channels(extended_sar_contract) -> None:
    spec = BandSpec.resolve(extended_sar_contract, None)
    assert spec.uses_default_bands
    assert spec.names("s1_t1") == ("VV_pre_db", "VH_pre_db")
    assert spec.names("s1_t2") == ("VV_event_db", "VH_event_db")
    assert spec.read_indexes(extended_sar_contract, "s1_t1") == (0, 1)
    assert spec.read_indexes(extended_sar_contract, "s1_t2") == (0, 1)


def test_explicit_band_selection_rejects_extra_source_channels(extended_sar_contract) -> None:
    selected = {
        group: data["band_descriptions"]
        for group, data in extended_sar_contract.payload["raster_groups"].items()
    }
    with pytest.raises(ValueError, match="Only VV/VH polarization bands"):
        BandSpec.resolve(extended_sar_contract, selected)
