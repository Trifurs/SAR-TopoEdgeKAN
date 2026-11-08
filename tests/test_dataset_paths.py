import pytest

from datasets.contract import ContractError, DatasetContract


def test_dataset_root_is_relative_to_contract_after_relocation(tmp_path, monkeypatch):
    """Moving an extracted project must not depend on the shell's directory."""
    project = tmp_path / "relocated_project"
    assets = project / "assets" / "dataset"
    data = project / "data" / "FloodDepthNet"
    assets.mkdir(parents=True)
    (data / "metadata").mkdir(parents=True)
    manifest = data / "metadata" / "training_manifest.csv"
    manifest.write_text("sample_id,split\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    contract = DatasetContract(assets / "dataset_contract.json", {
        "dataset_root": "../../data/FloodDepthNet",
        "manifest": {"relative_path": "metadata/training_manifest.csv"},
    })
    assert contract.dataset_root == data.resolve()
    assert contract.manifest_path == manifest.resolve()


def test_absolute_dataset_root_remains_supported(tmp_path):
    contract = DatasetContract(tmp_path / "contract.json", {"dataset_root": str(tmp_path)})
    assert contract.dataset_root == tmp_path.resolve()


def test_manifest_cannot_escape_relocated_dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("sample_id\n", encoding="utf-8")
    contract = DatasetContract(tmp_path / "contract.json", {
        "dataset_root": "dataset", "manifest": {"relative_path": "../outside.csv"},
    })
    with pytest.raises(ContractError, match="escapes dataset root"):
        contract.manifest_path
