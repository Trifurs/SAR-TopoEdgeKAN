from __future__ import annotations

import csv

from utils.logging import append_csv


def test_append_csv_accepts_reordered_fields(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 0, "loss": 0.5})
    append_csv(path, {"loss": 0.4, "epoch": 1})

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == ["epoch", "loss"]
    assert rows == [
        {"epoch": "0", "loss": "0.5"},
        {"epoch": "1", "loss": "0.4"},
    ]


def test_append_csv_expands_optional_metric_schema(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 0, "loss": 0.5})
    append_csv(path, {"epoch": 1, "mae": 0.4})

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["epoch", "loss", "mae"]
    assert rows == [
        {"epoch": "0", "loss": "0.5", "mae": ""},
        {"epoch": "1", "loss": "", "mae": "0.4"},
    ]
