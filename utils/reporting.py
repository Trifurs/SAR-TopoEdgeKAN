"""Heterogeneous experiment-table and numerical-summary helpers."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from utils.misc import atomic_write_text


def scalar_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def scalar_row(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in values.items() if scalar_value(value)}


def write_union_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows with a stable union schema instead of assuming identical keys."""

    materialized = [scalar_row(row) for row in rows]
    if not materialized:
        atomic_write_text(path, "")
        return
    preferred = [
        "model",
        "display_name",
        "model_family",
        "split",
        "source_run",
        "run_directory",
    ]
    all_keys = {key for row in materialized for key in row}
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(all_keys.difference(fieldnames)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(materialized)
    atomic_write_text(path, buffer.getvalue())


def numerical_summary_rows(
    rows: Iterable[Mapping[str, Any]], *, group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Return long-form mean/std/min/max summaries for all finite numeric fields."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    excluded = set(group_keys) | {"checkpoint_epoch"}
    for group, items in sorted(grouped.items()):
        names = sorted({name for item in items for name in item}.difference(excluded))
        for name in names:
            values: list[float] = []
            for item in items:
                value = item.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                if math.isfinite(number):
                    values.append(number)
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            result.append(
                {
                    **dict(zip(group_keys, group)),
                    "metric": name,
                    "n": int(array.size),
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                    "minimum": float(array.min()),
                    "maximum": float(array.max()),
                }
            )
    return result
