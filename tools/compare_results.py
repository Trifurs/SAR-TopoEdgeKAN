#!/usr/bin/env python3
"""Combine SAR-TopoEdgeKAN and named comparison summaries on one metric domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import write_rows
from utils.misc import atomic_write_json, atomic_write_text


METRICS = (
    "pixel_micro_mae", "pixel_micro_rmse", "pixel_micro_bias", "pixel_micro_log1p_mae",
    "event_macro_mae", "event_macro_rmse", "event_macro_bias", "event_macro_log1p_mae",
)

KNOWN_COMPARISON_MODELS = {
    "fwdet_v2",
    "tsa",
    "fldepth",
    "dlsim_attention_unet",
    "dlsim_linknet",
    "unet_depth_regression",
    "resnet18_depth_regression",
    "unetplusplus_depth_regression",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, support: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": name,
        "support_condition": support,
        "evaluated_pixels": int(payload["pixel_micro_pixels"]),
        **{metric: payload[metric] for metric in METRICS},
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    fields = ("method", "support_condition", "evaluated_pixels", *METRICS)
    header = "| " + " | ".join(field.replace("_", " ") for field in fields) + " |\n"
    separator = "|" + "|".join("---" for _ in fields) + "|\n"
    body = "".join(
        "| " + " | ".join(
            f"{float(row[field]):.4f}" if isinstance(row[field], float) else str(row[field])
            for field in fields
        ) + " |\n"
        for row in rows
    )
    return "# Comparable evaluation summary\n\n" + header + separator + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument(
        "--baseline-summary", type=Path, action="append", default=[],
        help="Traditional comparison summary (repeatable).",
    )
    parser.add_argument(
        "--model-summary", type=Path, action="append", default=[],
        help="Named learned or non-learned comparison summary (repeatable).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_summary = _read(args.main_summary)
    rows = [_row("SAR-TopoEdgeKAN", "not_applicable", main_summary)]
    summary_paths = [*args.baseline_summary, *args.model_summary]
    if not summary_paths:
        raise ValueError("At least one --baseline-summary or --model-summary is required")
    for path in summary_paths:
        summary = _read(path)
        if summary.get("flood_support") != "valid_depth_mask":
            raise ValueError(
                f"Baseline summary must use valid_depth_mask directly: {path}"
            )
        if summary.get("method") not in KNOWN_COMPARISON_MODELS:
            raise ValueError(f"Unknown comparison model in {path}: {summary.get('method')!r}")
        rows.append(_row(summary["method"], "valid_depth_mask", summary))
    domains = {row["evaluated_pixels"] for row in rows}
    if len(domains) != 1:
        raise ValueError(f"Cannot compare different evaluation domains: {sorted(domains)}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "comparison.csv", rows)
    atomic_write_json(output / "comparison.json", rows)
    atomic_write_text(output / "comparison.md", _markdown(rows))
    print(_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
