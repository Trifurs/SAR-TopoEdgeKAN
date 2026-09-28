"""Mask-aware train-statistics preprocessing for raster inputs and safe QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from datasets.contract import DatasetContract, sha256_file
from datasets.reliability_spec import RELIABILITY_NAMES, ReliabilitySpec


def reliability_spec_for_mode(input_mode: str = "s1_terrain") -> ReliabilitySpec:
    """Resolve the named reliability schema without positional assumptions."""

    return ReliabilitySpec.from_mode(input_mode)


class PreprocessingError(RuntimeError):
    """Raised when frozen train statistics do not match the audited dataset."""


class RobustNormalizer:
    """Apply train-only p0.5/p99.5 clipping and mean/std normalization."""

    def __init__(self, stats_path: Path, contract: DatasetContract) -> None:
        self.path = Path(stats_path).expanduser().resolve(strict=True)
        self.stats = json.loads(self.path.read_text(encoding="utf-8"))
        if self.stats.get("scope") != "train split only; valid pixels only; val/test excluded":
            raise PreprocessingError(f"Unsafe normalization scope: {self.stats.get('scope')!r}")
        expected_manifest = contract.payload["manifest"]["sha256"]
        if self.stats.get("manifest_sha256") != expected_manifest:
            raise PreprocessingError(
                "Normalization/manifest mismatch: "
                f"{self.stats.get('manifest_sha256')} != {expected_manifest}"
            )
        selected = contract.payload.get("normalization", {}).get("selected")
        if not selected or selected.get("sha256") != sha256_file(self.path):
            raise PreprocessingError("Normalization file is not fingerprint-bound to the contract")
        self._groups = {
            group: {entry["band"]: entry for entry in entries}
            for group, entries in self.stats["groups"].items()
        }
        self._qa_groups = {
            group: {entry["band"]: entry for entry in entries}
            for group, entries in self.stats["qa_groups"].items()
        }

    @staticmethod
    def _apply(array: np.ndarray, valid: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
        std = float(entry["std"])
        if not np.isfinite(std) or std <= 0:
            raise PreprocessingError(f"Invalid std for {entry['band']}: {std}")
        clipped = np.clip(array, float(entry["p0.5"]), float(entry["p99.5"]))
        normalized = (clipped - float(entry["mean"])) / std
        return np.where(valid & np.isfinite(normalized), normalized, 0.0).astype(np.float32)

    def continuous(
        self,
        group: str,
        descriptions: list[str],
        array: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        entries = self._groups.get(group)
        if entries is None:
            raise PreprocessingError(f"No train statistics for group {group}")
        result = np.empty_like(array, dtype=np.float32)
        for index, description in enumerate(descriptions):
            if description not in entries:
                raise PreprocessingError(f"No train statistics for {group}/{description}")
            result[index] = self._apply(array[index], valid[index], entries[description])
        return result

    def qa_feature(
        self, group: str, description: str, transformed: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        try:
            entry = self._qa_groups[group][description]
        except KeyError as exc:
            raise PreprocessingError(f"No safe-QA statistics for {group}/{description}") from exc
        return self._apply(transformed, valid, entry)

    @property
    def train_depth_bins(self) -> list[float]:
        return [float(value) for value in self.stats["train_depth"]["stratification_bin_edges"]]

    @property
    def train_depth_bin_counts(self) -> list[float] | None:
        counts = self.stats.get("train_depth", {}).get("stratification_bin_counts")
        if counts is None:
            return None
        return [float(value) for value in counts]


def resolve_depth_stratification_bins(
    loss_config: Mapping[str, Any], normalizer: RobustNormalizer
) -> list[float]:
    """Resolve frozen train-only depth strata and reject unsafe overrides.

    The normalization artifact retains the original quartile strata. Experiments may
    refine the heavy positive-depth tail with explicit edges computed from the same
    train pixels. Requiring the observed train extrema as endpoints prevents an
    internal threshold from being silently discarded as an endpoint by the loss and
    metric reducers.
    """

    configured = loss_config.get("depth_stratification_edges_m")
    if configured is None:
        return normalizer.train_depth_bins
    if not isinstance(configured, (list, tuple)):
        raise PreprocessingError("depth_stratification_edges_m must be an XML list")
    edges = [float(value) for value in configured]
    if len(edges) < 2 or not np.all(np.isfinite(edges)):
        raise PreprocessingError(
            "depth_stratification_edges_m must contain at least two finite values"
        )
    if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
        raise PreprocessingError("depth_stratification_edges_m must be strictly increasing")
    train_depth = normalizer.stats["train_depth"]
    observed_minimum = float(train_depth["minimum"])
    observed_maximum = float(train_depth["maximum"])
    tolerance = 1e-5
    if not np.isclose(edges[0], observed_minimum, rtol=0.0, atol=tolerance):
        raise PreprocessingError(
            "First depth-stratification edge must equal the observed train minimum: "
            f"{edges[0]} != {observed_minimum}"
        )
    if not np.isclose(edges[-1], observed_maximum, rtol=0.0, atol=tolerance):
        raise PreprocessingError(
            "Last depth-stratification edge must equal the observed train maximum: "
            f"{edges[-1]} != {observed_maximum}"
        )
    for primary_edge in normalizer.train_depth_bins:
        if not any(
            np.isclose(primary_edge, edge, rtol=0.0, atol=tolerance)
            for edge in edges
        ):
            raise PreprocessingError(
                "Refined depth strata must retain every original train-quantile edge: "
                f"missing {primary_edge}"
            )
    return edges
