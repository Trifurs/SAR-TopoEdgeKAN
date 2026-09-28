"""Atomic GeoTIFF export preserving the audited source grid."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def write_geotiff(
    path: Path,
    array: np.ndarray,
    *,
    crs: str | CRS,
    transform: Sequence[float] | Affine,
    nodata: float = -9999.0,
    valid_mask: np.ndarray | None = None,
    descriptions: Sequence[str] | None = None,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"GeoTIFF array must be [bands,height,width], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("GeoTIFF source array contains NaN or Inf")
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.ndim == 3:
            valid = valid[0]
        if valid.shape != values.shape[-2:]:
            raise ValueError(f"Mask shape {valid.shape} != raster shape {values.shape[-2:]}")
        values = np.where(valid[None], values, nodata)
    affine = transform if isinstance(transform, Affine) else Affine(*tuple(transform)[:6])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".tif", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=values.shape[-1],
            height=values.shape[-2],
            count=values.shape[0],
            dtype="float32",
            crs=crs,
            transform=affine,
            nodata=nodata,
            compress="deflate",
            predictor=3,
            tiled=True,
        ) as dataset:
            dataset.write(values)
            if descriptions is not None:
                if len(descriptions) != values.shape[0]:
                    raise ValueError("Description count does not match band count")
                dataset.descriptions = tuple(descriptions)
            if valid_mask is not None:
                dataset.write_mask(valid.astype(np.uint8) * 255)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def reference_grid_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "crs": metadata["crs"],
        "transform": metadata["transform"],
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
    }
