#!/usr/bin/env python3
"""Download the Meta Data-for-Good v2 canopy-height layers over the Cisokan AOI.

Source bucket: s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg/
Tile covering Cisokan: meta_chm_lat=-0.0_lon=100.0_<stat>.tif (lon 100-110E, lat 0 to -10S).

Streams the AOI window directly via /vsis3/ (no full tile download), reprojects
to EPSG:32748 (UTM 48S) at 3 m, saves to rasters/meta_v2/.

Anonymous S3 read; no AWS credentials needed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Anonymous read of the AWS Open Data bucket
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import rioxarray
from rasterio.enums import Resampling


PROJECT_ROOT = Path(__file__).resolve().parent
AOI_GPKG     = PROJECT_ROOT / "aoi_cisokan.gpkg"
OUT_DIR      = PROJECT_ROOT / "rasters" / "meta_v2"

S3_PREFIX = (
    "/vsis3/dataforgood-fb-data/forests/v1/"
    "alsgedi_global_v6_float_epsg4326_v3_10deg"
)
TILE_BASENAME = "meta_chm_lat=-0.0_lon=100.0"

# Statistics to download. avg + stdev are the headline biomass + heterogeneity
# pair; p95 captures emergents; cover is canopy fraction.
#
# The source COGs are uint16 with no embedded scale/nodata metadata:
#  - heights (avg / stdev / p95) are stored in centimetres (divide by 100 to get m)
#  - cover is stored as 0..1000 (divide by 1000 to get fraction)
#  - 65535 is used as a nodata sentinel
# We mask the sentinel and apply the scale before reprojecting so downstream
# consumers see clean float32 values in natural units.
STATS = {
    "avg":   {"scale": 1.0 / 100.0, "unit": "metres"},
    "stdev": {"scale": 1.0 / 100.0, "unit": "metres"},
    "p95":   {"scale": 1.0 / 100.0, "unit": "metres"},
    "cover": {"scale": 1.0 / 1000.0, "unit": "fraction"},
}
SOURCE_NODATA = 65535  # uint16 sentinel

# Output resolution and CRS
OUT_CRS = "EPSG:32748"
OUT_RES = 3.0  # metres, matches PlanetScope grid
AOI_BUFFER_DEG = 0.01  # ~1.1 km cushion around the AOI bbox in lon/lat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meta-canopy-dl")


def aoi_bbox_4326() -> tuple[float, float, float, float]:
    aoi = gpd.read_file(AOI_GPKG).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = aoi.total_bounds
    log.info("AOI bbox (EPSG:4326): %.4f, %.4f -> %.4f, %.4f",
             minx, miny, maxx, maxy)
    b = AOI_BUFFER_DEG
    return (minx - b, miny - b, maxx + b, maxy + b)


def download_one_stat(stat: str, scale: float, unit: str,
                      bbox4326: tuple) -> Path:
    s3_url = f"{S3_PREFIX}/{TILE_BASENAME}_{stat}.tif"
    out_path = OUT_DIR / f"canopy_height_v2_{stat}.tif"

    if out_path.exists():
        log.info("[skip] %s already exists (%.1f MB)",
                 out_path.name, out_path.stat().st_size / 1e6)
        return out_path

    log.info("[fetch] %s  -> %s", stat, unit)
    log.info("        from %s", s3_url)
    # masked=False so we keep the raw uint16 values; we'll handle the
    # sentinel + scale ourselves.
    import numpy as np
    import xarray as xr
    da = rioxarray.open_rasterio(s3_url, masked=False, chunks={"x": 2048, "y": 2048})
    log.info("        source CRS=%s, res=%.6f deg, shape=%s, dtype=%s",
             da.rio.crs, abs(da.rio.resolution()[0]), tuple(da.shape), da.dtype)

    # Window-read just the AOI (this is the cheap part with COG)
    da_clip = da.rio.clip_box(*bbox4326, crs="EPSG:4326")
    log.info("        clipped to AOI: shape=%s", tuple(da_clip.shape))

    # Mask the uint16 sentinel and apply the scale to get natural units.
    arr = da_clip.values.astype("float32")
    arr = np.where(arr >= SOURCE_NODATA, np.nan, arr) * scale
    da_scaled = xr.DataArray(
        arr,
        dims=da_clip.dims,
        coords=da_clip.coords,
        attrs={"units": unit, "long_name": f"meta_canopy_v2_{stat}"},
    )
    da_scaled.rio.write_crs(da_clip.rio.crs, inplace=True)
    da_scaled.rio.write_transform(da_clip.rio.transform(), inplace=True)
    da_scaled.rio.write_nodata(np.nan, inplace=True)

    valid = np.isfinite(arr)
    if valid.any():
        log.info("        scaled (units=%s): min=%.3f median=%.3f p95=%.3f max=%.3f",
                 unit,
                 float(np.nanmin(arr)), float(np.nanmedian(arr)),
                 float(np.nanpercentile(arr, 95)), float(np.nanmax(arr)))

    # Reproject to UTM at 3 m. Use 'average' so source pixels are spatially
    # averaged into 3 m output pixels; nodata propagates as NaN.
    da_utm = da_scaled.rio.reproject(
        OUT_CRS,
        resolution=OUT_RES,
        resampling=Resampling.average,
    )
    log.info("        reprojected to %s @ %.1f m: shape=%s",
             OUT_CRS, OUT_RES, tuple(da_utm.shape))

    # Explicit -9999 nodata for clean writing
    da_utm = da_utm.rio.write_nodata(-9999.0, encoded=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    da_utm.rio.to_raster(
        out_path,
        dtype="float32",
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    log.info("        wrote %s (%.1f MB)",
             out_path.name, out_path.stat().st_size / 1e6)
    return out_path


def main() -> int:
    bbox = aoi_bbox_4326()
    log.info("AOI bbox + %.3f deg buffer: %s", AOI_BUFFER_DEG, bbox)
    paths = []
    for stat, meta in STATS.items():
        try:
            paths.append(download_one_stat(stat, meta["scale"], meta["unit"], bbox))
        except Exception as e:
            log.error("[fail] stat=%s: %s: %s", stat, type(e).__name__, e)
            return 2
    log.info("Done. %d layers in %s", len(paths), OUT_DIR)
    for p in paths:
        log.info("  - %s", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
