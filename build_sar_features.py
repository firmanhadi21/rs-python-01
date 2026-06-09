#!/usr/bin/env python3
"""Build SAR feature stacks (Sentinel-1 temporal + ALOS-2 PALSAR) over the
Cisokan AOI from Earth Engine, write them as local GeoTIFFs that the v3
OBIA classifier picks up automatically.

Outputs (next to this script):
    S1_temporal_features.tif   - 19 bands of Sentinel-1 temporal stats
    PALSAR_features.tif        -  5 bands of PALSAR-2 backscatter + indices

Notes
-----
* Sentinel-1 window is 2024-03-01 to 2026-03-31 to bracket the PlanetScope
  10-epoch stack used by planetscope_10epoch_obia_v3.py.
* PALSAR-2 yearly mosaic is asset 'JAXA/ALOS/PALSAR/YEARLY/SAR'. The script
  probes years 2024 -> 2017 and uses the most recent year that returns data
  for the AOI. ALOS-1 (2007-2010) is excluded.
* Default scale is 30 m so the result fits Earth Engine's direct-download
  size cap (~32-50 MB). v3 reproject_match()'es to the PlanetScope grid, so
  the local resolution doesn't need to match. Pass --scale 10 to force full
  resolution; that will route the export through Drive (slower; needs
  --download-from-drive after).

Auth
----
Uses ee_init.init_ee(), which reads the same GEE_SERVICE_ACCOUNT_* env vars
as ~/Github/forest-analyzer. With no env set it falls back to the key under
forest-analyzer/config/ee-geodetic.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import ee

from ee_init import init_ee


PROJECT_ROOT = Path(__file__).resolve().parent
AOI_GPKG     = PROJECT_ROOT / "aoi_cisokan.gpkg"
S1_OUT       = PROJECT_ROOT / "S1_temporal_features.tif"
PALSAR_OUT   = PROJECT_ROOT / "PALSAR_features.tif"

S1_START = "2024-03-01"
S1_END   = "2026-03-31"

PALSAR_ASSET = "JAXA/ALOS/PALSAR/YEARLY/SAR"
PALSAR_YEAR_PROBE_RANGE = range(2024, 2016, -1)   # 2024 -> 2017

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build-sar")


# -------------------------------------------------------------------------
# AOI
# -------------------------------------------------------------------------
def aoi_to_ee_geometry(gpkg: Path) -> ee.Geometry:
    """Read the AOI from a GeoPackage and return it as an ee.Geometry in
    EPSG:4326."""
    import json
    import geopandas as gpd

    gdf = gpd.read_file(gpkg).to_crs("EPSG:4326")
    if len(gdf) == 1:
        return ee.Geometry(json.loads(gdf.iloc[[0]].to_json())["features"][0]["geometry"])
    # Multiple features: dissolve into a single MultiPolygon
    dissolved = gdf.dissolve()
    return ee.Geometry(json.loads(dissolved.to_json())["features"][0]["geometry"])


# -------------------------------------------------------------------------
# Sentinel-1 temporal stack
# -------------------------------------------------------------------------
def build_s1_temporal(geom: ee.Geometry, start: str, end: str) -> ee.Image:
    log.info("Building Sentinel-1 temporal stack: %s to %s", start, end)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
          .select(["VV", "VH"]))

    n = s1.size().getInfo()
    log.info("  %d S1 scenes after filtering", n)
    if n == 0:
        raise RuntimeError("No Sentinel-1 scenes match filters; widen window "
                           "or relax the orbit/polarization filters.")

    def to_db(img: ee.Image) -> ee.Image:
        # Clamp non-positive linear values to 1e-10 (~ -100 dB, well below the
        # S1 noise floor) so log10 stays finite and the mean() reducer has a
        # value at every pixel. Without this, very dark VH pixels (forest,
        # water, sub-noise) get masked, and the reducer propagates the mask
        # to the exported tile -- a single tile we observed with only 0.2%
        # finite VH pixels.
        return img.max(1e-10).log10().multiply(10) \
                  .copyProperties(img, ["system:time_start"])

    s1_db = s1.map(to_db)

    mean_   = s1_db.mean().rename(["VV_mean", "VH_mean"])
    std_    = s1_db.reduce(ee.Reducer.stdDev()).rename(["VV_stdDev", "VH_stdDev"])
    min_    = s1_db.reduce(ee.Reducer.min()).rename(["VV_min", "VH_min"])
    max_    = s1_db.reduce(ee.Reducer.max()).rename(["VV_max", "VH_max"])

    cv_vv = std_.select("VV_stdDev").divide(mean_.select("VV_mean")) \
                .abs().rename("VV_CV")
    cv_vh = std_.select("VH_stdDev").divide(mean_.select("VH_mean")) \
                .abs().rename("VH_CV")

    # Seasonal composites (Indonesia: wet Nov-Mar, dry Apr-Oct)
    wet = (s1_db.filter(ee.Filter.calendarRange(11, 3, "month"))
                .mean().rename(["VV_wet", "VH_wet"]))
    dry = (s1_db.filter(ee.Filter.calendarRange(4, 10, "month"))
                .mean().rename(["VV_dry", "VH_dry"]))
    seas_diff_vv = wet.select("VV_wet").subtract(dry.select("VV_dry")) \
                      .rename("VV_seasonal_diff")
    seas_diff_vh = wet.select("VH_wet").subtract(dry.select("VH_dry")) \
                      .rename("VH_seasonal_diff")

    # Indices on the temporal mean (3 NEW bands only -- avoid the duplicate-band
    # bug in the original gee.js where addS1Indices(s1_temporal) was added back
    # on top of s1_temporal).
    vv = mean_.select("VV_mean")
    vh = mean_.select("VH_mean")
    ratio = vv.subtract(vh).rename("VV_VH_ratio")
    rvi   = vh.multiply(4).divide(vv.add(vh)).rename("RVI_S1")
    dpsvi = vh.add(vv).divide(2).rename("DPSVI_S1")

    out = (mean_.addBands(std_).addBands(min_).addBands(max_)
                .addBands(cv_vv).addBands(cv_vh)
                .addBands(wet).addBands(dry)
                .addBands(seas_diff_vv).addBands(seas_diff_vh)
                .addBands(ratio).addBands(rvi).addBands(dpsvi)
                .toFloat()
                .clip(geom))
    log.info("  S1 stack bands: %s", out.bandNames().getInfo())
    return out


# -------------------------------------------------------------------------
# PALSAR-2 yearly mosaic
# -------------------------------------------------------------------------
def build_palsar(geom: ee.Geometry,
                 probe_years: range = PALSAR_YEAR_PROBE_RANGE
                 ) -> ee.Image:
    """Pick the most recent PALSAR yearly mosaic year that has data over the
    AOI, then build HH/HV in dB plus three indices."""
    chosen: Optional[int] = None
    for year in probe_years:
        col = (ee.ImageCollection(PALSAR_ASSET)
               .filterBounds(geom)
               .filterDate(f"{year}-01-01", f"{year}-12-31"))
        size = col.size().getInfo()
        if size > 0:
            chosen = year
            log.info("PALSAR yearly mosaic year selected: %d  (%d image(s))",
                     year, size)
            break
        log.debug("  no PALSAR data for %d", year)
    if chosen is None:
        raise RuntimeError(
            f"No PALSAR yearly mosaic found in {probe_years.start}..{probe_years.stop+1}"
            " over this AOI. Widen probe_years or check the asset id.")

    palsar = (ee.ImageCollection(PALSAR_ASSET)
              .filterBounds(geom)
              .filterDate(f"{chosen}-01-01", f"{chosen}-12-31")
              .select(["HH", "HV"])
              .mosaic()
              .clip(geom))

    # DN -> sigma0 (dB):  20*log10(DN) - 83
    palsar_db = ee.Image.cat([
        ee.Image(20).multiply(palsar.select("HH").log10()).subtract(83).rename("HH_db"),
        ee.Image(20).multiply(palsar.select("HV").log10()).subtract(83).rename("HV_db"),
    ])

    hh = palsar_db.select("HH_db")
    hv = palsar_db.select("HV_db")
    ratio = hh.subtract(hv).rename("HH_HV_ratio")
    rvi   = hv.multiply(4).divide(hh.add(hv)).rename("RVI_PALSAR")
    dpsvi = hv.add(hh).divide(2).rename("DPSVI_PALSAR")

    out = palsar_db.addBands([ratio, rvi, dpsvi]).toFloat().set("year", chosen)
    log.info("  PALSAR bands: %s", out.bandNames().getInfo())
    return out


# -------------------------------------------------------------------------
# Export helpers (direct download preferred; Drive as fallback)
# -------------------------------------------------------------------------
def export_direct(image: ee.Image, region: ee.Geometry, scale: int,
                  out_path: Path) -> None:
    """getDownloadURL -> single-shot fetch -> write multi-band TIFF with band
    descriptions. Handles both response shapes EE returns: a zip of per-band
    TIFs (large requests) or a single multi-band TIFF (small requests).
    Hard cap is ~50 MB; bump --scale if you bust it."""
    import io
    import tempfile
    import rasterio

    band_names = image.bandNames().getInfo()
    log.info("Direct download: %d bands at %d m -> %s",
             len(band_names), scale, out_path.name)

    url = image.getDownloadURL({
        "scale":  scale,
        "region": region,
        "crs":    "EPSG:32748",
        "format": "GEO_TIFF",
    })
    log.info("  fetching %s ...", url[:90] + "...")

    with urllib.request.urlopen(url) as resp:
        body = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    log.info("  received %.1f MB (%s)", len(body) / 1e6, ctype)

    is_zip = body[:2] == b"PK"
    is_tiff = body[:4] in (b"II*\x00", b"MM\x00*")

    if is_zip:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            with zipfile.ZipFile(io.BytesIO(body)) as zf:
                zf.extractall(tdp)
            per_band = [tdp / f"download.{b}.tif" for b in band_names]
            missing = [p.name for p in per_band if not p.exists()]
            if missing:
                existing = sorted(p.name for p in tdp.glob("*.tif"))
                raise RuntimeError(
                    f"Expected per-band TIFs {missing[:3]} not in zip. "
                    f"Found: {existing[:5]}")
            with rasterio.open(per_band[0]) as src0:
                profile = src0.profile.copy()
            profile.update(count=len(per_band), compress="deflate",
                           tiled=True, BIGTIFF="IF_SAFER")
            with rasterio.open(out_path, "w", **profile) as dst:
                for i, (p, name) in enumerate(zip(per_band, band_names),
                                              start=1):
                    with rasterio.open(p) as src:
                        dst.write(src.read(1), i)
                    dst.set_band_description(i, name)

    elif is_tiff:
        # Single multi-band TIFF: rewrite with deflate compression and
        # band descriptions, since the EE response may be uncompressed.
        with rasterio.open(io.BytesIO(body)) as src:
            if src.count != len(band_names):
                raise RuntimeError(
                    f"TIFF has {src.count} bands but EE reported "
                    f"{len(band_names)} band names")
            profile = src.profile.copy()
            profile.update(compress="deflate", tiled=True,
                           BIGTIFF="IF_SAFER")
            with rasterio.open(out_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
                    dst.set_band_description(i, band_names[i - 1])

    else:
        head = body[:200].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Unexpected response (not zip / not TIFF). Content-Type={ctype}. "
            f"First bytes: {head}")

    log.info("  wrote %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def export_drive(image: ee.Image, region: ee.Geometry, scale: int,
                 description: str, file_prefix: str, folder: str = "CSK_GEE"
                 ) -> ee.batch.Task:
    log.info("Submitting Drive export: %s @ %d m", description, scale)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_prefix,
        region=region,
        scale=scale,
        maxPixels=int(1e13),
        crs="EPSG:32748",
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def wait_for_task(task: ee.batch.Task, poll_sec: int = 30) -> None:
    while True:
        status = task.status()
        state = status.get("state")
        log.info("  Drive task %s: %s", task.id, state)
        if state == "COMPLETED":
            return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Export failed: {status}")
        time.sleep(poll_sec)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=int, default=30,
                   help="Export scale in metres (default 30; 10 forces Drive)")
    p.add_argument("--via-drive", action="store_true",
                   help="Force Drive export even at 30 m")
    p.add_argument("--skip-s1",     action="store_true")
    p.add_argument("--skip-palsar", action="store_true")
    args = p.parse_args()

    init_ee()

    if not AOI_GPKG.exists():
        log.error("AOI not found: %s", AOI_GPKG)
        return 2
    geom = aoi_to_ee_geometry(AOI_GPKG)

    use_drive = args.via_drive or args.scale < 20
    if use_drive:
        log.info("Routing exports through Drive (scale=%d, --via-drive=%s)",
                 args.scale, args.via_drive)
        log.info("Output files will land in your Google Drive under "
                 "folder 'CSK_GEE'; download them to %s manually.", PROJECT_ROOT)

    tasks: list[ee.batch.Task] = []

    if not args.skip_s1:
        s1 = build_s1_temporal(geom, S1_START, S1_END)
        if use_drive:
            tasks.append(export_drive(
                s1, geom, args.scale, "S1_temporal_features",
                "S1_temporal_features"))
        else:
            export_direct(s1, geom, args.scale, S1_OUT)

    if not args.skip_palsar:
        palsar = build_palsar(geom)
        if use_drive:
            tasks.append(export_drive(
                palsar, geom, args.scale, "PALSAR_features",
                "PALSAR_features"))
        else:
            export_direct(palsar, geom, args.scale, PALSAR_OUT)

    for t in tasks:
        wait_for_task(t)

    log.info("Done. v3 will pick these up automatically on its next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
