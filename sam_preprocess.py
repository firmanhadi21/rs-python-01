"""SAM preprocessing for World Imagery_9862E977172.tif.

Goal: produce a uint8 RGB COG over the WZ envelope (no L1 masking) plus a
sidecar L1=Dense binary mask used downstream to filter SAM segments.

Why no input masking: zeroing-out non-Dense pixels gives SAM huge uniform
"black" regions that it segments as a single object, and it also breaks
canopy continuity at L1 class boundaries. Better: feed SAM the natural
imagery, then in the filter step keep only polygons that intersect Dense.

Pipeline:
  1. Read WZ.gpkg (EPSG:32748) → bbox in EPSG:32748.
  2. Reproject WZ bbox to EPSG:3857 to clip the source TIF.
  3. Open World Imagery_9862E977172.tif (UInt32 4-band, EPSG:3857 implicit).
     Bands 1/2/3 are R/G/B with values already in [0, 255]; band 4 is alpha.
  4. Window-read the bbox region, cast bands 1-3 to uint8 → RGB.
  5. Reproject to EPSG:32748 at TARGET_PX_M.
  6. Build the L1=Dense mask on the same grid (nearest-neighbour resample
     of PS_LandCover_OBIA_v3.tif).
  7. Write outputs: full RGB (unmasked) + binary mask.

Outputs:
  rasters/sam_input_wz_dense.tif        — uint8 RGB, full WZ envelope (NO masking)
  rasters/sam_input_wz_dense_mask.tif   — uint8, 1 where L1=Dense else 0 (for filtering)
"""

from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.crs import CRS

ROOT = Path("/Users/firmanhadi/Works/Cisokan/2026/Apr26/process/CSK")
SRC_TIF = ROOT / "rasters" / "World Imagery_9862E977172.tif"
CLIP_GPKG = ROOT / "aoi_cisokan.gpkg"  # changed from WZ.gpkg to full AOI (265 km²)
L1_TIF = ROOT / "outputs_10epoch_obia_v3" / "PS_LandCover_OBIA_v3.tif"
OUT_RGB = ROOT / "rasters" / "sam_input_aoi_rgb.tif"
OUT_MASK = ROOT / "rasters" / "sam_input_aoi_dense_mask.tif"

CRS_3857 = CRS.from_epsg(3857)
CRS_UTM48S = CRS.from_epsg(32748)

DENSE_CLASS_ID = 5  # L1=Dense Vegetation
TARGET_PX_M = 3.0  # downsampled from native 0.3 m to PlanetScope-grain
                   # (3.0 m gives 100x fewer pixels; SAM still resolves canopy
                   #  boundaries better than LSMS over PlanetScope at the same px)


def main() -> None:
    # 1. Load AOI envelope in EPSG:32748
    aoi = gpd.read_file(CLIP_GPKG).to_crs(CRS_UTM48S)
    minx, miny, maxx, maxy = aoi.total_bounds
    # snap to integer metres + small buffer
    pad = 50.0
    minx, miny = np.floor(minx - pad), np.floor(miny - pad)
    maxx, maxy = np.ceil(maxx + pad), np.ceil(maxy + pad)
    print(f"AOI bbox (EPSG:32748): {minx:.0f} {miny:.0f} {maxx:.0f} {maxy:.0f}")
    print(f"  area: {(maxx - minx) * (maxy - miny) / 1e6:.1f} km²")

    # 2. Reproject AOI bbox to EPSG:3857 to clip the source.
    bbox_utm = gpd.GeoSeries.from_wkt(
        [f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"],
        crs=CRS_UTM48S,
    )
    bbox_3857 = bbox_utm.to_crs(CRS_3857).total_bounds
    print(f"AOI bbox (EPSG:3857): {bbox_3857}")

    # 3. Read World Imagery in EPSG:3857.
    print(f"\nReading {SRC_TIF.name} ...")
    with rasterio.open(SRC_TIF) as src:
        # The TIF reports LOCAL_CS but values match EPSG:3857. Force CRS.
        src_crs = CRS_3857
        win = from_bounds(*bbox_3857, transform=src.transform)
        win = win.round_offsets().round_lengths()
        print(f"  source window: col={int(win.col_off)}+{int(win.width)},"
              f" row={int(win.row_off)}+{int(win.height)}")
        # Read R, G, B (bands 1, 2, 3 — alpha band 4 dropped)
        rgb_u32 = src.read([1, 2, 3], window=win)
        print(f"  read shape (3, H, W): {rgb_u32.shape}, dtype {rgb_u32.dtype}")
        rgb = np.clip(rgb_u32, 0, 255).astype("uint8")
        # Window transform in source CRS
        src_transform_win = src.window_transform(win)

    # 4. Reproject to EPSG:32748 at TARGET_PX_M.
    width = int((maxx - minx) / TARGET_PX_M)
    height = int((maxy - miny) / TARGET_PX_M)
    dst_transform = rasterio.transform.from_origin(minx, maxy, TARGET_PX_M, TARGET_PX_M)
    print(f"\nReprojecting to EPSG:32748 grid: {width} x {height} px @ {TARGET_PX_M} m")

    dst_rgb = np.zeros((3, height, width), dtype="uint8")
    for i in range(3):
        reproject(
            source=rgb[i],
            destination=dst_rgb[i],
            src_transform=src_transform_win,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=CRS_UTM48S,
            resampling=Resampling.bilinear,
            num_threads=4,
        )
    print(f"  reprojected RGB: {dst_rgb.shape}, dtype {dst_rgb.dtype}")

    # 5. Build L1=Dense mask on the same grid.
    print(f"\nResampling L1 raster to SAM grid ...")
    dense_mask = np.zeros((height, width), dtype="uint8")
    with rasterio.open(L1_TIF) as l1:
        l1_arr_resampled = np.zeros((height, width), dtype=l1.dtypes[0])
        reproject(
            source=rasterio.band(l1, 1),
            destination=l1_arr_resampled,
            src_transform=l1.transform,
            src_crs=l1.crs,
            dst_transform=dst_transform,
            dst_crs=CRS_UTM48S,
            resampling=Resampling.nearest,
            num_threads=4,
        )
        dense_mask = (l1_arr_resampled == DENSE_CLASS_ID).astype("uint8")
    n_dense = int(dense_mask.sum())
    print(f"  Dense pixels: {n_dense:,} ({100 * n_dense / dense_mask.size:.1f}% of WZ extent)")
    print(f"  Dense area: {n_dense * TARGET_PX_M ** 2 / 10000:.1f} ha")

    # 6. (Skipped) — input masking removed. SAM works better on natural imagery.
    #    The L1=Dense mask is written as a sidecar for the downstream filter.

    # 7. Write outputs.
    print(f"\nWriting {OUT_RGB.name} ...")
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "width": width,
        "height": height,
        "transform": dst_transform,
        "crs": CRS_UTM48S,
        "compress": "DEFLATE",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
        "photometric": "RGB",
    }
    with rasterio.open(OUT_RGB, "w", **profile) as dst:
        dst.write(dst_rgb)
        dst.descriptions = ("R", "G", "B")

    profile_mask = profile.copy()
    profile_mask.update(count=1, photometric="MINISBLACK", nodata=0)
    profile_mask.pop("photometric", None)
    profile_mask["count"] = 1
    print(f"Writing {OUT_MASK.name} ...")
    with rasterio.open(OUT_MASK, "w", **profile_mask) as dst:
        dst.write(dense_mask, 1)

    rgb_size_mb = OUT_RGB.stat().st_size / 1e6
    mask_size_mb = OUT_MASK.stat().st_size / 1e6
    print(f"\n✓ Wrote {OUT_RGB.name} ({rgb_size_mb:.1f} MB)")
    print(f"✓ Wrote {OUT_MASK.name} ({mask_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
