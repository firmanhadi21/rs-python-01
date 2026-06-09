"""Run SamGeo (vit_b) on the preprocessed RGB COG.

Auto-downloads the SAM ViT-B checkpoint (≈ 360 MB) on first run.
Outputs raster + vector segments.

Tile size 1024 × 1024, overlap 256 px. On Apple Silicon MPS this completes
in ~15-30 min for the 1 Gpx WZ-clipped image.
"""

import os
import time
from pathlib import Path

import torch

ROOT = Path("/Users/firmanhadi/Works/Cisokan/2026/Apr26/process/CSK")
SAM_INPUT = ROOT / "rasters" / "sam_input_aoi_rgb.tif"
SAM_OUT_DIR = ROOT / "rasters" / "sam_output"
SAM_OUT_DIR.mkdir(parents=True, exist_ok=True)
SAM_RASTER = SAM_OUT_DIR / "sam_segments.tif"
SAM_VECTOR = SAM_OUT_DIR / "sam_segments.gpkg"
CHECKPOINT_DIR = ROOT / ".sam_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from samgeo import SamGeo

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"PyTorch device: {device}")
    print(f"Input:  {SAM_INPUT}")
    print(f"Output: {SAM_RASTER}")
    print(f"        {SAM_VECTOR}")

    # Relaxed thresholds → denser coverage (default 0.88 / 0.95 misses
    # homogeneous canopy at 3 m grain). With these settings we expect
    # ~3x more polygons but better whole-area coverage.
    sam_kwargs = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.78,
        "stability_score_thresh": 0.85,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 50,  # px (~450 m² at 3 m)
    }
    sam = SamGeo(
        model_type="vit_b",
        checkpoint=str(CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"),
        device=device,
        sam_kwargs=sam_kwargs,
    )

    t0 = time.time()
    sam.generate(
        source=str(SAM_INPUT),
        output=str(SAM_RASTER),
        batch=True,
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
    )
    elapsed = time.time() - t0
    print(f"\n✓ SAM raster written in {elapsed / 60:.1f} min")

    # Vectorise → GeoPackage
    print("Vectorising → GeoPackage ...")
    sam.tiff_to_gpkg(str(SAM_RASTER), str(SAM_VECTOR), simplify_tolerance=None)
    print(f"✓ {SAM_VECTOR}")


if __name__ == "__main__":
    # OTB poisons PROJ env vars; clear them.
    for k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA", "GDAL_DRIVER_PATH"):
        os.environ.pop(k, None)
    main()
