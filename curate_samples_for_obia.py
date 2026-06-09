#!/usr/bin/env python3
"""
Curate training samples for OBIA: one representative point per segment.

For each LSMS segment that contains training points:
  * All points agree on class  -> keep ONE point (closest to the segment's
    centroid, so its zonal signature best represents the segment).
  * Points disagree on class   -> export ALL points to `conflicts.gpkg` for
    manual review in QGIS. (Conflict segments usually reveal under-merging
    by the segmenter -- useful feedback loop.)

Inputs
------
    <project_root>/samples.gpkg
    <project_root>/outputs_10epoch_obia/lsms_labels.tif

Outputs
-------
    <project_root>/samples_curated.gpkg   -- clean, 1 point / segment
    <project_root>/conflicts.gpkg         -- to review in QGIS (if any)
    <project_root>/curation_report.csv    -- class counts before/after

Workflow
--------
1.  Run this script.
2.  Open conflicts.gpkg in QGIS alongside segments.gpkg and samples.gpkg.
    For each conflict segment: decide which class is correct, delete the
    other points (or reassign the class column), and save.
3.  Re-run this script with REVIEWED_CONFLICTS = True to merge the cleaned
    conflicts back into samples_curated.gpkg.
4.  Point planetscope_10epoch_obia.py at samples_curated.gpkg and run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from rasterio.transform import rowcol, xy
from scipy.ndimage import center_of_mass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("curate-obia")


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLES_GPKG = PROJECT_ROOT / "samples.gpkg"
SAMPLES_LAYER: Optional[str] = None
LABELS_TIF   = PROJECT_ROOT / "outputs_10epoch_obia" / "lsms_labels.tif"
CLASS_COL    = "class"

OUT_DIR      = PROJECT_ROOT
CURATED_GPKG = OUT_DIR / "samples_curated.gpkg"
CONFLICTS_GPKG = OUT_DIR / "conflicts.gpkg"
REPORT_CSV   = OUT_DIR / "curation_report.csv"

# Step 2 of the workflow: after you've edited conflicts.gpkg in QGIS, flip
# this to True and re-run to merge the cleaned conflicts into the curated file.
REVIEWED_CONFLICTS = False


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    log.info("Loading segmentation labels: %s", LABELS_TIF)
    lbl_da = rxr.open_rasterio(LABELS_TIF, masked=False).astype(np.uint32)
    if "band" in lbl_da.dims:
        lbl_da = lbl_da.isel(band=0).drop_vars("band", errors="ignore")
    labels = lbl_da.values
    transform = lbl_da.rio.transform()
    crs = lbl_da.rio.crs
    ny, nx = labels.shape
    log.info("  labels shape=%s, max_id=%d", labels.shape, int(labels.max()))

    log.info("Loading samples: %s", SAMPLES_GPKG)
    gdf = gpd.read_file(SAMPLES_GPKG, layer=SAMPLES_LAYER).to_crs(crs)
    if CLASS_COL not in gdf.columns:
        raise KeyError(f"{CLASS_COL!r} missing from samples columns: {list(gdf.columns)}")
    log.info("  %d points loaded", len(gdf))

    # Nearest-pixel segment lookup
    rows_f, cols_f = rowcol(transform, gdf.geometry.x.values, gdf.geometry.y.values,
                            op=round)
    rows_arr = np.clip(np.asarray(rows_f), 0, ny - 1).astype(int)
    cols_arr = np.clip(np.asarray(cols_f), 0, nx - 1).astype(int)
    gdf = gdf.assign(segment_id=labels[rows_arr, cols_arr].astype(np.int64))
    n_outside = int((gdf["segment_id"] == 0).sum())
    if n_outside:
        log.warning("%d points fall outside any segment (ID 0); dropping", n_outside)
    gdf = gdf[gdf["segment_id"] > 0].copy()

    # Per-segment centroids (pixel -> world). Only for segments that have points.
    unique_segs = np.sort(gdf["segment_id"].unique())
    log.info("Computing centroids for %d unique sampled segments", len(unique_segs))
    coms = center_of_mass(np.ones(labels.shape, dtype=bool),
                          labels, unique_segs.tolist())
    com = np.asarray(coms)  # (n_seg, 2) = (cy, cx) in pixel space
    cx_world, cy_world = xy(transform, com[:, 0].tolist(),
                             com[:, 1].tolist(), offset="center")
    cx_world = np.asarray(cx_world)
    cy_world = np.asarray(cy_world)
    cx_map = dict(zip(unique_segs.tolist(), cx_world.tolist()))
    cy_map = dict(zip(unique_segs.tolist(), cy_world.tolist()))

    gdf["seg_cx"] = gdf["segment_id"].map(cx_map)
    gdf["seg_cy"] = gdf["segment_id"].map(cy_map)
    gdf["dist_to_centroid"] = np.hypot(
        gdf.geometry.x.values - gdf["seg_cx"].to_numpy(),
        gdf.geometry.y.values - gdf["seg_cy"].to_numpy(),
    )
    gdf["n_points_in_segment"] = gdf["segment_id"].map(
        gdf.groupby("segment_id").size()
    )

    # Split: clean vs conflict
    per_seg_classes = gdf.groupby("segment_id")[CLASS_COL].nunique()
    clean_segs    = per_seg_classes[per_seg_classes == 1].index
    conflict_segs = per_seg_classes[per_seg_classes >  1].index
    log.info("Clean segments:    %d (1 class)", len(clean_segs))
    log.info("Conflict segments: %d (>1 class)", len(conflict_segs))

    # Clean: pick closest-to-centroid
    clean = gdf[gdf["segment_id"].isin(clean_segs)].copy()
    idx_keep = clean.groupby("segment_id")["dist_to_centroid"].idxmin()
    curated = clean.loc[idx_keep].drop(
        columns=["seg_cx", "seg_cy", "dist_to_centroid"]
    ).reset_index(drop=True)
    log.info("Curated clean segments: %d points kept from %d candidates",
             len(curated), len(clean))

    # Conflicts: export all conflict points
    conflicts = gdf[gdf["segment_id"].isin(conflict_segs)].copy()
    conflicts = conflicts.drop(columns=["seg_cx", "seg_cy", "dist_to_centroid"])
    conflicts["n_classes_in_segment"] = conflicts["segment_id"].map(per_seg_classes)

    # Optional: merge reviewed conflicts back in
    if REVIEWED_CONFLICTS and CONFLICTS_GPKG.exists():
        log.info("REVIEWED_CONFLICTS=True: merging cleaned %s", CONFLICTS_GPKG)
        reviewed = gpd.read_file(CONFLICTS_GPKG).to_crs(crs)
        # Recompute segment IDs against labels (coords may have shifted in QGIS)
        rr, cc = rowcol(transform, reviewed.geometry.x.values,
                         reviewed.geometry.y.values, op=round)
        rr = np.clip(np.asarray(rr), 0, ny - 1).astype(int)
        cc = np.clip(np.asarray(cc), 0, nx - 1).astype(int)
        reviewed["segment_id"] = labels[rr, cc].astype(np.int64)
        reviewed = reviewed[reviewed["segment_id"] > 0]
        # Re-check: any remaining multi-class segment is still a conflict
        r_per_seg = reviewed.groupby("segment_id")[CLASS_COL].nunique()
        r_clean = r_per_seg[r_per_seg == 1].index
        r_conf  = r_per_seg[r_per_seg >  1].index
        if len(r_conf):
            log.warning("Reviewed file still has %d conflict segments; skipping those",
                        len(r_conf))
        # Keep one point per reviewed segment (first occurrence)
        reviewed_clean = (reviewed[reviewed["segment_id"].isin(r_clean)]
                          .drop_duplicates(subset="segment_id", keep="first")
                          .reset_index(drop=True))
        # Drop any columns the reviewed file added that aren't in curated
        extra = set(reviewed_clean.columns) - set(curated.columns)
        reviewed_clean = reviewed_clean.drop(columns=list(extra), errors="ignore")
        # Remove segments already in curated (shouldn't overlap, but safety)
        reviewed_clean = reviewed_clean[
            ~reviewed_clean["segment_id"].isin(curated["segment_id"])
        ]
        log.info("Merging %d reviewed clean segments", len(reviewed_clean))
        curated = pd.concat([curated, reviewed_clean], ignore_index=True)
        curated = gpd.GeoDataFrame(curated, geometry="geometry", crs=crs)

    # Report
    before_ct = gdf[CLASS_COL].value_counts().sort_index().rename("before")
    after_ct  = curated[CLASS_COL].value_counts().sort_index().rename("after_curation")
    report = pd.concat([before_ct, after_ct], axis=1).fillna(0).astype(int)
    report.index.name = "class"
    report.to_csv(REPORT_CSV)
    log.info("Class counts (before vs after curation):\n%s", report.to_string())

    # Write outputs (overwrite)
    for path in (CURATED_GPKG, CONFLICTS_GPKG):
        if path.exists():
            path.unlink()
    curated.to_file(CURATED_GPKG, layer="samples_curated", driver="GPKG")
    log.info("Wrote %s  (%d points)", CURATED_GPKG, len(curated))
    if len(conflicts):
        conflicts.to_file(CONFLICTS_GPKG, layer="conflicts", driver="GPKG")
        log.info("Wrote %s  (%d points in %d conflict segments)",
                 CONFLICTS_GPKG, len(conflicts), len(conflict_segs))
        log.info("Open in QGIS to resolve: delete wrong-class points, save, "
                 "then re-run with REVIEWED_CONFLICTS=True.")
    else:
        log.info("No conflicts -- samples_curated.gpkg is your final training set.")


if __name__ == "__main__":
    main()
