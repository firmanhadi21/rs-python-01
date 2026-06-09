#!/usr/bin/env python3
"""
Object-based (OBIA) land-cover classification for the 10-epoch PlanetScope
stack -- EXTENDED feature set (v3).

v3 vs v2
--------
Keeps the v2 LSMS segmentation, zonal mean+std, 70/30 holdout + 5-fold CV,
full-raster LUT prediction, and full-raster / top-N export. DIFFERS FROM v2 on
training-row construction: reverts to the v1 scheme (majority vote per segment,
ties discarded, no sample weighting). v2's point-level expansion with weights
introduced cross-split label conflicts and caused the v2 regression from v1's
~84% OA to ~60% OA; v3 does not reproduce that bug.

v3 ADDS on top of v2:

  1. Extra per-epoch spectral indices (SuperDove-aware, red-edge aware)
     NDRE, CIre, RENDVI, SAVI, MSAVI2, OSAVI, GNDVI, ARVI, VARI, BSI
  2. Extra temporal descriptors
     p10 / p50 / p90 percentiles per index, harmonic fit (amplitude /
     phase / offset) across the 10 epochs, year-over-year march deltas
  3. Topographic covariates (optional -- skipped if DEM missing)
     slope, aspect (sin, cos), TPI, TRI
  4. Segment-only texture (pure numpy, fast)
     histogram entropy of NDVI + NIR, zonal range and CV of NDVI, zonal
     mean absolute deviation of NDVI
  5. Segment-only shape / geometry
     area, perimeter, compactness, bbox elongation, fill ratio

Reuses the existing v1/v2 segmentation (lsms_labels.tif) if present; set
`force_resegment=True` to rerun OTB from scratch. Reads the 138 cached pixel
features produced by `planetscope_10epoch_local.py`; if any are missing, the
script will raise with a clear error.

Band mapping note
-----------------
The legacy pixel script (planetscope_10epoch_local.py) treats
`b2=Blue, b3=Green, b4=Red, b6=? (pseudo-SWIR), b8=NIR`. v3 keeps those legacy
bands intact (they already live in the pixel cache) AND declares additional
SuperDove bands for the new indices via CONFIG.bands:
   coastal_blue=1, blue=2, green_i=3, green=4, yellow=5, red=6,
   red_edge=7, nir=8
The new indices follow the SuperDove mapping; the legacy NDVI/NDBI/EVI/NDWI
are untouched. Both conventions are retained so the paper can disclose the
full feature provenance.

    python planetscope_10epoch_obia_v3.py

Prereq:
  1. planetscope_10epoch_local.py has been run (produces the 138-feature cache)
  2. Optionally, planetscope_10epoch_obia.py or planetscope_10epoch_obia_v2.py
     has been run at least once (produces lsms_labels.tif for reuse)
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ps10epoch-obia-v3")


# Project root resolves from the script's own location, so the pipeline runs
# wherever the CSK directory is checked out.
PROJECT_ROOT = Path(__file__).resolve().parent


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    # -------- Inputs (shared with the pixel script) --------
    raster_dir: Path = PROJECT_ROOT / "rasters"
    feature_cache: Path = PROJECT_ROOT / "outputs_10epoch" / "feature_cache"
    aoi_gpkg: Path = PROJECT_ROOT / "aoi_cisokan.gpkg"
    aoi_layer: Optional[str] = None
    samples_gpkg: Path = PROJECT_ROOT / "samples.gpkg"
    samples_layer: Optional[str] = None
    class_column: str = "class"

    # Optional DEM for topographic features. If missing, topo features are
    # skipped with a warning (pipeline still runs).
    dem_tif: Path = PROJECT_ROOT / "dem.tif"

    # -------- Optional Meta v2 canopy-height layers (high-res complement to
    # the legacy ETH 10 m canopy_height.tif) ---------------------------------
    # 4 single-band GeoTIFFs in EPSG:32748 at 3 m, downloaded by
    # download_meta_canopy_v2.py from Meta's Data-for-Good aggregated CHM
    # (`alsgedi_global_v6_float_epsg4326_v3_10deg`). Each file becomes two
    # zonal features (mean+std per segment).
    meta_v2_dir: Path = PROJECT_ROOT / "rasters" / "meta_v2"
    meta_v2_stats: List[str] = field(default_factory=lambda: [
        "avg", "stdev", "p95", "cover",
    ])

    # -------- Optional SAR layers (exported from GEE; see gee.js) --------
    # Multi-band GeoTIFFs in any CRS/resolution; we reproject_match to the
    # PlanetScope grid and compute zonal mean+std per segment per band.
    # Missing files are skipped with a warning; the pipeline still runs.
    s1_temporal_tif: Path = PROJECT_ROOT / "S1_temporal_features.tif"
    palsar_tif:      Path = PROJECT_ROOT / "PALSAR_features.tif"
    # Override band names if the GeoTIFF lacks usable descriptions. Length
    # must equal the number of bands in the file; leave as None to read
    # band descriptions / fall back to b1..bN.
    s1_temporal_band_names: Optional[List[str]] = None
    palsar_band_names:      Optional[List[str]] = None

    # Filename per epoch label. Must match the cache keys exactly.
    epoch_labels: List[str] = field(default_factory=lambda: [
        "march", "june", "aug", "sept", "jan25",
        "may25", "aug25", "sep25", "nov25", "mar26",
    ])
    epoch_files: Dict[str, str] = field(default_factory=lambda: {
        "march":  "CSK_032124.tif",
        "june":   "CSK_063024.tif",
        "aug":    "CSK_080524.tif",
        "sept":   "CSK_091624.tif",
        "jan25":  "CSK_011025.tif",
        "may25":  "CSK_050525.tif",
        "aug25":  "CSK_081525.tif",
        "sep25":  "CSK_090725.tif",
        "nov25":  "CSK_112425.tif",
        "mar26":  "CSK_031726.tif",
    })
    # Day-of-year for each epoch -- used for the harmonic fit. Approximate
    # seasonal positioning; does not need to be exact for feature extraction.
    epoch_doy: Dict[str, int] = field(default_factory=lambda: {
        "march":  81,   # 21 Mar 2024
        "june":  182,   # 30 Jun 2024
        "aug":   218,   #  5 Aug 2024
        "sept":  260,   # 16 Sep 2024
        "jan25":  10,   # 10 Jan 2025
        "may25": 125,   #  5 May 2025
        "aug25": 227,   # 15 Aug 2025
        "sep25": 250,   #  7 Sep 2025
        "nov25": 328,   # 24 Nov 2025
        "mar26":  76,   # 17 Mar 2026
    })

    # -------- SuperDove band mapping (for NEW indices only) --------
    # Legacy indices in the pixel cache keep the original mapping; see module
    # docstring. These indices (1-based) are used exclusively by v3 for new
    # index computation.
    bands: Dict[str, int] = field(default_factory=lambda: {
        "coastal_blue": 1,
        "blue":         2,
        "green_i":      3,
        "green":        4,
        "yellow":       5,
        "red":          6,
        "red_edge":     7,
        "nir":          8,
    })

    # Which indices get full temporal treatment (percentiles + harmonic + YoY)
    # Using a curated subset keeps the feature count tractable. The pixel
    # cache already has max/min/std/amp for the legacy four.
    temporal_indices: List[str] = field(default_factory=lambda: [
        "NDVI", "EVI", "NDRE", "GNDVI",
    ])

    # -------- OTB install --------
    otb_home: Path = Path.home() / "OTB-8.1.2-Darwin64"

    # -------- Segmentation parameters (mean-shift; see OTB docs) --------
    # Tightened on 2026-05-02 from (5, 25.0, 100) -> finer segments that
    # better track river edges and crown-scale variation visible in the
    # Wayback overlay. Expected ~3-5x more segments (~200-300k vs 60k).
    seg_spatialr: int = 3
    seg_ranger: float = 12.0
    seg_minsize: int = 50
    seg_tile_px: int = 1024
    seg_maxiter: int = 100
    seg_thres: float = 0.1
    force_resegment: bool = False
    # Priority list for reusing an existing segmentation before running OTB.
    reuse_seg_candidates: List[Path] = field(default_factory=lambda: [
        PROJECT_ROOT / "outputs_10epoch_obia_v2",
        PROJECT_ROOT / "outputs_10epoch_obia",
    ])

    # -------- Percentile stretch (segmentation composite) --------
    stretch_low: float = 2.0
    stretch_high: float = 98.0

    # -------- Classifier --------
    rf_trees: int = 100
    rf_min_leaf: int = 1
    rf_bag_fraction: float = 0.5
    random_seed: int = 42
    train_fraction: float = 0.7
    n_cv_folds: int = 5
    top_n_features: int = 20

    # -------- Texture --------
    texture_levels: int = 32  # quantization levels for histogram entropy

    # -------- Haralick GLCM textures (temporal-median NIR + NDVI) -----------
    # Disabled by default after the 2026-05-03 ablation: Haralick on the
    # temporal-median PlanetScope didn't help in Cisokan -- avg per-feature
    # importance was 0.0004 (vs 0.0063 for Tree_Height, 0.0020 for PALSAR),
    # and the extra 32 noisy features slightly hurt L2 (CV OA 0.65 -> 0.61
    # with only 97 training rows). Cached files in feature_cache_v3 can be
    # reused instantly if re-enabled. See PIPELINE.md for diagnosis.
    haralick_enable: bool = False

    class_labels: Dict[int, str] = field(default_factory=lambda: {
        1: "Waterbody",
        2: "Paddy",
        3: "Built-up",
        4: "Others",
        5: "Dense Vegetation",
        6: "Sparse Vegetation",
        7: "Crops",
        # Bareland (was 8) dropped: in Cisokan it is operationally a transient
        # state, not a stable land cover. Existing class-8 sample points are
        # ignored at training time via the isin(class_labels) filter.
    })

    # -------- Hierarchical L2 (forest subtypes within Dense Vegetation) ------
    # Set l2_enable=False to skip L2 entirely and produce only the L1 raster
    # (useful while iterating on samples or for ablation).
    l2_enable: bool = True
    # L2 assignment method: "polygon" (deterministic overlay from a digitized
    # land-cover shapefile) or "random_forest" (the original ML L2 path).
    # Polygon mode is preferred where high-quality vector reference data
    # exists -- it sidesteps the Natural-vs-Production ML confusion entirely.
    l2_method: str = "random_forest"
    l2_class_column: str = "class_l2"
    l2_parent_class: int = 5  # L1 class to subdivide

    # -------- L2 polygon overlay (used when l2_method == "polygon") ----------
    l2_polygon_path: Path = Path(
        "/Users/firmanhadi/Works/Cisokan/2026/GIS_Deliverables/"
        "Landcover_digitized/"
        "Landcover (Agroforest, Production Forest, Natural Forest).shp"
    )
    l2_polygon_label_col: str = "Landcover"
    l2_polygon_label_to_final_id: Dict[str, int] = field(default_factory=lambda: {
        "Natural Forest":    5,
        "Production Forest": 6,
        "Agroforest":        7,
    })
    # Dense Vegetation segments that fall outside every L2 polygon get this
    # final id by default. 7 == Agroforest (intentional: in this AOI most
    # un-digitized dense patches are mixed-canopy / settlement-fringe).
    l2_polygon_default_final_id: int = 7
    # Mapping from string subtype label (in samples.gpkg.class_l2) to
    # the integer ID used in the final 10-class raster.
    l2_class_to_final_id: Dict[str, int] = field(default_factory=lambda: {
        "Natural":    5,
        "Production": 6,
        "Agroforest": 7,
    })
    # Remap of L1 IDs to final-class IDs for non-Dense classes. L1 class 5
    # is handled by L2 and intentionally absent here.
    l1_to_final_remap: Dict[int, int] = field(default_factory=lambda: {
        1: 1,  # Waterbody
        2: 2,  # Paddy
        3: 3,  # Built-up
        4: 4,  # Others
        6: 8,  # Sparse Vegetation -> 8 (was 6)
        7: 9,  # Crops             -> 9 (was 7)
    })
    final_class_labels: Dict[int, str] = field(default_factory=lambda: {
        1:  "Waterbody",
        2:  "Paddy",
        3:  "Built-up",
        4:  "Others",
        5:  "Natural Forest",
        6:  "Production Forest",
        7:  "Agroforest",
        8:  "Sparse Vegetation",
        9:  "Crops",
        # Final id 10 (YRF) only appears if cfg.yrf_apply=True.
    })

    # -------- YRF post-hoc rule ----------------------------------------------
    # Reclassify segments that look like young secondary regrowth: short canopy,
    # vigorous greenness. Applied to final_class in yrf_eligible_final_ids only.
    # 2026-05-03: disabled by default after the rule overfired (20% of AOI ->
    # YRF) on Cisokan's mid-NDVI mid-canopy mosaic. Re-enable only if explicit
    # YRF training samples are added.
    yrf_apply: bool = False
    yrf_final_id: int = 10
    yrf_canopy_min: float = 3.0    # m
    yrf_canopy_max: float = 10.0   # m
    yrf_ndvi_min: float = 0.6
    yrf_canopy_col: str = "tree_height_mean__mean"
    yrf_ndvi_col: str = "p50NDVI__mean"
    yrf_eligible_final_ids: Tuple[int, ...] = (7, 8)  # Agroforest, Sparse Veg

    # -------- Outputs --------
    out_dir: Path = PROJECT_ROOT / "outputs_10epoch_obia_v3"
    ext_cache: Path = PROJECT_ROOT / "outputs_10epoch_obia_v3" / "feature_cache_v3"

    # -------- Ablation switches (E1c / E1d / E1e: 2x2 design) ----------------
    # `epochs_subset` filters PlanetScope features to a chosen subset; sister
    # AOI-summary multi-epoch features and temporal percentiles/harmonic/YoY
    # are also skipped automatically.
    # `ps_only` drops every non-PlanetScope source: SAR (S1 + PALSAR), Meta v2
    # canopy stats, and ETH legacy tree-height. Together they give a 2x2
    # ablation matrix: {full vs PS-only} x {10-epoch vs 1-epoch}. Output goes
    # to a sibling directory with a suffix that encodes the configuration.
    epochs_subset: Optional[List[str]] = None
    ps_only: bool = False
    out_dir_suffix: str = ""  # "" = canonical; e.g. "_1ep_psonly" for ablation


CFG = Config()


# =============================================================================
# OTB SUBPROCESS HELPER
# =============================================================================
def run_otb(cfg: Config, app: str, args: List[str]) -> None:
    bin_path = cfg.otb_home / "bin" / app
    if not bin_path.exists():
        raise FileNotFoundError(f"OTB binary not found: {bin_path}")
    profile = cfg.otb_home / "otbenv.profile"
    cmd = [str(bin_path)] + list(args)
    if profile.exists():
        shell_cmd = (
            f"source {shlex.quote(str(profile))} && "
            + " ".join(shlex.quote(c) for c in cmd)
        )
        full = ["bash", "-c", shell_cmd]
    else:
        full = cmd
    log.info("OTB: %s %s", app, " ".join(args))
    subprocess.run(full, check=True)


# =============================================================================
# RASTER I/O HELPERS
# =============================================================================
def read_cached(cache: Path, name: str) -> xr.DataArray:
    path = cache / f"{name}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached feature: {path}")
    da = rxr.open_rasterio(path, masked=True).astype("float32")
    if "band" in da.dims:
        da = da.isel(band=0).drop_vars("band", errors="ignore")
    return da


def list_legacy_features(cfg: Config) -> List[str]:
    """The 138 features written by planetscope_10epoch_local.py.

    When `cfg.epochs_subset` is set, the per-epoch features are filtered to
    that subset and the AOI-summary multi-epoch features
    (max/min/std/amp × {NDVI,NDBI,EVI,NDWI}) are dropped, since they require
    >1 epoch to compute. tree_height_* are static and stay regardless.
    """
    names: List[str] = []
    epochs = cfg.epochs_subset if cfg.epochs_subset else cfg.epoch_labels
    for lbl in epochs:
        for bi in range(1, 9):
            names.append(f"{lbl}_b{bi}")
        for idx in ("NDVI", "NDWI", "NDBI", "EVI"):
            names.append(f"{lbl}_{idx}")
    if not cfg.epochs_subset:
        # AOI-summary features (multi-epoch derived) — only meaningful in full mode.
        for idx in ("NDVI", "NDBI", "EVI", "NDWI"):
            for stat in ("max", "min", "std", "amp"):
                names.append(f"{stat}{idx}")
    if not cfg.ps_only:
        # tree_height_* are from ETH (a non-PlanetScope source); drop in
        # ps_only mode so this experiment is genuinely PlanetScope-only.
        names += ["tree_height_mean", "tree_height_std"]
    missing = [n for n in names if not (cfg.feature_cache / f"{n}.tif").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cached features in {cfg.feature_cache}: "
            f"{missing[:5]}. Run planetscope_10epoch_local.py first.")
    return names


def open_planet_tif(path: Path, ref: Optional[xr.DataArray] = None) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Missing PlanetScope raster: {path}")
    da = rxr.open_rasterio(path, masked=True).astype("float32")
    if da.sizes["band"] < 8:
        raise ValueError(f"{path.name} has {da.sizes['band']} bands, expected 8")
    da = da.isel(band=slice(0, 8))
    if ref is not None:
        da = da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    return da


def write_cached(da: xr.DataArray, path: Path,
                 crs, transform) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = da.astype("float32")
    out.rio.write_crs(crs, inplace=True)
    out.rio.write_transform(transform, inplace=True)
    out.rio.to_raster(path, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")


# =============================================================================
# NEW PER-EPOCH INDICES (SuperDove-aware)
# =============================================================================
def compute_new_indices(arr: xr.DataArray, bands: Dict[str, int]
                        ) -> Dict[str, xr.DataArray]:
    """Red-edge, soil-adjusted, chlorophyll, and bare-soil indices.
    `arr` has dim ('band', y, x) with 8 bands in SuperDove order."""
    eps = 1e-10
    blue     = arr.sel(band=bands["blue"])
    green    = arr.sel(band=bands["green"])
    yellow   = arr.sel(band=bands["yellow"])
    red      = arr.sel(band=bands["red"])
    red_edge = arr.sel(band=bands["red_edge"])
    nir      = arr.sel(band=bands["nir"])

    out: Dict[str, xr.DataArray] = {}
    # Red-edge family
    out["NDRE"]   = (nir - red_edge) / (nir + red_edge + eps)
    out["CIre"]   = nir / (red_edge + eps) - 1.0
    out["RENDVI"] = (red_edge - red) / (red_edge + red + eps)

    # Soil-adjusted family (bare/sparse-tolerant)
    L_savi = 0.5
    out["SAVI"]   = ((nir - red) * (1.0 + L_savi)) / (nir + red + L_savi + eps)
    out["OSAVI"]  = (nir - red) / (nir + red + 0.16 + eps)
    msavi2_num = (2.0 * nir + 1.0)
    msavi2_rad = msavi2_num * msavi2_num - 8.0 * (nir - red)
    msavi2_rad = np.maximum(msavi2_rad, 0.0)
    out["MSAVI2"] = 0.5 * (msavi2_num - np.sqrt(msavi2_rad))

    # Chlorophyll / vigour family
    out["GNDVI"] = (nir - green) / (nir + green + eps)
    out["ARVI"]  = (nir - (2.0 * red - blue)) / (nir + (2.0 * red - blue) + eps)
    out["VARI"]  = (green - red) / (green + red - blue + eps)

    # Bare Soil Index (PlanetScope SuperDove -- no SWIR; Yellow substitutes)
    out["BSI"] = ((yellow + red) - (nir + blue)) / ((yellow + red) + (nir + blue) + eps)

    return out


def build_new_index_cache(cfg: Config) -> Tuple[List[str], xr.DataArray]:
    """Compute the 10 new per-epoch indices (10 epochs x 10 indices = 100
    features) and write them to the v3 extended cache. Returns the list of
    names and a reference DataArray aligned to the pixel cache grid.

    When `cfg.epochs_subset` is set, only that subset of epoch names are
    returned in the feature list (the cache is still iterated over all
    epochs that have not yet been computed, since cache hits are free)."""
    new_index_names = ["NDRE", "CIre", "RENDVI",
                       "SAVI", "OSAVI", "MSAVI2",
                       "GNDVI", "ARVI", "VARI", "BSI"]
    feature_names: List[str] = []
    return_epochs = set(cfg.epochs_subset) if cfg.epochs_subset else set(cfg.epoch_labels)

    # Use a cached NDVI tif as the reference grid (legacy cache). This keeps
    # the v3 extended cache pixel-aligned with the pixel cache.
    ref = read_cached(cfg.feature_cache, f"{cfg.epoch_labels[0]}_NDVI")
    ref_crs = ref.rio.crs
    ref_transform = ref.rio.transform()

    cfg.ext_cache.mkdir(parents=True, exist_ok=True)
    for label in cfg.epoch_labels:
        raw_path = cfg.raster_dir / cfg.epoch_files[label]
        missing_here = [
            n for n in new_index_names
            if not (cfg.ext_cache / f"{label}_{n}.tif").exists()
        ]
        if not missing_here:
            if label in return_epochs:
                feature_names.extend(f"{label}_{n}" for n in new_index_names)
            continue
        log.info("  Computing new indices for epoch %s  (%s)", label,
                 raw_path.name)
        arr = open_planet_tif(raw_path, ref=ref)
        idx_map = compute_new_indices(arr, cfg.bands)
        for n, da in idx_map.items():
            name = f"{label}_{n}"
            write_cached(da, cfg.ext_cache / f"{name}.tif",
                         ref_crs, ref_transform)
            if label in return_epochs:
                feature_names.append(name)
        del arr, idx_map
    log.info("New per-epoch indices returned: %d (%d epochs x %d indices)",
             len(feature_names), len(return_epochs), len(new_index_names))
    return feature_names, ref


# =============================================================================
# NEW TEMPORAL FEATURES (percentiles, harmonic, YoY)
# =============================================================================
def _stack_index_across_epochs(cfg: Config, idx_name: str
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (stack[T,Y,X], doys[T]) for an index across the 10 epochs.
    Reads from legacy cache for NDVI/NDBI/EVI/NDWI, else from ext cache."""
    legacy = {"NDVI", "NDBI", "EVI", "NDWI"}
    slabs = []
    doys = []
    for label in cfg.epoch_labels:
        cache = cfg.feature_cache if idx_name in legacy else cfg.ext_cache
        slabs.append(read_cached(cache, f"{label}_{idx_name}").values.astype("float32"))
        doys.append(cfg.epoch_doy[label])
    return np.stack(slabs, axis=0), np.asarray(doys, dtype="float32")


def compute_temporal_features(cfg: Config, ref: xr.DataArray) -> List[str]:
    """Compute percentiles, harmonic fit and YoY march deltas for the curated
    index subset. Writes to the extended cache. Returns the list of names.

    When `cfg.epochs_subset` is set, the temporal features are skipped
    entirely (they require >1 epoch by definition; this is the point of the
    single-epoch ablation).
    """
    if cfg.epochs_subset is not None:
        log.info("Temporal features skipped — single-epoch subset active "
                 "(%s)", cfg.epochs_subset)
        return []

    names: List[str] = []
    ref_crs = ref.rio.crs
    ref_transform = ref.rio.transform()

    for idx_name in cfg.temporal_indices:
        stack, doys = _stack_index_across_epochs(cfg, idx_name)

        # 1) Percentiles across epochs (per pixel)
        for q, label in [(10, "p10"), (50, "p50"), (90, "p90")]:
            name = f"{label}{idx_name}"
            out_path = cfg.ext_cache / f"{name}.tif"
            if not out_path.exists():
                p = np.nanpercentile(stack, q, axis=0).astype("float32")
                da = xr.DataArray(p, dims=("y", "x"),
                                  coords={"y": ref["y"], "x": ref["x"]})
                write_cached(da, out_path, ref_crs, ref_transform)
            names.append(name)
        log.info("  %-6s percentiles written (p10, p50, p90)", idx_name)

        # 2) Harmonic fit y = a0 + a1*cos(2pi*doy/365) + b1*sin(2pi*doy/365)
        harm_names = [f"harmAmp{idx_name}", f"harmPhase{idx_name}",
                      f"harmOffset{idx_name}"]
        harm_paths = [cfg.ext_cache / f"{n}.tif" for n in harm_names]
        if not all(p.exists() for p in harm_paths):
            theta = 2.0 * np.pi * doys / 365.25
            cos = np.cos(theta).astype("float32")
            sin = np.sin(theta).astype("float32")
            # Solve y = X * beta per pixel via normal equations (T=10 is fine)
            # X = [1, cos, sin], shape (T, 3)
            T, Y, X = stack.shape
            flat = stack.reshape(T, -1).astype("float32")
            # Drop epochs where a pixel is NaN by simple mean-imputation for
            # the harmonic (rare since the pixel script fills NaN earlier).
            nan_mask = ~np.isfinite(flat)
            if nan_mask.any():
                col_means = np.nanmean(flat, axis=0, keepdims=True)
                flat = np.where(nan_mask, np.broadcast_to(col_means, flat.shape),
                                flat)
            X_mat = np.stack([np.ones_like(cos), cos, sin], axis=1)  # (T,3)
            # beta = (X^T X)^-1 X^T Y, shape (3, pixels)
            gram_inv = np.linalg.inv(X_mat.T @ X_mat)
            beta = gram_inv @ (X_mat.T @ flat)
            a0 = beta[0].reshape(Y, X).astype("float32")
            a1 = beta[1].reshape(Y, X).astype("float32")
            b1 = beta[2].reshape(Y, X).astype("float32")
            amp   = np.hypot(a1, b1).astype("float32")
            phase = np.arctan2(b1, a1).astype("float32")
            for arr2d, path in zip((amp, phase, a0), harm_paths):
                da = xr.DataArray(arr2d, dims=("y", "x"),
                                  coords={"y": ref["y"], "x": ref["x"]})
                write_cached(da, path, ref_crs, ref_transform)
        names.extend(harm_names)
        log.info("  %-6s harmonic fit written (amp, phase, offset)", idx_name)

        # 3) YoY delta march 2024 -> march 2026 (present for all curated indices
        #    since "march" and "mar26" are both in the epoch list)
        yoy_name = f"yoy{idx_name}_mar26_mar24"
        yoy_path = cfg.ext_cache / f"{yoy_name}.tif"
        if not yoy_path.exists():
            i_mar24 = cfg.epoch_labels.index("march")
            i_mar26 = cfg.epoch_labels.index("mar26")
            delta = (stack[i_mar26] - stack[i_mar24]).astype("float32")
            da = xr.DataArray(delta, dims=("y", "x"),
                              coords={"y": ref["y"], "x": ref["x"]})
            write_cached(da, yoy_path, ref_crs, ref_transform)
        names.append(yoy_name)
        log.info("  %-6s YoY march delta written", idx_name)

    return names


# =============================================================================
# TOPOGRAPHIC FEATURES (optional)
# =============================================================================
def compute_topo_features(cfg: Config, ref: xr.DataArray) -> List[str]:
    if not cfg.dem_tif.exists():
        log.warning("DEM not found at %s -- skipping topographic features",
                    cfg.dem_tif)
        return []
    names = ["slope", "aspect_sin", "aspect_cos", "tpi", "tri"]
    if all((cfg.ext_cache / f"{n}.tif").exists() for n in names):
        log.info("Topographic features already cached")
        return names

    log.info("Computing topographic features from %s", cfg.dem_tif)
    dem = rxr.open_rasterio(cfg.dem_tif, masked=True).astype("float32")
    if "band" in dem.dims:
        dem = dem.isel(band=0).drop_vars("band", errors="ignore")
    dem = dem.rio.reproject_match(ref, resampling=Resampling.bilinear)
    dem_np = dem.values.astype("float32")
    dem_np = np.where(np.isfinite(dem_np), dem_np, np.nanmedian(dem_np))

    px = abs(float(ref.rio.resolution()[0]))
    # Central-difference gradients (meters per meter)
    gy, gx = np.gradient(dem_np, px, px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    aspect = np.arctan2(-gx, gy)  # radians; 0 = north
    aspect_sin = np.sin(aspect).astype("float32")
    aspect_cos = np.cos(aspect).astype("float32")

    # TPI = elev - mean(elev in 3x3 neighborhood)
    k = np.ones((3, 3), dtype="float32")
    neigh_mean = ndimage.convolve(dem_np, k / k.sum(), mode="reflect")
    tpi = (dem_np - neigh_mean).astype("float32")

    # TRI (Riley et al. 1999): mean absolute difference with 8 neighbors
    pad = np.pad(dem_np, 1, mode="edge")
    tri = np.zeros_like(dem_np, dtype="float32")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            tri += np.abs(dem_np - pad[1+dy:1+dy+dem_np.shape[0],
                                        1+dx:1+dx+dem_np.shape[1]])
    tri = (tri / 8.0).astype("float32")

    ref_crs = ref.rio.crs
    ref_transform = ref.rio.transform()
    yx = {"y": ref["y"], "x": ref["x"]}
    for arr2d, name in zip(
        (slope, aspect_sin, aspect_cos, tpi, tri), names
    ):
        da = xr.DataArray(arr2d, dims=("y", "x"), coords=yx)
        write_cached(da, cfg.ext_cache / f"{name}.tif", ref_crs, ref_transform)
    log.info("Topographic features cached: %s", names)
    return names


# =============================================================================
# HARALICK GLCM TEXTURE  (temporal-median NIR + NDVI, OTB)
# =============================================================================
# Captures spatial pattern at the planting-row scale (3 m PlanetScope; offset
# of 1 px catches Indonesian timber-spacing of 3-4 m). Targets the L2
# Natural-vs-Production confusion: production stands have periodic row
# patterns that produce high ASM/energy and low contrast at the row offset;
# natural forest is irregular (high contrast, high entropy). v3's existing
# segment-level "texture" features are histogram-based and orientation-blind
# -- this is the spatial-pattern complement.
#
# OTB's "simple" texture mode produces 8 features per source band, in this
# order:
_HARALICK_SIMPLE_FEATURES: List[str] = [
    "energy", "entropy", "correlation", "idm",
    "inertia", "clushade", "cluprom", "hcorr",
]


def _percentile_stretch_to_uint8(arr: np.ndarray,
                                 lo_pct: float = 1.0,
                                 hi_pct: float = 99.0) -> np.ndarray:
    """Stretch a float array to uint8 [0, 255], using percentile clipping.
    NaN pixels become 0. Used to prepare an input for OTB Haralick."""
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype="uint8")
    lo, hi = np.percentile(arr[valid], [lo_pct, hi_pct])
    scale = max(hi - lo, 1e-6)
    out = np.clip((arr - lo) / scale, 0, 1) * 255.0
    out = np.where(valid, out, 0).astype("uint8")
    return out


def compute_haralick_features(cfg: Config, ref: xr.DataArray) -> List[str]:
    """Per-pixel Haralick GLCM textures from temporal-median NIR (b8) and
    temporal-median NDVI. 8 features per source band -> 16 new pixel-level
    features. Window radius 1 (3x3), offset (1, 1), 8 GLCM bins.

    Cached aggressively. First run: ~5-10 minutes for the OTB step plus the
    temporal-median build. Subsequent runs reuse the per-feature TIFs."""
    ref_crs = ref.rio.crs
    ref_transform = ref.rio.transform()
    feature_names: List[str] = []

    # Each source = (cache-name prefix, list of cached pixel features to median)
    sources: List[Tuple[str, List[str]]] = [
        ("NIR",  [f"{lbl}_b8"   for lbl in cfg.epoch_labels]),
        ("NDVI", [f"{lbl}_NDVI" for lbl in cfg.epoch_labels]),
    ]

    for src_name, epoch_files in sources:
        per_feat_paths = [cfg.ext_cache / f"har_{src_name}_{f}.tif"
                          for f in _HARALICK_SIMPLE_FEATURES]
        if all(p.exists() for p in per_feat_paths):
            log.info("Haralick(%s) features already cached", src_name)
            feature_names.extend(f"har_{src_name}_{f}"
                                 for f in _HARALICK_SIMPLE_FEATURES)
            continue

        median_path = cfg.ext_cache / f"har_{src_name}_temporal_median_u8.tif"
        if not median_path.exists():
            log.info("Building temporal-median %s for Haralick (10 epochs)",
                     src_name)
            stack = np.stack(
                [read_cached(cfg.feature_cache, name).values
                 for name in epoch_files],
                axis=0,
            ).astype("float32")
            med = np.nanmedian(stack, axis=0).astype("float32")
            del stack
            scaled = _percentile_stretch_to_uint8(med)
            del med
            da = xr.DataArray(scaled, dims=("y", "x"),
                              coords={"y": ref["y"], "x": ref["x"]})
            da.rio.write_crs(ref_crs, inplace=True)
            da.rio.write_transform(ref_transform, inplace=True)
            da.rio.to_raster(median_path, dtype="uint8", compress="deflate",
                             tiled=True, BIGTIFF="IF_SAFER")
            log.info("  wrote %s (%.1f MB)", median_path.name,
                     median_path.stat().st_size / 1e6)

        otb_out = cfg.ext_cache / f"har_{src_name}_w3_simple.tif"
        if not otb_out.exists():
            log.info("OTB Haralick(%s) on %s (window=3, offset=(1,1), 8 bins)",
                     src_name, median_path.name)
            run_otb(cfg, "otbcli_HaralickTextureExtraction", [
                "-in",                  str(median_path),
                "-channel",             "1",
                "-texture",             "simple",
                "-parameters.xrad",     "1",
                "-parameters.yrad",     "1",
                "-parameters.xoff",     "1",
                "-parameters.yoff",     "1",
                "-parameters.min",      "0",
                "-parameters.max",      "255",
                "-parameters.nbbin",    "8",
                "-out",                 str(otb_out), "float",
            ])

        # Split the 8-band OTB output into per-feature single-band TIFs so
        # the rest of v3 can read them by name like any other pixel feature.
        log.info("Splitting %s into 8 single-band features", otb_out.name)
        tex_da = rxr.open_rasterio(otb_out, masked=True).astype("float32")
        for i, fname in enumerate(_HARALICK_SIMPLE_FEATURES):
            band = tex_da.isel(band=i).drop_vars("band", errors="ignore")
            out_path = cfg.ext_cache / f"har_{src_name}_{fname}.tif"
            write_cached(band, out_path, ref_crs, ref_transform)
        del tex_da

        feature_names.extend(f"har_{src_name}_{f}"
                             for f in _HARALICK_SIMPLE_FEATURES)

    log.info("Haralick features ready: %d (%s)", len(feature_names),
             ", ".join(feature_names[:4]) + ", ...")
    return feature_names


# =============================================================================
# OPTIONAL META v2 CANOPY-HEIGHT FEATURES  (high-res complement to ETH)
# =============================================================================
def compute_meta_canopy_features(
    cfg: Config, ref: xr.DataArray, labels: np.ndarray, n_seg: int
) -> Dict[str, np.ndarray]:
    """Zonal mean+std for each Meta v2 canopy-height layer in cfg.meta_v2_dir.

    Each layer becomes two segment-level features:
      meta_canopy_<stat>__mean / __std

    Layers are reprojected onto the PlanetScope grid via bilinear resampling
    before zonal stats. Missing files are skipped with a warning; the pipeline
    still runs without them. Complements the legacy ETH 10 m
    `canopy_height.tif` (which feeds `tree_height_*` in the pixel cache);
    these are *additional* features, not a replacement.

    Returns empty under ps_only (Meta v2 is not a PlanetScope source).
    """
    if cfg.ps_only:
        log.info("Meta v2 canopy features skipped — ps_only mode active")
        return {}
    out: Dict[str, np.ndarray] = {}
    for stat in cfg.meta_v2_stats:
        path = cfg.meta_v2_dir / f"canopy_height_v2_{stat}.tif"
        if not path.exists():
            log.warning("Meta v2 canopy missing -- skipping %s (%s)",
                        stat, path)
            continue
        log.info("Adding Meta v2 canopy feature: %s", path.name)

        da = rxr.open_rasterio(path, masked=True).astype("float32")
        if "band" in da.dims:
            da = da.isel(band=0).drop_vars("band", errors="ignore")
        da_match = da.rio.reproject_match(ref, resampling=Resampling.bilinear)
        arr = da_match.values.astype("float32")

        feat = f"meta_canopy_{stat}"
        mean, std = zonal_mean_std(labels, arr, n_seg)
        out[f"{feat}__mean"] = mean[1:]
        out[f"{feat}__std"]  = std[1:]

    log.info("Meta v2 canopy features: %d", len(out))
    return out


# =============================================================================
# OPTIONAL SAR FEATURES  (Sentinel-1 temporal + PALSAR, exported from GEE)
# =============================================================================
def _band_names_from_da(da: xr.DataArray, fallback_prefix: str
                        ) -> List[str]:
    n = int(da.sizes["band"])
    descs = da.attrs.get("long_name")
    if isinstance(descs, (list, tuple)) and len(descs) == n:
        return [str(d).replace(" ", "_") for d in descs]
    if isinstance(descs, str) and n == 1:
        return [descs.replace(" ", "_")]
    return [f"{fallback_prefix}_b{i+1}" for i in range(n)]


def compute_sar_features(
    cfg: Config, ref: xr.DataArray, labels: np.ndarray, n_seg: int
) -> Dict[str, np.ndarray]:
    """Zonal mean+std for each band of the optional Sentinel-1 / PALSAR
    GeoTIFFs. Each band yields two columns: <prefix>_<band>__mean / __std.
    Missing files are skipped with a warning. Returns empty under ps_only
    (SAR is not a PlanetScope source)."""
    if cfg.ps_only:
        log.info("SAR features skipped — ps_only mode active")
        return {}
    out: Dict[str, np.ndarray] = {}
    sources = [
        ("S1",     cfg.s1_temporal_tif, cfg.s1_temporal_band_names),
        ("PALSAR", cfg.palsar_tif,      cfg.palsar_band_names),
    ]

    for prefix, path, override in sources:
        if not path.exists():
            log.warning("SAR file missing -- skipping %s features (%s)",
                        prefix, path)
            continue
        log.info("Adding %s SAR features from %s", prefix, path.name)

        da = rxr.open_rasterio(path, masked=True).astype("float32")
        if "band" not in da.dims:
            da = da.expand_dims("band")
        n_bands = int(da.sizes["band"])

        if override is not None:
            if len(override) != n_bands:
                raise ValueError(
                    f"{path.name}: file has {n_bands} bands but "
                    f"{len(override)} override names were given")
            band_names = list(override)
        else:
            band_names = _band_names_from_da(da, prefix)

        # Reproject the whole stack once, then iterate bands.
        da_match = da.rio.reproject_match(ref, resampling=Resampling.bilinear)

        for bi in range(n_bands):
            arr = da_match.isel(band=bi).values.astype("float32")
            feat = f"{prefix}_{band_names[bi]}"
            mean, std = zonal_mean_std(labels, arr, n_seg)
            out[f"{feat}__mean"] = mean[1:]
            out[f"{feat}__std"]  = std[1:]

        log.info("  %s: %d bands -> %d zonal features",
                 prefix, n_bands, n_bands * 2)

    return out


# =============================================================================
# SEGMENTATION COMPOSITE + OTB LSMS  (identical to v2)
# =============================================================================
def build_segmentation_composite(cfg: Config
                                 ) -> Tuple[Path, xr.DataArray, np.ndarray]:
    log.info("Building 4-band temporal-median composite for segmentation")
    indices = ("NDVI", "NDBI", "EVI", "NDWI")
    ref = read_cached(cfg.feature_cache, f"{cfg.epoch_labels[0]}_NDVI")
    ref_crs = ref.rio.crs
    ref_transform = ref.rio.transform()

    aoi = gpd.read_file(cfg.aoi_gpkg, layer=cfg.aoi_layer).to_crs(ref_crs)
    aoi_mask = rasterize(
        [(geom, 1) for geom in aoi.geometry],
        out_shape=(ref.sizes["y"], ref.sizes["x"]),
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    log.info("  AOI covers %d / %d pixels (%.1f%%)",
             aoi_mask.sum(), aoi_mask.size,
             100.0 * aoi_mask.sum() / aoi_mask.size)

    bands = []
    for idx in indices:
        stack = np.stack(
            [read_cached(cfg.feature_cache, f"{lbl}_{idx}").values
             for lbl in cfg.epoch_labels],
            axis=0,
        )
        med = np.nanmedian(stack, axis=0).astype("float32")
        valid = med[aoi_mask & np.isfinite(med)]
        lo, hi = np.percentile(valid, [cfg.stretch_low, cfg.stretch_high])
        scaled = np.clip((med - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
        scaled = np.where(aoi_mask & np.isfinite(med), scaled, 0.0).astype("float32")
        bands.append(scaled)
        log.info("  %s median stretch [%.3f, %.3f] -> 0-255", idx, lo, hi)

    composite = np.stack(bands, axis=0)
    comp_da = xr.DataArray(
        composite, dims=("band", "y", "x"),
        coords={"band": np.arange(1, 5), "y": ref["y"], "x": ref["x"]},
    )
    comp_da.rio.write_crs(ref_crs, inplace=True)
    comp_da.rio.write_transform(ref_transform, inplace=True)

    out_path = cfg.out_dir / "seg_composite.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp_da.rio.to_raster(out_path, dtype="float32",
                          compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
    log.info("  wrote %s  shape=%s", out_path, composite.shape)
    return out_path, ref, aoi_mask


def run_lsms_segmentation(cfg: Config, composite_tif: Path) -> Path:
    smoothed = cfg.out_dir / "lsms_smoothed.tif"
    spatial  = cfg.out_dir / "lsms_spatial.tif"
    raw_lbl  = cfg.out_dir / "lsms_labels_raw.tif"
    merged   = cfg.out_dir / "lsms_labels.tif"
    vec_gpkg = cfg.out_dir / "segments.gpkg"

    if merged.exists() and not cfg.force_resegment:
        log.info("Reusing existing segmentation: %s  (set force_resegment=True to rerun)",
                 merged)
        return merged

    run_otb(cfg, "otbcli_MeanShiftSmoothing", [
        "-in", str(composite_tif),
        "-fout", str(smoothed),
        "-foutpos", str(spatial),
        "-spatialr", str(cfg.seg_spatialr),
        "-ranger", str(cfg.seg_ranger),
        "-thres", str(cfg.seg_thres),
        "-maxiter", str(cfg.seg_maxiter),
        "-modesearch", "1",
    ])
    run_otb(cfg, "otbcli_LSMSSegmentation", [
        "-in", str(smoothed),
        "-inpos", str(spatial),
        "-out", str(raw_lbl), "uint32",
        "-spatialr", str(cfg.seg_spatialr),
        "-ranger", str(cfg.seg_ranger),
        "-minsize", "0",
        "-tilesizex", str(cfg.seg_tile_px),
        "-tilesizey", str(cfg.seg_tile_px),
    ])
    run_otb(cfg, "otbcli_LSMSSmallRegionsMerging", [
        "-in", str(smoothed),
        "-inseg", str(raw_lbl),
        "-out", str(merged), "uint32",
        "-minsize", str(cfg.seg_minsize),
        "-tilesizex", str(cfg.seg_tile_px),
        "-tilesizey", str(cfg.seg_tile_px),
    ])
    try:
        run_otb(cfg, "otbcli_LSMSVectorization", [
            "-in", str(composite_tif),
            "-inseg", str(merged),
            "-out", str(vec_gpkg),
            "-tilesizex", str(cfg.seg_tile_px),
            "-tilesizey", str(cfg.seg_tile_px),
        ])
    except subprocess.CalledProcessError as exc:
        log.warning("Vectorization failed (non-fatal): %s", exc)
    return merged


# =============================================================================
# ZONAL STATISTICS  (identical to v2)
# =============================================================================
def zonal_mean_std(labels: np.ndarray, feature: np.ndarray, n_seg: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    valid = (labels > 0) & np.isfinite(feature)
    lbl = labels[valid].astype(np.int64)
    val = feature[valid].astype(np.float64)
    count = np.bincount(lbl, minlength=n_seg + 1)
    s  = np.bincount(lbl, weights=val,         minlength=n_seg + 1)
    sq = np.bincount(lbl, weights=val * val,   minlength=n_seg + 1)
    safe = count.astype(np.float64)
    safe[safe == 0] = 1.0
    mean = s / safe
    var  = np.maximum(sq / safe - mean * mean, 0.0)
    std  = np.sqrt(var)
    mean[count == 0] = np.nan
    std [count == 0] = np.nan
    return mean.astype("float32"), std.astype("float32")


def build_segment_features_for_names(
    cfg: Config, labels: np.ndarray, n_seg: int,
    names: List[str], source_dirs: List[Path],
    include_std: bool = True,
) -> Dict[str, np.ndarray]:
    """Mean (+std) per segment for a list of pixel-feature names. Each name
    is looked up in the first existing source dir from source_dirs."""
    data: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names, start=1):
        feat_path = None
        for d in source_dirs:
            if (d / f"{name}.tif").exists():
                feat_path = d / f"{name}.tif"
                break
        if feat_path is None:
            raise FileNotFoundError(
                f"Feature {name} not found in any of: {source_dirs}")
        feat = rxr.open_rasterio(feat_path, masked=True).astype("float32")
        if "band" in feat.dims:
            feat = feat.isel(band=0).drop_vars("band", errors="ignore")
        feat_np = feat.values
        mean, std = zonal_mean_std(labels, feat_np, n_seg)
        data[f"{name}__mean"] = mean[1:]
        if include_std:
            data[f"{name}__std"] = std[1:]
        if i % 40 == 0 or i == len(names):
            log.info("  %4d / %d zonal-stat features done", i, len(names))
        del feat, feat_np
    return data


# =============================================================================
# SEGMENT-LEVEL TEXTURE  (pure numpy, O(n_pixels))
# =============================================================================
def _quantize_to_levels(feat: np.ndarray, valid: np.ndarray,
                        n_levels: int) -> np.ndarray:
    """Return uint8 quantization (0..n_levels-1) of feat over valid pixels;
    invalid pixels get 255 (ignored downstream)."""
    out = np.full(feat.shape, 255, dtype=np.uint8)
    if not valid.any():
        return out
    vals = feat[valid]
    lo, hi = np.nanpercentile(vals, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
    scaled = np.clip((feat[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    q = np.clip((scaled * (n_levels - 1)).round().astype(np.int32), 0, n_levels - 1)
    out[valid] = q.astype(np.uint8)
    return out


def segment_histogram_entropy(labels: np.ndarray, feat: np.ndarray,
                              n_seg: int, n_levels: int) -> np.ndarray:
    """Shannon entropy (nats) of the in-segment pixel-intensity histogram,
    per segment. Fast because it uses two bincounts: one for (seg, level)
    counts and one for segment totals."""
    valid = (labels > 0) & np.isfinite(feat)
    q = _quantize_to_levels(feat, valid, n_levels)
    lbl = labels[valid].astype(np.int64)
    qv  = q[valid].astype(np.int64)

    # Joint index = lbl * n_levels + qv
    joint = lbl * n_levels + qv
    joint_counts = np.bincount(joint, minlength=(n_seg + 1) * n_levels)
    joint_counts = joint_counts.reshape(n_seg + 1, n_levels)
    totals = joint_counts.sum(axis=1).astype(np.float64)
    safe = np.where(totals > 0, totals, 1.0)
    p = joint_counts / safe[:, None]
    # 0*log(0) := 0
    logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    entropy = (-(p * logp).sum(axis=1)).astype("float32")
    entropy[totals == 0] = np.nan
    return entropy[1:]


def segment_range_p90_p10(labels: np.ndarray, feat: np.ndarray,
                          n_seg: int) -> np.ndarray:
    """p90 - p10 per segment. Requires sorting pixels per segment, done via
    lexsort on (label, value) and boundary indexing; O(n_pix log n_pix)."""
    valid = (labels > 0) & np.isfinite(feat)
    lbl = labels[valid].astype(np.int64)
    val = feat[valid].astype(np.float32)
    if lbl.size == 0:
        return np.full(n_seg, np.nan, dtype="float32")
    order = np.lexsort((val, lbl))
    lbl_s, val_s = lbl[order], val[order]
    # Segment boundaries in the sorted array
    bounds = np.concatenate(([0],
                             np.where(np.diff(lbl_s) != 0)[0] + 1,
                             [lbl_s.size]))
    out = np.full(n_seg, np.nan, dtype="float32")
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seg_id = int(lbl_s[a])
        if seg_id <= 0 or seg_id > n_seg:
            continue
        n = b - a
        # Index into the segment's sorted values
        p10_idx = min(n - 1, int(np.floor(0.10 * (n - 1))))
        p90_idx = min(n - 1, int(np.ceil (0.90 * (n - 1))))
        out[seg_id - 1] = float(val_s[a + p90_idx] - val_s[a + p10_idx])
    return out


def segment_mad(labels: np.ndarray, feat: np.ndarray, n_seg: int,
                seg_mean: np.ndarray) -> np.ndarray:
    """Mean absolute deviation of pixels within each segment (per segment)."""
    valid = (labels > 0) & np.isfinite(feat)
    lbl = labels[valid].astype(np.int64)
    val = feat[valid].astype(np.float64)
    # seg_mean is 0-indexed (shape n_seg). Pad a zero at index 0 for seg 0.
    seg_mean_pad = np.concatenate(([0.0], seg_mean.astype(np.float64)))
    resid = np.abs(val - seg_mean_pad[lbl])
    count = np.bincount(lbl, minlength=n_seg + 1)
    s     = np.bincount(lbl, weights=resid, minlength=n_seg + 1)
    safe = count.astype(np.float64)
    safe[safe == 0] = 1.0
    mad = (s / safe).astype("float32")
    mad[count == 0] = np.nan
    return mad[1:]


def compute_texture_features(cfg: Config, labels: np.ndarray, n_seg: int
                             ) -> Dict[str, np.ndarray]:
    """6 segment-level texture features:
       entropyNDVI, entropyNIR, rangeNDVI, rangeNIR, cvNDVI, madNDVI.
    Uses the median NDVI / median NIR across epochs to avoid being dominated
    by any single date."""
    log.info("Computing segment-level texture features")
    # Temporal-median NDVI
    ndvi_stack = np.stack([
        read_cached(cfg.feature_cache, f"{lbl}_NDVI").values.astype("float32")
        for lbl in cfg.epoch_labels
    ], axis=0)
    ndvi_med = np.nanmedian(ndvi_stack, axis=0)
    del ndvi_stack
    # Temporal-median NIR (b8 in legacy mapping)
    nir_stack = np.stack([
        read_cached(cfg.feature_cache, f"{lbl}_b8").values.astype("float32")
        for lbl in cfg.epoch_labels
    ], axis=0)
    nir_med = np.nanmedian(nir_stack, axis=0)
    del nir_stack

    out: Dict[str, np.ndarray] = {}
    out["entropyNDVI__seg"] = segment_histogram_entropy(
        labels, ndvi_med, n_seg, cfg.texture_levels)
    out["entropyNIR__seg"]  = segment_histogram_entropy(
        labels, nir_med,  n_seg, cfg.texture_levels)
    out["rangeNDVI__seg"]   = segment_range_p90_p10(labels, ndvi_med, n_seg)
    out["rangeNIR__seg"]    = segment_range_p90_p10(labels, nir_med,  n_seg)

    mean_ndvi, std_ndvi = zonal_mean_std(labels, ndvi_med, n_seg)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (std_ndvi[1:] / np.where(np.abs(mean_ndvi[1:]) > 1e-6,
                                      mean_ndvi[1:], np.nan)).astype("float32")
    out["cvNDVI__seg"] = cv
    out["madNDVI__seg"] = segment_mad(labels, ndvi_med, n_seg, mean_ndvi[1:])
    log.info("  6 texture features computed")
    return out


# =============================================================================
# SEGMENT-LEVEL SHAPE  (5 features, derived from the labels raster)
# =============================================================================
def compute_shape_features(labels: np.ndarray, n_seg: int, pixel_size_m: float
                           ) -> Dict[str, np.ndarray]:
    """5 segment-only shape features: area_m2, perimeter_m, compactness,
    elongation (bbox major/minor), fill_ratio (area / bbox_area)."""
    log.info("Computing segment-level shape features")
    sizes = np.bincount(labels[labels > 0].ravel(),
                        minlength=n_seg + 1)[1:]  # shape n_seg
    area_m2 = (sizes.astype("float32") * (pixel_size_m ** 2))

    # Perimeter in meters via 4-connected edge counting: each interior edge
    # between two pixels with different labels contributes 1 meter to EACH
    # adjacent pixel's perimeter. np.roll is used to pair neighbors; the
    # wrap-around row/col is masked out (we don't count image-boundary edges,
    # which would only affect AOI-boundary segments by at most one pixel
    # per column/row -- negligible for classification).
    perim_pix = np.zeros(n_seg + 1, dtype=np.int64)
    for axis in (0, 1):
        a = labels
        b = np.roll(labels, 1, axis=axis)
        diff = (a != b)
        if axis == 0:
            diff[0, :] = False   # mask wrap-around: top row vs last row
        else:
            diff[:, 0] = False   # mask wrap-around: left col vs last col
        contrib_a = diff & (a > 0)
        contrib_b = diff & (b > 0)
        perim_pix += np.bincount(a[contrib_a].astype(np.int64),
                                 minlength=n_seg + 1)
        perim_pix += np.bincount(b[contrib_b].astype(np.int64),
                                 minlength=n_seg + 1)
    perim_m = (perim_pix[1:].astype("float32") * pixel_size_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        compactness = (4.0 * np.pi * area_m2
                       / np.where(perim_m > 0, perim_m ** 2, np.nan)
                       ).astype("float32")

    # Bounding-box metrics via scipy.ndimage.find_objects
    log.info("  find_objects on %d segments", n_seg)
    bboxes = ndimage.find_objects(labels, max_label=n_seg)
    elong = np.full(n_seg, np.nan, dtype="float32")
    fill  = np.full(n_seg, np.nan, dtype="float32")
    for i, sl in enumerate(bboxes, start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        major = max(h, w); minor = max(1, min(h, w))
        elong[i - 1] = float(major) / float(minor)
        bbox_pix = h * w
        if bbox_pix > 0:
            fill[i - 1] = float(sizes[i - 1]) / float(bbox_pix)

    return {
        "area_m2__seg":     area_m2,
        "perimeter_m__seg": perim_m,
        "compactness__seg": compactness,
        "elongation__seg":  elong,
        "fill_ratio__seg":  fill,
    }


# =============================================================================
# TRAIN-POINT -> SEGMENT LABELING  (v1 scheme: majority vote, drop ties)
# =============================================================================
# v2 used point-level expansion with sample_weight, which introduces conflict
# rows (same features, different labels) when multiple classes fall in one
# segment. A random train/val split then leaks those conflicts across folds
# and tanks accuracy (v2 regressed from v1's 84% OA to ~60% OA).
# v3 reverts to the v1 scheme: majority class per segment, ties discarded,
# no sample weighting. Returns DataFrame(segment_id, class).
def assign_labels_to_segments(cfg: Config, labels: np.ndarray,
                              ref: xr.DataArray) -> pd.DataFrame:
    gdf = gpd.read_file(cfg.samples_gpkg, layer=cfg.samples_layer).to_crs(ref.rio.crs)
    gdf = gdf[gdf[cfg.class_column].isin(cfg.class_labels.keys())].copy()

    x_idx = ref.indexes["x"].get_indexer(gdf.geometry.x.values, method="nearest")
    y_idx = ref.indexes["y"].get_indexer(gdf.geometry.y.values, method="nearest")
    seg_ids = labels[y_idx, x_idx]

    pts = pd.DataFrame({
        "segment_id": seg_ids.astype(np.int64),
        "class":      gdf[cfg.class_column].to_numpy(dtype=np.int64),
    })
    outside = int((pts["segment_id"] == 0).sum())
    pts = pts[pts["segment_id"] > 0]

    tallies = pts.groupby(["segment_id", "class"]).size().unstack(fill_value=0)
    arr = tallies.to_numpy()
    sorted_desc = -np.sort(-arr, axis=1)
    top = sorted_desc[:, 0]
    second = sorted_desc[:, 1] if sorted_desc.shape[1] > 1 else np.zeros_like(top)
    keep = top > second  # strict majority; drops ties
    winners = tallies.columns.to_numpy()[np.argmax(arr, axis=1)]

    result = pd.DataFrame({
        "segment_id": tallies.index.to_numpy()[keep],
        "class":      winners[keep].astype(np.int64),
    })
    log.info(
        "Labeled %d segments from %d points  (outside-AOI: %d, ties dropped: %d)",
        len(result), len(gdf), outside, int((~keep).sum()),
    )
    return result


def assign_l2_labels_to_segments(cfg: Config, labels: np.ndarray,
                                 ref: xr.DataArray) -> pd.DataFrame:
    """Same majority-vote-per-segment scheme as L1, but reads class_l2 from
    the samples GPKG and only considers points where the L1 class equals
    cfg.l2_parent_class (Dense Vegetation). String subtype labels are mapped
    to the final integer IDs via cfg.l2_class_to_final_id.

    Returns DataFrame(segment_id, l2_final_id). May be empty if no L2 samples
    are available (no class_l2 column, or no Dense points labelled)."""
    gdf = gpd.read_file(cfg.samples_gpkg, layer=cfg.samples_layer).to_crs(ref.rio.crs)
    if cfg.l2_class_column not in gdf.columns:
        log.warning("samples GPKG has no '%s' column; skipping L2",
                    cfg.l2_class_column)
        return pd.DataFrame(columns=["segment_id", "l2_final_id"])

    parent_mask = (gdf[cfg.class_column] == cfg.l2_parent_class)
    sub_mask = parent_mask & gdf[cfg.l2_class_column].notna()
    gdf = gdf[sub_mask].copy()
    if gdf.empty:
        log.warning("No L2-labelled samples found (class==%d AND %s notna)",
                    cfg.l2_parent_class, cfg.l2_class_column)
        return pd.DataFrame(columns=["segment_id", "l2_final_id"])

    # Map string subtype to the integer ID used in the final raster.
    raw_labels = gdf[cfg.l2_class_column].astype(str).str.strip()
    unknown = sorted(set(raw_labels) - set(cfg.l2_class_to_final_id.keys()))
    if unknown:
        log.warning("Unknown L2 labels in samples: %s -- they will be dropped. "
                    "Allowed values: %s",
                    unknown, list(cfg.l2_class_to_final_id.keys()))
    sub_id = raw_labels.map(cfg.l2_class_to_final_id)
    gdf = gdf.loc[sub_id.notna()].copy()
    sub_id = sub_id.dropna().astype(np.int64)

    x_idx = ref.indexes["x"].get_indexer(gdf.geometry.x.values, method="nearest")
    y_idx = ref.indexes["y"].get_indexer(gdf.geometry.y.values, method="nearest")
    seg_ids = labels[y_idx, x_idx]

    pts = pd.DataFrame({
        "segment_id":  seg_ids.astype(np.int64),
        "l2_final_id": sub_id.to_numpy(dtype=np.int64),
    })
    outside = int((pts["segment_id"] == 0).sum())
    pts = pts[pts["segment_id"] > 0]

    tallies = pts.groupby(["segment_id", "l2_final_id"]).size().unstack(fill_value=0)
    if tallies.empty:
        return pd.DataFrame(columns=["segment_id", "l2_final_id"])
    arr = tallies.to_numpy()
    sorted_desc = -np.sort(-arr, axis=1)
    top = sorted_desc[:, 0]
    second = sorted_desc[:, 1] if sorted_desc.shape[1] > 1 else np.zeros_like(top)
    keep = top > second
    winners = tallies.columns.to_numpy()[np.argmax(arr, axis=1)]

    result = pd.DataFrame({
        "segment_id":   tallies.index.to_numpy()[keep],
        "l2_final_id": winners[keep].astype(np.int64),
    })
    log.info(
        "L2 labelled %d Dense segments from %d L2 points  "
        "(outside-AOI: %d, ties dropped: %d)",
        len(result), len(gdf), outside, int((~keep).sum()),
    )
    return result


def assign_l2_polygon_overlay(cfg: Config, labels: np.ndarray, n_seg: int,
                              ref: xr.DataArray) -> pd.Series:
    """Deterministic L2 assignment from a digitized land-cover shapefile.

    Reads cfg.l2_polygon_path, reprojects polygons to the segment grid, and
    rasterizes them as final-class IDs (5 = Natural, 6 = Production,
    7 = Agroforest) into a per-pixel grid (0 outside any polygon). For each
    segment, computes the majority polygon class over its in-AOI pixels.
    Segments with no in-polygon pixels get cfg.l2_polygon_default_final_id
    (Agroforest, by design -- the un-digitized dense vegetation in this AOI
    is mostly mixed-canopy / settlement-fringe).

    Returns Series indexed by segment_id (1..n_seg) with uint8 final IDs.
    Used in place of the L2 RandomForest when cfg.l2_method == "polygon".
    """
    log.info("L2 polygon overlay: reading %s", cfg.l2_polygon_path.name)
    if not cfg.l2_polygon_path.exists():
        raise FileNotFoundError(
            f"L2 polygon shapefile not found: {cfg.l2_polygon_path}")

    polys = gpd.read_file(cfg.l2_polygon_path).to_crs(ref.rio.crs)
    if cfg.l2_polygon_label_col not in polys.columns:
        raise ValueError(
            f"Column {cfg.l2_polygon_label_col!r} not in shapefile; "
            f"have {list(polys.columns)}")
    polys[cfg.l2_polygon_label_col] = (polys[cfg.l2_polygon_label_col]
                                       .astype(str).str.strip())
    polys["final_id"] = polys[cfg.l2_polygon_label_col].map(
        cfg.l2_polygon_label_to_final_id)
    unknown = sorted(set(polys.loc[polys["final_id"].isna(),
                                   cfg.l2_polygon_label_col].unique()))
    if unknown:
        log.warning("Unknown polygon labels (will be skipped): %s. "
                    "Allowed: %s", unknown,
                    list(cfg.l2_polygon_label_to_final_id.keys()))
    polys = polys.dropna(subset=["final_id"]).copy()
    polys["final_id"] = polys["final_id"].astype("uint8")
    log.info("  %d polygons across %d classes; total area %.2f km^2",
             len(polys), polys["final_id"].nunique(),
             polys.area.sum() / 1e6)

    # Rasterize polygons onto the segment grid: final ID per pixel,
    # 0 outside any polygon.
    poly_raster = rasterize(
        [(geom, fid) for geom, fid
         in zip(polys.geometry, polys["final_id"])],
        out_shape=labels.shape,
        transform=ref.rio.transform(),
        fill=0,
        dtype="uint8",
    )

    # Per-segment majority via combined bincount: counts[s, k] = number of
    # pixels in segment s belonging to polygon class k.
    n_classes = int(max(cfg.l2_polygon_label_to_final_id.values())) + 1
    valid = labels > 0
    seg_flat = labels[valid].astype(np.int64)
    pol_flat = poly_raster[valid].astype(np.int64)
    combined = seg_flat * n_classes + pol_flat
    counts = np.bincount(
        combined, minlength=(n_seg + 1) * n_classes
    ).reshape(n_seg + 1, n_classes)

    # Class 0 means "outside any polygon"; majority over classes 1..n-1 only.
    in_poly = counts[:, 1:]
    total_in_poly = in_poly.sum(axis=1)
    majority_id = (in_poly.argmax(axis=1) + 1).astype("uint8")
    final_ids = np.where(total_in_poly > 0,
                         majority_id,
                         cfg.l2_polygon_default_final_id).astype("uint8")

    # Drop segment 0 (out-of-AOI) and return the (1..n_seg) Series.
    result = pd.Series(final_ids[1:],
                       index=np.arange(1, n_seg + 1),
                       name="l2_final_id")
    inside = int((total_in_poly[1:] > 0).sum())
    log.info("  segments overlapping a polygon: %d / %d  "
             "(%d defaulted to id %d)",
             inside, n_seg, n_seg - inside, cfg.l2_polygon_default_final_id)
    return result


# =============================================================================
# CLASSIFIER + EVAL  (identical to v2)
# =============================================================================
def make_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        min_samples_leaf=cfg.rf_min_leaf,
        max_samples=cfg.rf_bag_fraction,
        bootstrap=True,
        max_features="sqrt",
        random_state=cfg.random_seed,
        n_jobs=-1,
    )


def evaluate_model(clf, X_val, y_val, cfg: Config,
                   sample_weight: Optional[np.ndarray] = None,
                   labels_dict: Optional[Dict[int, str]] = None):
    """Evaluate `clf` on (X_val, y_val). `labels_dict` overrides
    `cfg.class_labels` when set -- needed for L2 evaluation, where the
    class IDs map to the final-class label namespace, not the L1 one."""
    y_pred = clf.predict(X_val)
    label_map = labels_dict if labels_dict is not None else cfg.class_labels
    labels_order = sorted(label_map.keys())
    cm = confusion_matrix(y_val, y_pred, labels=labels_order,
                          sample_weight=sample_weight)
    oa = accuracy_score(y_val, y_pred, sample_weight=sample_weight)
    kappa = cohen_kappa_score(y_val, y_pred, labels=labels_order,
                              sample_weight=sample_weight)
    cm_df = pd.DataFrame(cm,
                         index=[label_map[c] for c in labels_order],
                         columns=[label_map[c] for c in labels_order])
    per_class = []
    for i, c in enumerate(labels_order):
        tp = cm[i, i]
        col_sum = cm[:, i].sum()
        row_sum = cm[i, :].sum()
        prec = tp / col_sum if col_sum else 0.0
        rec  = tp / row_sum if row_sum else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({
            "class": c,
            "class_name": label_map[c],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support_val": row_sum,
        })
    return ({"overall_accuracy": oa, "kappa": kappa},
            cm_df, pd.DataFrame(per_class))


def cross_validate(cfg: Config, X: np.ndarray, y: np.ndarray,
                   w: Optional[np.ndarray] = None) -> pd.DataFrame:
    kf = KFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.random_seed)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        clf = make_rf(cfg)
        w_tr = w[tr] if w is not None else None
        w_te = w[te] if w is not None else None
        clf.fit(X[tr], y[tr], sample_weight=w_tr)
        y_pred = clf.predict(X[te])
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[te], y_pred, sample_weight=w_te),
            "kappa":    cohen_kappa_score(y[te], y_pred, sample_weight=w_te),
            "train_size": len(tr),
            "test_size":  len(te),
        })
    return pd.DataFrame(rows)


# =============================================================================
# FEATURE GROUPING (v3-aware)
# =============================================================================
def group_importance(importance_df: pd.DataFrame, cfg: Config):
    feats = importance_df.copy()

    def base_of(name: str) -> str:
        # Drop the __mean / __std / __seg suffix for grouping purposes.
        for suffix in ("__mean", "__std", "__seg"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    topo_names = {"slope", "aspect_sin", "aspect_cos", "tpi", "tri"}
    shape_names = {"area_m2", "perimeter_m", "compactness",
                   "elongation", "fill_ratio"}
    tex_bases = {"entropyNDVI", "entropyNIR", "rangeNDVI", "rangeNIR",
                 "cvNDVI", "madNDVI"}
    harm_prefixes = ("harmAmp", "harmPhase", "harmOffset")
    pctile_prefixes = ("p10", "p50", "p90")

    def group_of(name: str) -> str:
        b = base_of(name)
        if b.startswith("har_NIR_"):                        return "Haralick_NIR"
        if b.startswith("har_NDVI_"):                       return "Haralick_NDVI"
        if b.startswith("S1_"):                             return "SAR_S1"
        if b.startswith("PALSAR_"):                         return "SAR_PALSAR"
        if b.startswith("meta_canopy_"):                    return "Meta_Canopy_v2"
        if "tree_height" in b:                              return "Tree_Height"
        if b in topo_names:                                 return "Topography"
        if b in shape_names:                                return "Shape"
        if b in tex_bases:                                  return "Texture"
        if b.startswith("yoy"):                             return "Temporal_YoY"
        if any(b.startswith(p) for p in harm_prefixes):     return "Temporal_Harmonic"
        if any(b.startswith(p) for p in pctile_prefixes):   return "Temporal_Percentiles"
        if any(b.startswith(s) and len(b) > len(s) and b[len(s):].isupper()
               for s in ("max", "min", "std", "amp")):
            return "Temporal_MinMaxStd"
        # Per-epoch indices
        for idx in ("NDRE", "CIre", "RENDVI", "SAVI", "OSAVI", "MSAVI2",
                    "GNDVI", "ARVI", "VARI", "BSI"):
            if idx in b:
                return f"{idx}_Indices"
        if "NDVI" in b:                                     return "NDVI_Indices"
        if "NDBI" in b:                                     return "NDBI_Indices"
        if "NDWI" in b:                                     return "NDWI_Indices"
        if "EVI"  in b:                                     return "EVI_Indices"
        if any(b.endswith(f"_b{i}") for i in range(1, 9)):  return "Spectral_Bands"
        return "Other"

    feats["group"] = feats["feature"].map(group_of)
    g = feats.groupby("group").agg(
        total_importance=("importance", "sum"),
        feature_count=("feature", "count"),
        avg_importance=("importance", "mean"),
    ).reset_index().sort_values("total_importance", ascending=False)

    sorted_labels = sorted(cfg.epoch_labels, key=len, reverse=True)

    def temporal_of(name: str) -> Optional[str]:
        b = base_of(name)
        for lbl in sorted_labels:
            if b.startswith(lbl + "_"):
                return lbl
        return None

    feats["epoch"] = feats["feature"].map(temporal_of)
    t = feats.dropna(subset=["epoch"]).groupby("epoch").agg(
        total_importance=("importance", "sum"),
        feature_count=("feature", "count"),
        avg_importance=("importance", "mean"),
    ).reset_index().sort_values("total_importance", ascending=False)
    return g, t


# =============================================================================
# PREDICTION BACK TO RASTER  (identical to v2, uses same LUT)
# =============================================================================
def predict_to_raster(clf, seg_features: pd.DataFrame, labels: np.ndarray,
                      ref: xr.DataArray, feature_cols: List[str]) -> xr.DataArray:
    X = seg_features[feature_cols]
    valid = X.dropna()
    preds = clf.predict(valid.to_numpy(dtype="float32")).astype("uint8")
    lut = np.zeros(int(labels.max()) + 1, dtype="uint8")
    lut[valid.index.to_numpy()] = preds
    out = lut[labels]
    out_da = xr.DataArray(out, dims=("y", "x"),
                          coords={"y": ref["y"], "x": ref["x"]},
                          name="classified")
    out_da.rio.write_crs(ref.rio.crs, inplace=True)
    out_da.rio.write_transform(ref.rio.transform(), inplace=True)
    return out_da


def save_geotiff(arr: xr.DataArray, path: Path,
                 dtype: str = "uint8", nodata: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing %s", path)
    arr.rio.to_raster(path, dtype=dtype, compress="deflate", nodata=nodata)


def predict_hierarchical(
    rf_l1, rf_l2, seg_features: pd.DataFrame, feature_cols: List[str],
    cfg: Config,
    l2_overrides: Optional[pd.Series] = None,
) -> pd.Series:
    """L1 prediction over all valid segments, L2 assignment over the Dense
    subset, then remap to final-class IDs. Optionally apply the YRF post-hoc
    rule. Returns a Series indexed by segment_id with final-class uint8.
    Segments dropped at zonal-stats time (NaN features) become 0 (nodata).

    L2 assignment can be one of:
      - `l2_overrides` Series (segment_id -> final_id): used directly for any
        segment whose L1 prediction is `cfg.l2_parent_class`. This is how
        polygon-overlay mode works.
      - `rf_l2` model: if no overrides, fit predict() on the same features.
      - Neither: Dense segments fall back to Natural Forest as placeholder.
    """
    X = seg_features[feature_cols]
    valid = X.dropna()
    seg_ids = valid.index.to_numpy()

    # L1
    l1_pred = rf_l1.predict(valid.to_numpy(dtype="float32")).astype(np.int64)
    final = pd.Series(0, index=seg_features.index, dtype="int64", name="final_class")
    # Remap non-Dense L1 IDs straight to final IDs
    remap = cfg.l1_to_final_remap
    for src_id, dst_id in remap.items():
        final.loc[seg_ids[l1_pred == src_id]] = dst_id

    # L2 on segments where L1 said class == l2_parent_class
    dense_mask = (l1_pred == cfg.l2_parent_class)
    dense_ids = seg_ids[dense_mask]

    if l2_overrides is not None and dense_ids.size:
        # Polygon-overlay mode (or any precomputed assignment).
        ov = l2_overrides.reindex(dense_ids)
        missing = int(ov.isna().sum())
        if missing:
            log.warning("L2 overrides missing for %d Dense segments; "
                        "defaulting to Natural Forest (final id 5)", missing)
            ov = ov.fillna(5)
        final.loc[dense_ids] = ov.astype(np.int64).to_numpy()
    elif rf_l2 is not None and dense_ids.size:
        X_dense = valid.loc[dense_ids].to_numpy(dtype="float32")
        l2_pred = rf_l2.predict(X_dense).astype(np.int64)
        final.loc[dense_ids] = l2_pred
    elif dense_ids.size:
        log.warning("No L2 model or overrides -- Dense segments shown as "
                    "final ID 5 (Natural Forest, placeholder).")
        final.loc[dense_ids] = cfg.l2_class_to_final_id.get("Natural", 5)

    # Post-hoc YRF rule
    if cfg.yrf_apply and cfg.yrf_canopy_col in seg_features.columns \
            and cfg.yrf_ndvi_col in seg_features.columns:
        cand = final.isin(cfg.yrf_eligible_final_ids)
        canopy = seg_features[cfg.yrf_canopy_col]
        ndvi = seg_features[cfg.yrf_ndvi_col]
        rule = (
            cand
            & canopy.between(cfg.yrf_canopy_min, cfg.yrf_canopy_max)
            & (ndvi > cfg.yrf_ndvi_min)
        )
        n_yrf = int(rule.sum())
        if n_yrf:
            final.loc[rule] = cfg.yrf_final_id
            log.info("YRF rule reclassified %d segments to final id %d "
                     "(canopy %.0f-%.0f m AND p50NDVI > %.2f)",
                     n_yrf, cfg.yrf_final_id, cfg.yrf_canopy_min,
                     cfg.yrf_canopy_max, cfg.yrf_ndvi_min)
        else:
            log.info("YRF rule found no eligible segments "
                     "(canopy %.0f-%.0f m AND p50NDVI > %.2f)",
                     cfg.yrf_canopy_min, cfg.yrf_canopy_max, cfg.yrf_ndvi_min)
    elif cfg.yrf_apply:
        log.warning("YRF rule skipped: missing column %s or %s in seg_df",
                    cfg.yrf_canopy_col, cfg.yrf_ndvi_col)

    return final.astype("uint8")


def final_classes_to_raster(final: pd.Series, labels: np.ndarray,
                            ref: xr.DataArray) -> xr.DataArray:
    lut = np.zeros(int(labels.max()) + 1, dtype="uint8")
    lut[final.index.to_numpy()] = final.to_numpy(dtype="uint8")
    out = lut[labels]
    out_da = xr.DataArray(out, dims=("y", "x"),
                          coords={"y": ref["y"], "x": ref["x"]},
                          name="final_class")
    out_da.rio.write_crs(ref.rio.crs, inplace=True)
    out_da.rio.write_transform(ref.rio.transform(), inplace=True)
    return out_da


# =============================================================================
# REUSE EXISTING SEGMENTATION
# =============================================================================
def try_reuse_segmentation(cfg: Config) -> None:
    """Symlink (or copy) lsms_labels.tif + seg_composite.tif from the first
    candidate dir that has them, into cfg.out_dir."""
    if cfg.force_resegment:
        return
    for src_dir in cfg.reuse_seg_candidates:
        if not (src_dir / "lsms_labels.tif").exists():
            continue
        for name in ("lsms_labels.tif", "seg_composite.tif",
                     "lsms_smoothed.tif", "lsms_spatial.tif"):
            src = src_dir / name
            dst = cfg.out_dir / name
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                    log.info("Reusing %s from %s", name, src_dir)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
                    log.info("Copied %s from %s", name, src_dir)
        return


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    cfg = CFG
    # If an ablation switch is set, redirect outputs to a sibling directory
    # so the canonical full-stack outputs are not overwritten. Auto-build the
    # suffix from the active switches.
    if cfg.epochs_subset is not None or cfg.ps_only:
        if not cfg.out_dir_suffix:
            parts = []
            if cfg.epochs_subset is not None:
                parts.append("_1ep" if len(cfg.epochs_subset) == 1
                             else f"_{len(cfg.epochs_subset)}ep")
            if cfg.ps_only:
                parts.append("_psonly")
            cfg.out_dir_suffix = "".join(parts)
        canonical_v3_dir = cfg.out_dir
        cfg.out_dir = cfg.out_dir.parent / (cfg.out_dir.name + cfg.out_dir_suffix)
        # Prepend the canonical v3 dir to the seg-reuse candidates so we pick up
        # the fine 316k segmentation rather than the coarse v1/v2 versions.
        if canonical_v3_dir not in cfg.reuse_seg_candidates:
            cfg.reuse_seg_candidates = [canonical_v3_dir] + list(cfg.reuse_seg_candidates)
        log.info("Ablation active: epochs_subset=%s ps_only=%s — outputs to %s",
                 cfg.epochs_subset, cfg.ps_only, cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.ext_cache.mkdir(parents=True, exist_ok=True)

    # ---- 0) Inventory of pixel-level features --------------------------------
    legacy_features = list_legacy_features(cfg)
    log.info("Legacy cached features: %d", len(legacy_features))

    new_index_features, ref = build_new_index_cache(cfg)
    log.info("New per-epoch index features: %d", len(new_index_features))

    temporal_features = compute_temporal_features(cfg, ref)
    log.info("New temporal features: %d", len(temporal_features))

    topo_features = compute_topo_features(cfg, ref)
    log.info("Topographic features: %d", len(topo_features))

    if cfg.haralick_enable:
        haralick_features = compute_haralick_features(cfg, ref)
        log.info("Haralick GLCM features: %d", len(haralick_features))
    else:
        haralick_features = []
        log.info("Haralick GLCM features: disabled (cfg.haralick_enable=False)")

    # Consolidated list of PIXEL features (these get zonal mean+std)
    pixel_features = (legacy_features + new_index_features
                      + temporal_features + topo_features
                      + haralick_features)
    log.info("Total pixel-level features: %d", len(pixel_features))

    # ---- 1) Segmentation: reuse if possible, else build from scratch --------
    try_reuse_segmentation(cfg)
    composite_tif, ref2, aoi_mask = build_segmentation_composite(cfg)
    # ref2 is identical to ref but re-derived here for aoi_mask alignment
    labels_tif = run_lsms_segmentation(cfg, composite_tif)

    # Load labels, mask out-of-AOI, compact IDs
    lbl_da = rxr.open_rasterio(labels_tif, masked=False).astype(np.uint32)
    if "band" in lbl_da.dims:
        lbl_da = lbl_da.isel(band=0).drop_vars("band", errors="ignore")
    labels = lbl_da.values.copy()
    labels[~aoi_mask] = 0
    log.info("Compacting label IDs (original max=%d)", int(labels.max()))
    unique_ids, inv = np.unique(labels, return_inverse=True)
    labels = inv.reshape(labels.shape).astype(np.uint32)
    n_seg = int(labels.max())
    n_aoi_px = int(aoi_mask.sum())
    avg_px_per_seg = n_aoi_px / max(n_seg, 1)
    log.info("Segment count: %d   avg %.1f pixels/segment",
             n_seg, avg_px_per_seg)
    if avg_px_per_seg < 20:
        raise RuntimeError(
            f"Segmentation collapsed: only {avg_px_per_seg:.1f} px/segment. "
            "Increase seg_ranger / seg_minsize / seg_spatialr, delete "
            f"{labels_tif.name} and rerun.")

    pixel_size_m = abs(float(ref2.rio.resolution()[0]))
    log.info("Pixel size: %.3f m", pixel_size_m)

    # ---- 2) Pixel-feature zonal mean+std -------------------------------------
    log.info("Computing zonal mean+std for %d pixel features",
             len(pixel_features))
    zonal: Dict[str, np.ndarray] = build_segment_features_for_names(
        cfg, labels, n_seg, pixel_features,
        source_dirs=[cfg.feature_cache, cfg.ext_cache],
        include_std=True,
    )

    # Boundary-sliver mask: segments with < 4 valid pixels in any feature
    sizes = np.bincount(labels[labels > 0].ravel(),
                        minlength=n_seg + 1)[1:]
    tiny = sizes < 4

    # ---- 3) Segment-only texture + shape -------------------------------------
    texture = compute_texture_features(cfg, labels, n_seg)
    shape = compute_shape_features(labels, n_seg, pixel_size_m)

    # ---- 3b) Optional SAR features (no-op if files missing) ------------------
    sar = compute_sar_features(cfg, ref2, labels, n_seg)

    # ---- 3c) Optional Meta v2 canopy features (no-op if files missing) -------
    meta_canopy = compute_meta_canopy_features(cfg, ref2, labels, n_seg)

    # ---- 4) Assemble segment DataFrame ---------------------------------------
    seg_df = pd.DataFrame(zonal)
    for name, arr in texture.items():
        seg_df[name] = arr
    for name, arr in shape.items():
        seg_df[name] = arr
    for name, arr in sar.items():
        seg_df[name] = arr
    for name, arr in meta_canopy.items():
        seg_df[name] = arr
    seg_df.index = np.arange(1, n_seg + 1)
    seg_df.index.name = "segment_id"
    if tiny.any():
        log.info("Dropping %d boundary-sliver segments (<4 in-AOI pixels)",
                 int(tiny.sum()))
        seg_df.loc[seg_df.index[tiny], :] = np.nan

    feature_cols = list(seg_df.columns)
    log.info("Total segment features: %d  "
             "(%d zonal + %d texture + %d shape + %d SAR + %d Meta canopy)",
             len(feature_cols), len(zonal), len(texture),
             len(shape), len(sar), len(meta_canopy))

    # Median-impute partial-NaN features. Tiny segments (set fully-NaN above)
    # are excluded from imputation so they continue to be dropped downstream.
    # Recovers segments that have a few stray NaN features -- e.g. 13 segments
    # in the 2026-05-03 run were dropped because sep25_b1 / nov25_b1 have a
    # ragged pixel edge in those two epochs. Median imputation is computed
    # over non-sliver segments only and applied per column.
    non_sliver = ~seg_df[feature_cols].isna().all(axis=1)
    n_partial_nan = int(seg_df.loc[non_sliver, feature_cols]
                        .isna().any(axis=1).sum())
    if n_partial_nan:
        medians = seg_df.loc[non_sliver, feature_cols].median()
        seg_df.loc[non_sliver, feature_cols] = (
            seg_df.loc[non_sliver, feature_cols].fillna(medians)
        )
        log.info("Median-imputed %d non-sliver segments with partial NaN "
                 "features (per-column median over the remaining %d segments)",
                 n_partial_nan, int(non_sliver.sum()))

    # ---- 5) Train-point -> segment labeling ----------------------------------
    labeled = assign_labels_to_segments(cfg, labels, ref2)
    labeled.to_csv(cfg.out_dir / "training_segments_labeled.csv", index=False)

    seg_lifted = seg_df.loc[labeled["segment_id"]].reset_index()
    train_df = seg_lifted.copy()
    train_df["class"] = labeled["class"].to_numpy()

    before = len(train_df)
    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(train_df) < before:
        log.warning("Dropped %d labeled rows with NaN features",
                    before - len(train_df))

    cls_counts = train_df["class"].value_counts().sort_index().astype(int)
    log.info("Class distribution (one row per segment):\n%s",
             cls_counts.to_string())
    cls_counts.rename("segment_count").to_csv(
        cfg.out_dir / "class_distribution.csv")

    # ---- 6) 70/30 holdout ----------------------------------------------------
    rng = np.random.default_rng(cfg.random_seed)
    mask = rng.random(len(train_df)) < cfg.train_fraction
    tr = train_df.loc[mask].reset_index(drop=True)
    va = train_df.loc[~mask].reset_index(drop=True)
    log.info("Train / validation segments: %d / %d", len(tr), len(va))

    X_tr = tr[feature_cols].to_numpy(dtype="float32")
    y_tr = tr["class"].to_numpy(dtype="int32")
    X_va = va[feature_cols].to_numpy(dtype="float32")
    y_va = va["class"].to_numpy(dtype="int32")

    log.info("Training RandomForest on all %d features", len(feature_cols))
    rf = make_rf(cfg)
    rf.fit(X_tr, y_tr)
    metrics_all, cm_all, per_class_all = evaluate_model(
        rf, X_va, y_va, cfg)
    log.info("OBIA-v3 full-feature: OA=%.4f  Kappa=%.4f",
             metrics_all["overall_accuracy"], metrics_all["kappa"])
    cm_all.to_csv(cfg.out_dir / "confusion_matrix_all.csv")
    per_class_all.to_csv(cfg.out_dir / "per_class_metrics_all.csv", index=False)

    # ---- 7) Feature importance + group analysis ------------------------------
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(cfg.out_dir / "feature_importance_obia_v3.csv", index=False)
    log.info("Top 15 features:\n%s", imp_df.head(15).to_string(index=False))

    group_df, temporal_df = group_importance(imp_df, cfg)
    group_df.to_csv(cfg.out_dir / "feature_group_analysis_obia_v3.csv", index=False)
    temporal_df.to_csv(cfg.out_dir / "temporal_group_analysis_obia_v3.csv",
                       index=False)

    # ---- 8) 5-fold CV on all features ----------------------------------------
    log.info("5-fold CV on all features")
    X_all = train_df[feature_cols].to_numpy(dtype="float32")
    y_all = train_df["class"].to_numpy(dtype="int32")
    cv_all = cross_validate(cfg, X_all, y_all)
    cv_all.to_csv(cfg.out_dir / "cv_results_obia_v3_all_features.csv",
                  index=False)
    log.info("CV(all):  OA=%.4f +/- %.4f   K=%.4f +/- %.4f",
             cv_all["accuracy"].mean(), cv_all["accuracy"].std(),
             cv_all["kappa"].mean(),    cv_all["kappa"].std())

    # ---- 9) Top-N subset -----------------------------------------------------
    top_n = imp_df.head(cfg.top_n_features)["feature"].tolist()
    log.info("Retraining on top %d features", cfg.top_n_features)
    rf_top = make_rf(cfg)
    rf_top.fit(tr[top_n].to_numpy(dtype="float32"), y_tr)
    metrics_top, cm_top, per_class_top = evaluate_model(
        rf_top, va[top_n].to_numpy(dtype="float32"), y_va, cfg)
    log.info("Top-%d  OA=%.4f  Kappa=%.4f",
             cfg.top_n_features, metrics_top["overall_accuracy"], metrics_top["kappa"])
    cm_top.to_csv(cfg.out_dir / f"confusion_matrix_top{cfg.top_n_features}.csv")
    per_class_top.to_csv(
        cfg.out_dir / f"per_class_metrics_top{cfg.top_n_features}.csv",
        index=False)
    cv_top = cross_validate(cfg, train_df[top_n].to_numpy(dtype="float32"),
                            y_all)
    cv_top.to_csv(cfg.out_dir / f"cv_results_obia_v3_top{cfg.top_n_features}.csv",
                  index=False)
    log.info("CV(top%d): OA=%.4f +/- %.4f   K=%.4f +/- %.4f",
             cfg.top_n_features,
             cv_top["accuracy"].mean(), cv_top["accuracy"].std(),
             cv_top["kappa"].mean(),    cv_top["kappa"].std())

    # ---- 10) Full-raster predictions via LUT ---------------------------------
    classified = predict_to_raster(rf, seg_df, labels, ref2, feature_cols)
    save_geotiff(classified, cfg.out_dir / "PS_LandCover_OBIA_v3.tif")
    classified_top = predict_to_raster(rf_top, seg_df, labels, ref2, top_n)
    save_geotiff(classified_top,
                 cfg.out_dir / f"PS_LandCover_OBIA_v3_Top{cfg.top_n_features}.tif")

    # ---- 11) L2 forest-subtype assignment ------------------------------------
    # L2-related outputs are suffixed when running in RF mode so the polygon
    # canonical outputs aren't overwritten. L1-only outputs are mode-agnostic
    # (same model fit, deterministic seed) so they stay unsuffixed.
    l2_out_suffix = "_RF" if cfg.l2_method == "random_forest" else ""

    rf_l2 = None
    l2_overrides: Optional[pd.Series] = None
    l2_metrics: Optional[Dict[str, float]] = None

    if cfg.l2_enable and cfg.l2_method == "polygon":
        l2_overrides = assign_l2_polygon_overlay(cfg, labels, n_seg, ref2)
        # Tabulate the per-class segment counts for the Dense subset only,
        # mirroring class_distribution_l2.csv from the RF path. This gives a
        # quick sanity check on how the polygon overlay distributed Dense
        # segments across Natural / Production / Agroforest.
        l2_dist = (l2_overrides
                   .value_counts()
                   .sort_index()
                   .rename("segment_count"))
        l2_dist.index.name = "l2_final_id"
        l2_dist.to_csv(cfg.out_dir / f"class_distribution_l2{l2_out_suffix}.csv")
        log.info("L2 polygon overlay distribution (all segments):\n%s",
                 l2_dist.to_string())
        l2_metrics = {
            "method":          "polygon",
            "polygon_path":    str(cfg.l2_polygon_path),
            "n_segments_assigned": int(len(l2_overrides)),
        }
    elif cfg.l2_enable and cfg.l2_method == "random_forest":
        l2_labeled = assign_l2_labels_to_segments(cfg, labels, ref2)
        if len(l2_labeled) >= 12:  # need enough for a 70/30 + 5-fold CV
            l2_seg_lifted = seg_df.loc[l2_labeled["segment_id"]].reset_index()
            l2_train_df = l2_seg_lifted.copy()
            l2_train_df["l2_final_id"] = l2_labeled["l2_final_id"].to_numpy()
            l2_train_df = l2_train_df.dropna(subset=feature_cols).reset_index(drop=True)
            log.info("L2 training rows: %d  (after dropna)", len(l2_train_df))

            l2_counts = l2_train_df["l2_final_id"].value_counts().sort_index()
            log.info("L2 class distribution (final ids):\n%s",
                     l2_counts.to_string())
            l2_counts.rename("segment_count").to_csv(
                cfg.out_dir / f"class_distribution_l2{l2_out_suffix}.csv")

            # 70/30 holdout (independent rng draw to avoid replicating L1 mask)
            rng_l2 = np.random.default_rng(cfg.random_seed + 1)
            mask_l2 = rng_l2.random(len(l2_train_df)) < cfg.train_fraction
            tr_l2 = l2_train_df.loc[mask_l2].reset_index(drop=True)
            va_l2 = l2_train_df.loc[~mask_l2].reset_index(drop=True)
            log.info("L2 train / val: %d / %d", len(tr_l2), len(va_l2))

            X_tr_l2 = tr_l2[feature_cols].to_numpy(dtype="float32")
            y_tr_l2 = tr_l2["l2_final_id"].to_numpy(dtype="int32")
            X_va_l2 = va_l2[feature_cols].to_numpy(dtype="float32")
            y_va_l2 = va_l2["l2_final_id"].to_numpy(dtype="int32")

            rf_l2 = make_rf(cfg)
            rf_l2.fit(X_tr_l2, y_tr_l2)
            l2_label_map = {v: cfg.final_class_labels[v]
                            for v in cfg.l2_class_to_final_id.values()
                            if v in cfg.final_class_labels}
            metrics_l2, cm_l2, per_class_l2 = evaluate_model(
                rf_l2, X_va_l2, y_va_l2, cfg, labels_dict=l2_label_map)
            log.info("L2 (forest subtypes): OA=%.4f  Kappa=%.4f",
                     metrics_l2["overall_accuracy"], metrics_l2["kappa"])
            cm_l2.to_csv(cfg.out_dir / f"confusion_matrix_l2{l2_out_suffix}.csv")
            per_class_l2.to_csv(
                cfg.out_dir / f"per_class_metrics_l2{l2_out_suffix}.csv",
                index=False)

            X_all_l2 = l2_train_df[feature_cols].to_numpy(dtype="float32")
            y_all_l2 = l2_train_df["l2_final_id"].to_numpy(dtype="int32")
            cv_l2 = cross_validate(cfg, X_all_l2, y_all_l2)
            cv_l2.to_csv(cfg.out_dir / f"cv_results_l2{l2_out_suffix}.csv",
                         index=False)
            log.info("L2 CV: OA=%.4f +/- %.4f  K=%.4f +/- %.4f",
                     cv_l2["accuracy"].mean(), cv_l2["accuracy"].std(),
                     cv_l2["kappa"].mean(),    cv_l2["kappa"].std())

            l2_metrics = {
                "method":           "random_forest",
                "n_training_rows":  int(len(l2_train_df)),
                "holdout_accuracy": float(metrics_l2["overall_accuracy"]),
                "holdout_kappa":    float(metrics_l2["kappa"]),
                "cv_accuracy_mean": float(cv_l2["accuracy"].mean()),
                "cv_accuracy_std":  float(cv_l2["accuracy"].std()),
                "cv_kappa_mean":    float(cv_l2["kappa"].mean()),
                "cv_kappa_std":     float(cv_l2["kappa"].std()),
            }
        else:
            log.warning("L2 disabled: only %d Dense segments labelled "
                        "(need >= 12). Run with more L2 samples to enable.",
                        len(l2_labeled))

    # ---- 12) Hierarchical final raster + YRF rule ----------------------------
    final_series = predict_hierarchical(rf, rf_l2, seg_df, feature_cols, cfg,
                                        l2_overrides=l2_overrides)
    final_da = final_classes_to_raster(final_series, labels, ref2)
    save_geotiff(final_da,
                 cfg.out_dir / f"PS_LandCover_OBIA_v3_Final{l2_out_suffix}.tif")

    # Final-class distribution over predicted pixels (within AOI only).
    in_aoi = labels > 0
    final_arr = final_da.values
    classes, counts = np.unique(final_arr[in_aoi], return_counts=True)
    final_dist = pd.DataFrame({
        "final_id": classes.astype(int),
        "name":     [cfg.final_class_labels.get(int(c), f"id_{c}")
                     for c in classes],
        "n_pixels": counts.astype(int),
        "pct_aoi":  100.0 * counts / counts.sum(),
    }).sort_values("final_id")
    final_dist.to_csv(
        cfg.out_dir / f"final_class_distribution{l2_out_suffix}.csv",
        index=False)
    log.info("Final-class distribution (pixels in AOI):\n%s",
             final_dist.to_string(index=False))

    # ---- 13) Summary ---------------------------------------------------------
    summary = {
        "n_segments": n_seg,
        "n_training_rows": int(len(train_df)),
        "n_unique_labeled_segments": int(train_df["segment_id"].nunique()),
        "n_pixel_features": len(pixel_features),
        "n_zonal_features": len(zonal),
        "n_texture_features": len(texture),
        "n_shape_features": len(shape),
        "n_features_total": len(feature_cols),
        "feature_group_counts": {
            "legacy_pixel_cache": len(legacy_features),
            "new_per_epoch_indices": len(new_index_features),
            "temporal_percentiles_harmonic_yoy": len(temporal_features),
            "topography": len(topo_features),
            "haralick_glcm": len(haralick_features),
            "texture_segment_only": len(texture),
            "shape_segment_only": len(shape),
            "meta_canopy_v2": len(meta_canopy),
        },
        "seg_params": {
            "spatialr": cfg.seg_spatialr,
            "ranger":   cfg.seg_ranger,
            "minsize":  cfg.seg_minsize,
        },
        "all_features": {
            "holdout_accuracy": float(metrics_all["overall_accuracy"]),
            "holdout_kappa":    float(metrics_all["kappa"]),
            "cv_accuracy_mean": float(cv_all["accuracy"].mean()),
            "cv_accuracy_std":  float(cv_all["accuracy"].std()),
            "cv_kappa_mean":    float(cv_all["kappa"].mean()),
            "cv_kappa_std":     float(cv_all["kappa"].std()),
        },
        f"top_{cfg.top_n_features}": {
            "holdout_accuracy": float(metrics_top["overall_accuracy"]),
            "holdout_kappa":    float(metrics_top["kappa"]),
            "cv_accuracy_mean": float(cv_top["accuracy"].mean()),
            "cv_accuracy_std":  float(cv_top["accuracy"].std()),
            "cv_kappa_mean":    float(cv_top["kappa"].mean()),
            "cv_kappa_std":     float(cv_top["kappa"].std()),
        },
        "most_important_group": group_df.iloc[0]["group"] if not group_df.empty else None,
        "best_temporal_epoch": temporal_df.iloc[0]["epoch"] if not temporal_df.empty else None,
        "exported_rasters": [
            "PS_LandCover_OBIA_v3.tif",
            f"PS_LandCover_OBIA_v3_Top{cfg.top_n_features}.tif",
            f"PS_LandCover_OBIA_v3_Final{l2_out_suffix}.tif",
        ],
        "l2": l2_metrics,
        "yrf_rule": {
            "applied":     cfg.yrf_apply,
            "canopy_min":  cfg.yrf_canopy_min,
            "canopy_max":  cfg.yrf_canopy_max,
            "ndvi_min":    cfg.yrf_ndvi_min,
            "eligible":    list(cfg.yrf_eligible_final_ids),
            "final_id":    cfg.yrf_final_id,
        },
    }
    with (cfg.out_dir / f"summary{l2_out_suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("DONE. Outputs written to %s", cfg.out_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
