#!/usr/bin/env python3
"""
Local Python port of Planetscope_10epoch_Apr26.js (GEE Code Editor).

Replicates the 10-epoch PlanetScope PS7 classification workflow on a local
machine (Mac/Linux) using rioxarray + dask + scikit-learn, so no Earth Engine
calls are required at runtime.

Pipeline
--------
1.  Read 10 PlanetScope 8-band GeoTIFFs (one per epoch).
2.  Per-epoch: compute NDVI, NDWI, NDBI, EVI (same formulas as the GEE script).
3.  Temporal metrics: max/min/std/amplitude across the 10 epochs for each
    of the four indices (16 bands).
4.  Tree height features: focal mean + focal std within a 30 m radius circle
    (2 bands, computed from a local canopy-height GeoTIFF).
5.  Stack all 138 bands, clip to AOI, min-max normalize per band.
6.  Sample features at training points (gpkg with `class` column, values 1-8).
7.  Train RandomForest (100 trees, min_samples_leaf=1, max_samples=0.5).
8.  Accuracy assessment on 30% hold-out + 5-fold cross-validation.
9.  Feature importance + top-20 subset retraining & CV.
10. Apply classifier to the full normalized stack (chunked via dask).
11. Focal-mode smoothing (1-pixel radius).
12. Write classified GeoTIFFs and analysis CSVs.

Configure paths in the CONFIG section below and run:

    python planetscope_10epoch_local.py
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
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
log = logging.getLogger("ps10epoch")


# =============================================================================
# CONFIG  ---  edit these paths to match your filesystem
# =============================================================================
@dataclass
class Config:
    # Directory holding the 10 PlanetScope GeoTIFFs (pattern: CSK_mmddyy.tif)
    raster_dir: Path = Path("/Users/macbook/CSK/rasters")

    # Epoch label -> filename. Labels MUST stay as-is (used for band prefixes
    # and temporal importance grouping). Only change the right-hand filenames.
    epochs: Dict[str, str] = field(default_factory=lambda: {
        "march":  "CSK_032124.tif",   # 21 Mar 2024
        "june":   "CSK_063024.tif",   # 30 Jun 2024
        "aug":    "CSK_080524.tif",   #  5 Aug 2024
        "sept":   "CSK_091624.tif",   # 16 Sep 2024
        "jan25":  "CSK_011025.tif",   # 10 Jan 2025
        #"apr25":  "CSK_042825.tif",   # 28 Apr 2025
        "may25":  "CSK_050525.tif",   #  5 May 2025
        "aug25":  "CSK_081525.tif",   # 15 Aug 2025
        "sep25":  "CSK_090725.tif",   #  7 Sep 2025
        "nov25":  "CSK_112425.tif",   # 24 Nov 2025
        "mar26": "CSK_031726.tif"
    })

    # Canopy height GeoTIFF (single-band, any CRS -- will be reprojected)
    canopy_height: Path = Path("/Users/macbook/CSK/canopy_height.tif")

    # AOI boundary (GeoPackage). If `aoi_layer` is None, the first layer is used.
    aoi_gpkg: Path = Path("/Users/macbook/CSK/aoi_cisokan.gpkg")
    aoi_layer: Optional[str] = None

    # Training samples (GeoPackage) with a class column (values 1-8)
    samples_gpkg: Path = Path("/Users/macbook/CSK/samples.gpkg")
    samples_layer: Optional[str] = None
    class_column: str = "class"

    # Processing
    chunk_size: int = 1024        # dask chunk in pixels (both x and y)
    rf_trees: int = 100
    rf_min_leaf: int = 1
    rf_bag_fraction: float = 0.5  # GEE bagFraction -> sklearn max_samples
    random_seed: int = 42
    train_fraction: float = 0.7
    n_cv_folds: int = 5
    top_n_features: int = 20

    # 8-band PlanetScope expected band order (1-indexed in rioxarray)
    n_bands: int = 8

    # Tree-height neighborhood radius (meters) -- converted to pixels using
    # the PlanetScope resolution at read-time.
    tree_height_radius_m: float = 30.0

    # Class labels (keys must match integer values in `class_column`)
    class_labels: Dict[int, str] = field(default_factory=lambda: {
        1: "Waterbody",
        2: "Paddy",
        3: "Built-up",
        4: "Clouds",
        5: "Dense Vegetation",
        6: "Sparse Vegetation",
        7: "Ladang",
        8: "Bareland",
    })

    # Output directory
    out_dir: Path = Path("/Users/macbook/CSK/outputs_10epoch")


CFG = Config()


# =============================================================================
# UTILITIES
# =============================================================================
def chunks_kw(cfg: Config) -> Dict[str, int]:
    return {"x": cfg.chunk_size, "y": cfg.chunk_size}


def open_planet_tif(path: Path, cfg: Config, ref: Optional[xr.DataArray] = None):
    """Open an 8-band PlanetScope GeoTIFF and, if `ref` is given, reproject to
    match the reference grid. Returns a float32 DataArray with dim (band, y, x)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing PlanetScope raster: {path}")
    da = rxr.open_rasterio(path, chunks=chunks_kw(cfg), masked=True).astype("float32")
    if da.sizes["band"] < cfg.n_bands:
        raise ValueError(
            f"{path.name} has {da.sizes['band']} bands, expected >= {cfg.n_bands}"
        )
    da = da.isel(band=slice(0, cfg.n_bands))
    if ref is not None:
        da = da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    return da


def compute_indices_for_epoch(arr: xr.DataArray) -> Dict[str, xr.DataArray]:
    """NDVI, NDWI, NDBI, EVI using the same band assignments as the GEE script.
    Assumes the 8-band order where b2=Blue, b3=Green, b4=Red, b6=?, b8=NIR."""
    eps = 1e-10
    b2 = arr.sel(band=2)
    b3 = arr.sel(band=3)
    b4 = arr.sel(band=4)
    b6 = arr.sel(band=6)
    b8 = arr.sel(band=8)

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b2 - b4) / (b2 + b4 + eps)
    ndbi = (b6 - b3) / (b6 + b3 + eps)
    evi = 2.5 * (b8 - b4) / (b8 + 6.0 * b4 - 7.5 * b2 + 1.0)

    return {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "EVI": evi}


def circular_kernel(radius_pix: int) -> np.ndarray:
    """Binary circular kernel of given radius in pixels."""
    r = int(radius_pix)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = (x * x + y * y <= r * r).astype("float32")
    return k


def focal_mean_circle(arr2d: np.ndarray, radius_pix: int) -> np.ndarray:
    k = circular_kernel(radius_pix)
    k /= k.sum()
    return ndimage.convolve(arr2d, k, mode="reflect")


def focal_std_circle(arr2d: np.ndarray, radius_pix: int) -> np.ndarray:
    k = circular_kernel(radius_pix)
    k_norm = k / k.sum()
    mean = ndimage.convolve(arr2d, k_norm, mode="reflect")
    mean_sq = ndimage.convolve(arr2d * arr2d, k_norm, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def focal_mode_categorical(arr2d: np.ndarray, radius_pix: int = 1,
                           n_classes: int = 8) -> np.ndarray:
    """Mode filter for an integer-class raster. Uses per-class convolution,
    which is O(n_classes * pixels) but fully vectorized."""
    k = circular_kernel(radius_pix).astype("int32")
    arr = arr2d.astype(np.int16)
    best_count = np.zeros_like(arr, dtype=np.int32)
    best_val = arr.copy()
    for c in range(1, n_classes + 1):
        count = ndimage.convolve((arr == c).astype(np.int32), k,
                                 mode="constant", cval=0)
        update = count > best_count
        best_count = np.where(update, count, best_count)
        best_val = np.where(update, c, best_val)
    return best_val.astype(np.uint8)


# =============================================================================
# FEATURE BUILDING
# =============================================================================
def build_feature_stack(cfg: Config) -> Tuple[xr.DataArray, np.ndarray, xr.DataArray]:
    """Build the 138-band feature stack. Returns:
        stacked   : xr.DataArray with dim ('feature','y','x'), dask-backed
        aoi_mask  : 2D numpy bool (True = inside AOI)
        ref       : reference 2D DataArray (for CRS/transform metadata)
    """
    log.info("Loading reference grid from %s", list(cfg.epochs.values())[0])
    ref_path = cfg.raster_dir / list(cfg.epochs.values())[0]
    ref_epoch = open_planet_tif(ref_path, cfg)
    ref2d = ref_epoch.isel(band=0)

    cache_dir = cfg.out_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_crs = ref_epoch.rio.crs
    ref_transform = ref_epoch.rio.transform()

    def _write_feature(da: xr.DataArray, name: str) -> None:
        out = da.astype("float32")
        out = out.rio.write_crs(ref_crs)
        out = out.rio.write_transform(ref_transform)
        out.rio.to_raster(cache_dir / f"{name}.tif",
                          compress="deflate", tiled=True, BIGTIFF="IF_SAFER")

    def _read_feature(name: str) -> xr.DataArray:
        da = rxr.open_rasterio(cache_dir / f"{name}.tif",
                               chunks=chunks_kw(cfg), masked=True).astype("float32")
        if "band" in da.dims:
            da = da.isel(band=0)
        da = da.drop_vars("band", errors="ignore")
        return da.assign_coords(y=ref2d["y"], x=ref2d["x"])

    feature_names: List[str] = []

    # 1-3) Per-epoch: load, reproject, compute indices, persist each band, free RAM
    for label, fname in cfg.epochs.items():
        log.info("  epoch %-6s  %s", label, fname)
        arr = open_planet_tif(cfg.raster_dir / fname, cfg, ref=ref_epoch)
        for bi in range(1, cfg.n_bands + 1):
            name = f"{label}_b{bi}"
            _write_feature(arr.sel(band=bi).drop_vars("band", errors="ignore"), name)
            feature_names.append(name)
        for idx_name, idx_arr in compute_indices_for_epoch(arr).items():
            name = f"{label}_{idx_name}"
            _write_feature(idx_arr, name)
            feature_names.append(name)
        del arr

    # 4) Temporal metrics: reduce each index lazily across 10 epoch cache tifs
    log.info("Computing temporal metrics across 10 epochs")
    for idx_name in ("NDVI", "NDBI", "EVI", "NDWI"):
        stack = xr.concat(
            [_read_feature(f"{label}_{idx_name}") for label in cfg.epochs.keys()],
            dim="epoch",
        )
        _write_feature(stack.max(dim="epoch"), f"max{idx_name}")
        _write_feature(stack.min(dim="epoch"), f"min{idx_name}")
        _write_feature(stack.std(dim="epoch"), f"std{idx_name}")
        amp = _read_feature(f"max{idx_name}") - _read_feature(f"min{idx_name}")
        _write_feature(amp, f"amp{idx_name}")
        feature_names += [f"max{idx_name}", f"min{idx_name}",
                          f"std{idx_name}", f"amp{idx_name}"]

    # 5) Tree-height mean + std in 30 m circle neighborhood
    log.info("Processing canopy-height layer")
    ch = rxr.open_rasterio(cfg.canopy_height, chunks=chunks_kw(cfg), masked=True).astype("float32")
    if ch.sizes["band"] > 1:
        ch = ch.isel(band=slice(0, 1))
    ch = ch.rio.reproject_match(ref_epoch, resampling=Resampling.bilinear).isel(band=0)

    px_size = abs(float(ref2d.rio.resolution()[0]))
    radius_pix = max(1, int(round(cfg.tree_height_radius_m / px_size)))
    log.info("  pixel size %.3f m -> tree-height radius = %d pixels", px_size, radius_pix)

    ch_np = ch.compute().values
    ch_np = np.where(np.isfinite(ch_np), ch_np, 0.0)
    th_mean = focal_mean_circle(ch_np, radius_pix).astype("float32")
    th_std = focal_std_circle(ch_np, radius_pix).astype("float32")
    yx_coords = {"y": ref2d["y"], "x": ref2d["x"]}
    _write_feature(xr.DataArray(th_mean, coords=yx_coords, dims=("y", "x")),
                   "tree_height_mean")
    _write_feature(xr.DataArray(th_std, coords=yx_coords, dims=("y", "x")),
                   "tree_height_std")
    feature_names += ["tree_height_mean", "tree_height_std"]
    del ch, ch_np, th_mean, th_std

    # 6) AOI mask (rasterize gpkg to reference grid)
    log.info("Rasterizing AOI to reference grid")
    aoi = gpd.read_file(cfg.aoi_gpkg, layer=cfg.aoi_layer)
    aoi = aoi.to_crs(ref_crs)
    mask_np = rasterize(
        [(geom, 1) for geom in aoi.geometry],
        out_shape=(ref2d.sizes["y"], ref2d.sizes["x"]),
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    log.info("  AOI covers %d of %d pixels (%.1f%%)",
             mask_np.sum(), mask_np.size, 100.0 * mask_np.sum() / mask_np.size)

    # 7) Reopen every cached feature lazily and concat into a dask-backed stack
    log.info("Reopening %d feature bands as lazy dask stack", len(feature_names))
    stacked = xr.concat([_read_feature(n) for n in feature_names], dim="feature")
    stacked = stacked.assign_coords(feature=feature_names)
    stacked = stacked.chunk({"feature": -1, "y": cfg.chunk_size, "x": cfg.chunk_size})
    stacked = stacked.rio.write_crs(ref_crs).rio.write_transform(ref_transform)

    return stacked, mask_np, ref2d


def normalize_stack(stacked: xr.DataArray, aoi_mask: np.ndarray) -> Tuple[xr.DataArray, pd.DataFrame]:
    """Min-max normalize each feature band using statistics inside the AOI.

    Returns normalized stack (float32, dask-backed) and a DataFrame of
    per-feature min/max used for normalization (persisted later as JSON).
    """
    log.info("Computing per-band min/max over AOI (one dask reduction)")
    mask_da = xr.DataArray(
        aoi_mask,
        dims=("y", "x"),
        coords={"y": stacked["y"], "x": stacked["x"]},
    )
    masked = stacked.where(mask_da)
    mn = masked.min(dim=("y", "x")).compute()
    mx = masked.max(dim=("y", "x")).compute()
    rng = (mx - mn).where((mx - mn) > 1e-12, other=1.0)
    log.info("  min/max computation done")

    norm = (stacked - mn) / rng
    norm = norm.where(mask_da)
    stats = pd.DataFrame({
        "feature": stacked.coords["feature"].values,
        "min": mn.values,
        "max": mx.values,
    })
    return norm.astype("float32"), stats


# =============================================================================
# SAMPLING AT POINTS
# =============================================================================
def sample_at_points(normalized: xr.DataArray, gdf: gpd.GeoDataFrame,
                     class_col: str) -> pd.DataFrame:
    """Sample all feature bands at point geometries using nearest-neighbor
    selection. Only loads the chunks containing the points."""
    log.info("Sampling %d points from %d feature bands",
             len(gdf), normalized.sizes["feature"])
    xs = xr.DataArray(gdf.geometry.x.values, dims="sample")
    ys = xr.DataArray(gdf.geometry.y.values, dims="sample")
    vals = normalized.sel(x=xs, y=ys, method="nearest").compute()
    # shape: (feature, sample)
    df = pd.DataFrame(
        vals.values.T,
        columns=normalized.coords["feature"].values,
    )
    df[class_col] = gdf[class_col].to_numpy()
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        log.warning("Dropped %d points with NaN feature values (likely outside AOI/nodata)",
                    before - len(df))
    return df


# =============================================================================
# TRAINING / EVALUATION
# =============================================================================
def make_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        min_samples_leaf=cfg.rf_min_leaf,
        max_samples=cfg.rf_bag_fraction,   # GEE bagFraction
        bootstrap=True,
        max_features="sqrt",               # variablesPerSplit default
        random_state=cfg.random_seed,
        n_jobs=-1,
    )


def evaluate_model(clf, X_val, y_val, cfg: Config) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    y_pred = clf.predict(X_val)
    labels = sorted(cfg.class_labels.keys())
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    oa = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=[cfg.class_labels[c] for c in labels],
                         columns=[cfg.class_labels[c] for c in labels])

    per_class = []
    for i, c in enumerate(labels):
        tp = cm[i, i]
        col_sum = cm[:, i].sum()   # predicted as c
        row_sum = cm[i, :].sum()   # actually c
        prec = tp / col_sum if col_sum else 0.0
        rec = tp / row_sum if row_sum else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({
            "class": c,
            "class_name": cfg.class_labels[c],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support_val": row_sum,
        })
    return ({"overall_accuracy": oa, "kappa": kappa},
            cm_df,
            pd.DataFrame(per_class))


def cross_validate(cfg: Config, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    kf = KFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.random_seed)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        clf = make_rf(cfg)
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[te], y_pred),
            "kappa": cohen_kappa_score(y[te], y_pred),
            "train_size": len(tr),
            "test_size": len(te),
        })
    return pd.DataFrame(rows)


# =============================================================================
# RASTER-WIDE PREDICTION
# =============================================================================
def predict_full_raster(normalized: xr.DataArray, clf: RandomForestClassifier,
                        feature_names: List[str]) -> xr.DataArray:
    """Apply a trained classifier across the whole normalized stack, chunk by
    chunk. Returns an (y, x) uint8 DataArray; 0 = masked/nodata."""
    subset = normalized.sel(feature=feature_names)
    log.info("Classifying full raster in dask chunks  (%d features)", len(feature_names))

    def _predict_block(block: np.ndarray) -> np.ndarray:
        # block shape: (n_features, by, bx)
        nf, by, bx = block.shape
        flat = block.reshape(nf, -1).T     # (pixels, n_features)
        valid = np.all(np.isfinite(flat), axis=1)
        out = np.zeros(flat.shape[0], dtype="uint8")
        if valid.any():
            out[valid] = clf.predict(flat[valid]).astype("uint8")
        return out.reshape(by, bx)

    dask_arr = subset.data  # dask array shape (feature, y, x)
    import dask.array as dsa
    pred = dsa.map_blocks(
        _predict_block,
        dask_arr,
        drop_axis=0,
        dtype="uint8",
        chunks=(dask_arr.chunks[1], dask_arr.chunks[2]),
    )
    pred_da = xr.DataArray(
        pred,
        dims=("y", "x"),
        coords={"y": subset["y"], "x": subset["x"]},
        name="classified",
    )
    pred_da.rio.write_crs(normalized.rio.crs, inplace=True)
    pred_da.rio.write_transform(normalized.rio.transform(), inplace=True)
    return pred_da


def smooth_classified(classified: xr.DataArray, n_classes: int = 8) -> xr.DataArray:
    """Focal-mode smoothing with 1-pixel circular kernel."""
    log.info("Focal-mode smoothing")
    arr = classified.compute().values
    sm = focal_mode_categorical(arr, radius_pix=1, n_classes=n_classes)
    out = xr.DataArray(sm, dims=classified.dims, coords=classified.coords,
                       name="classified_smoothed")
    out.rio.write_crs(classified.rio.crs, inplace=True)
    out.rio.write_transform(classified.rio.transform(), inplace=True)
    return out


# =============================================================================
# FEATURE IMPORTANCE / GROUP ANALYSIS
# =============================================================================
def group_importance(importance_df: pd.DataFrame, epoch_labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group feature importance by feature-type and by temporal epoch."""
    feats = importance_df.copy()

    def group_of(name: str) -> str:
        up = name
        if "tree_height" in name:
            return "Tree_Height"
        if "NDVI" in up:
            return "NDVI_Indices"
        if "NDBI" in up:
            return "NDBI_Indices"
        if "NDWI" in up:
            return "NDWI_Indices"
        if "EVI" in up:
            return "EVI_Indices"
        # spectral bands: tokens of the form <epoch>_b<N>
        if "_b" in name and any(name.endswith(f"_b{i}") for i in range(1, 9)):
            return "Spectral_Bands"
        return "Other"

    feats["group"] = feats["feature"].map(group_of)
    g = feats.groupby("group").agg(
        total_importance=("importance", "sum"),
        feature_count=("feature", "count"),
        avg_importance=("importance", "mean"),
    ).reset_index().sort_values("total_importance", ascending=False)

    # Temporal grouping: match the epoch prefix exactly (longest-match first
    # so 'aug25' wins over 'aug').
    sorted_labels = sorted(epoch_labels, key=len, reverse=True)

    def temporal_of(name: str) -> Optional[str]:
        for label in sorted_labels:
            if name.startswith(label + "_"):
                return label
        return None

    feats["epoch"] = feats["feature"].map(temporal_of)
    t = feats.dropna(subset=["epoch"]).groupby("epoch").agg(
        total_importance=("importance", "sum"),
        feature_count=("feature", "count"),
        avg_importance=("importance", "mean"),
    ).reset_index().sort_values("total_importance", ascending=False)

    return g, t


# =============================================================================
# I/O HELPERS
# =============================================================================
def save_geotiff(arr: xr.DataArray, path: Path, dtype: str = "uint8",
                 nodata: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing %s", path)
    arr.rio.to_raster(path, dtype=dtype, compress="deflate", nodata=nodata)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    cfg = CFG
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- feature stack + normalization ----
    stacked, aoi_mask, ref2d = build_feature_stack(cfg)
    normalized, norm_stats = normalize_stack(stacked, aoi_mask)
    norm_stats.to_csv(cfg.out_dir / "normalization_min_max.csv", index=False)

    feature_names: List[str] = list(normalized.coords["feature"].values)
    log.info("Total feature bands: %d", len(feature_names))

    # ---- training points ----
    log.info("Loading samples from %s", cfg.samples_gpkg)
    gdf = gpd.read_file(cfg.samples_gpkg, layer=cfg.samples_layer)
    gdf = gdf.to_crs(normalized.rio.crs)
    if cfg.class_column not in gdf.columns:
        raise KeyError(f"Column '{cfg.class_column}' not found in samples: {list(gdf.columns)}")
    gdf = gdf[gdf[cfg.class_column].isin(cfg.class_labels.keys())].copy()

    # class distribution
    cls_counts = gdf[cfg.class_column].value_counts().sort_index()
    log.info("Class distribution:\n%s", cls_counts.to_string())
    cls_counts.rename("count").to_csv(cfg.out_dir / "class_distribution.csv")

    sampled = sample_at_points(normalized, gdf, cfg.class_column)
    sampled.to_csv(cfg.out_dir / "training_sampled_values.csv", index=False)

    # ---- train / val split ----
    rng = np.random.default_rng(cfg.random_seed)
    mask = rng.random(len(sampled)) < cfg.train_fraction
    train = sampled.loc[mask].reset_index(drop=True)
    valid = sampled.loc[~mask].reset_index(drop=True)
    log.info("Train / validation sizes: %d / %d", len(train), len(valid))

    X_tr = train[feature_names].to_numpy(dtype="float32")
    y_tr = train[cfg.class_column].to_numpy(dtype="int32")
    X_va = valid[feature_names].to_numpy(dtype="float32")
    y_va = valid[cfg.class_column].to_numpy(dtype="int32")

    # ---- full-feature RF ----
    log.info("Training RandomForest on all %d features", len(feature_names))
    rf = make_rf(cfg)
    rf.fit(X_tr, y_tr)
    metrics_all, cm_all, per_class_all = evaluate_model(rf, X_va, y_va, cfg)
    log.info("Full-feature OA=%.4f  Kappa=%.4f",
             metrics_all["overall_accuracy"], metrics_all["kappa"])
    cm_all.to_csv(cfg.out_dir / "confusion_matrix_all.csv")
    per_class_all.to_csv(cfg.out_dir / "per_class_metrics_all.csv", index=False)

    # ---- feature importance ----
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(cfg.out_dir / "feature_importance_10epoch.csv", index=False)
    log.info("Top 10 features:\n%s", imp_df.head(10).to_string(index=False))

    # ---- group analysis ----
    group_df, temporal_df = group_importance(imp_df, list(cfg.epochs.keys()))
    group_df.to_csv(cfg.out_dir / "feature_group_analysis_10epoch.csv", index=False)
    temporal_df.to_csv(cfg.out_dir / "temporal_group_analysis_10epoch.csv", index=False)

    # ---- 5-fold CV on all features ----
    log.info("5-fold CV on all features")
    X_all = sampled[feature_names].to_numpy(dtype="float32")
    y_all = sampled[cfg.class_column].to_numpy(dtype="int32")
    cv_all = cross_validate(cfg, X_all, y_all)
    cv_all.to_csv(cfg.out_dir / "cv_results_10epoch_all_features.csv", index=False)
    log.info("CV(all):  mean OA=%.4f  std=%.4f   mean K=%.4f  std=%.4f",
             cv_all["accuracy"].mean(), cv_all["accuracy"].std(),
             cv_all["kappa"].mean(), cv_all["kappa"].std())

    # ---- top-N features ----
    top_n = imp_df.head(cfg.top_n_features)["feature"].tolist()
    log.info("Retraining on top %d features", cfg.top_n_features)
    rf_top = make_rf(cfg)
    rf_top.fit(train[top_n].to_numpy(dtype="float32"), y_tr)
    metrics_top, cm_top, per_class_top = evaluate_model(
        rf_top, valid[top_n].to_numpy(dtype="float32"), y_va, cfg)
    log.info("Top-%d  OA=%.4f  Kappa=%.4f",
             cfg.top_n_features, metrics_top["overall_accuracy"], metrics_top["kappa"])
    cm_top.to_csv(cfg.out_dir / f"confusion_matrix_top{cfg.top_n_features}.csv")
    per_class_top.to_csv(cfg.out_dir / f"per_class_metrics_top{cfg.top_n_features}.csv", index=False)

    cv_top = cross_validate(cfg, sampled[top_n].to_numpy(dtype="float32"), y_all)
    cv_top.to_csv(cfg.out_dir / f"cv_results_10epoch_top{cfg.top_n_features}.csv", index=False)
    log.info("CV(top%d): mean OA=%.4f  std=%.4f   mean K=%.4f  std=%.4f",
             cfg.top_n_features,
             cv_top["accuracy"].mean(), cv_top["accuracy"].std(),
             cv_top["kappa"].mean(), cv_top["kappa"].std())

    # ---- full-raster classification ----
    classified = predict_full_raster(normalized, rf, feature_names)
    save_geotiff(classified, cfg.out_dir / "PS_LandCover_10epoch_2024_2025.tif")
    smoothed = smooth_classified(classified, n_classes=max(cfg.class_labels.keys()))
    save_geotiff(smoothed, cfg.out_dir / "PS_LandCover_10epoch_2024_2025_Smoothed.tif")

    classified_top = predict_full_raster(normalized, rf_top, top_n)
    save_geotiff(classified_top, cfg.out_dir / f"PS_LandCover_10epoch_Top{cfg.top_n_features}.tif")
    smoothed_top = smooth_classified(classified_top, n_classes=max(cfg.class_labels.keys()))
    save_geotiff(smoothed_top, cfg.out_dir / f"PS_LandCover_10epoch_Top{cfg.top_n_features}_Smoothed.tif")

    # ---- summary ----
    summary = {
        "n_features": len(feature_names),
        "top_n_features": cfg.top_n_features,
        "train_size": int(len(train)),
        "validation_size": int(len(valid)),
        "all_features": {
            "holdout_accuracy": float(metrics_all["overall_accuracy"]),
            "holdout_kappa": float(metrics_all["kappa"]),
            "cv_accuracy_mean": float(cv_all["accuracy"].mean()),
            "cv_accuracy_std": float(cv_all["accuracy"].std()),
            "cv_kappa_mean": float(cv_all["kappa"].mean()),
            "cv_kappa_std": float(cv_all["kappa"].std()),
        },
        f"top_{cfg.top_n_features}": {
            "holdout_accuracy": float(metrics_top["overall_accuracy"]),
            "holdout_kappa": float(metrics_top["kappa"]),
            "cv_accuracy_mean": float(cv_top["accuracy"].mean()),
            "cv_accuracy_std": float(cv_top["accuracy"].std()),
            "cv_kappa_mean": float(cv_top["kappa"].mean()),
            "cv_kappa_std": float(cv_top["kappa"].std()),
        },
        "most_important_group": group_df.iloc[0]["group"] if not group_df.empty else None,
        "best_temporal_epoch": temporal_df.iloc[0]["epoch"] if not temporal_df.empty else None,
        "exported_rasters": [
            "PS_LandCover_10epoch_2024_2025.tif",
            "PS_LandCover_10epoch_2024_2025_Smoothed.tif",
            f"PS_LandCover_10epoch_Top{cfg.top_n_features}.tif",
            f"PS_LandCover_10epoch_Top{cfg.top_n_features}_Smoothed.tif",
        ],
    }
    with (cfg.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("DONE. Outputs written to %s", cfg.out_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
