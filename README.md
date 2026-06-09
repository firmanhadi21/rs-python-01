# rs-python-01

Object-Based Image Analysis (OBIA) pipeline for land-cover classification of the **Cisokan watershed**, West Java, Indonesia. Combines multi-temporal PlanetScope imagery with Sentinel-1 SAR, PALSAR-2, and canopy height data to produce a 9-class hierarchical land-cover map.

## Results

Verified 2026-05-03 on 316k LSMS segments:

| Level | OA | Kappa | CV OA |
|-------|----|-------|-------|
| L1 (7 classes) | 89.5% | 0.875 | 92.0% ± 1.6% |
| L2 (3 forest subtypes) | 57.5% | 0.324 | 64.9% ± 7.6% |

## Class scheme

| ID | Class | Source |
|----|-------|--------|
| 1 | Waterbody | L1 |
| 2 | Paddy | L1 |
| 3 | Built-up | L1 |
| 4 | Others | L1 |
| 5 | Natural Forest | L2 (within Dense Vegetation) |
| 6 | Production Forest | L2 (within Dense Vegetation) |
| 7 | Agroforest | L2 (within Dense Vegetation) |
| 8 | Sparse Vegetation | L1 |
| 9 | Crops | L1 |

## Feature stack (591 features per segment)

| Group | Features |
|-------|----------|
| PlanetScope spectral bands (8 bands × 10 epochs) | 80 |
| Per-epoch indices (NDVI, NDWI, NDBI, EVI × 10 epochs) | 40 |
| Temporal summaries (max/min/std/amp) | 16 |
| New per-epoch indices (NDRE, CIre, RENDVI, SAVI, OSAVI, MSAVI2, GNDVI, ARVI, VARI, BSI × 10 epochs) | 100 |
| Temporal percentiles + harmonic fit + YoY delta | 28 |
| Tree height (ETH 10 m canopy height) | 2 |
| Segment texture (entropy, range, CV, MAD) | 6 |
| Segment shape (area, perimeter, compactness, elongation, fill) | 5 |
| Sentinel-1 temporal SAR | 38 |
| PALSAR-2 backscatter + indices | 10 |

## Data sources

- **PlanetScope SuperDove** — 10 epochs, 2024-03 to 2026-03, 8 bands at 3 m
- **Sentinel-1 GRD** — IW descending, 2024-03 to 2026-03 (via Earth Engine)
- **ALOS-2 PALSAR yearly mosaic** — JAXA, year 2020 (via Earth Engine)
- **ETH canopy height** — `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1`
- **Training samples** — `samples.gpkg` (L1 class + L2 class_l2 columns)

## Scripts

| Script | Purpose |
|--------|---------|
| `planetscope_10epoch_obia_v3.py` | **Main classifier** — runs the full OBIA pipeline end-to-end |
| `planetscope_10epoch_local.py` | Pixel-level feature cache builder (prerequisite for v3) |
| `build_sar_features.py` | Downloads S1 + PALSAR feature stacks from Earth Engine |
| `ee_init.py` | Earth Engine service-account authentication helper |
| `download_meta_canopy_v2.py` | Downloads Meta Data-for-Good v2 canopy-height layers |
| `curate_samples_for_obia.py` | Curates training samples — one point per segment, flags conflicts |
| `separability_plot.py` | Pairwise class separability heatmap across all features |
| `spectral_signature_plot.py` | Mean spectral signature per class across all 138 legacy features |
| `sam_preprocess.py` | Prepares RGB COG + Dense mask for SAM segmentation |
| `sam_run.py` | Runs SamGeo (ViT-B) over the AOI as an alternative segmenter |
| `sam_filter.py` | Filters SAM polygons and fills residuals within the Dense mask |
| `sam_option_a.py` | Alternative: subdivides L1=Dense components using SAM cut lines |

## Setup

### Requirements

- Python 3.14
- OTB 8.1.2 (for LSMS segmentation)
- Earth Engine service account

```bash
python3.14 -m venv .venv
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install numpy pandas scipy scikit-learn \
    geopandas rasterio>=1.5 rioxarray xarray shapely pyogrio pyproj \
    earthengine-api
```

### Earth Engine credentials

Set environment variables (or let `ee_init.py` fall back to its built-in defaults):

```bash
export GEE_PROJECT_ID=your-project-id
export GEE_SERVICE_ACCOUNT_EMAIL=your-sa@your-project.iam.gserviceaccount.com
export GEE_SERVICE_ACCOUNT_KEY_FILE=~/path/to/key.json
```

Smoke test:

```bash
.venv/bin/python ee_init.py
# expected: "EE round-trip on project ...: 1 + 1 = 2"
```

## Running the pipeline

### 1. Build the pixel feature cache (one-time)

```bash
.venv/bin/python planetscope_10epoch_local.py
```

### 2. Download SAR features from Earth Engine (one-time or when refreshing)

```bash
.venv/bin/python build_sar_features.py
```

### 3. Run the main OBIA classifier

```bash
.venv/bin/python -u planetscope_10epoch_obia_v3.py
```

Cached segmentation and feature TIFs make subsequent runs fast (~5–8 min). To force a clean rebuild:

```bash
rm -rf outputs_10epoch_obia_v3/feature_cache_v3
.venv/bin/python -u planetscope_10epoch_obia_v3.py
```

## Training samples

`samples.gpkg` contains point features with two label columns:

| Column | Used by | Values |
|--------|---------|--------|
| `class` | L1 classifier | 1–7 (Waterbody, Paddy, Built-up, Others, Dense Vegetation, Sparse Vegetation, Crops) |
| `class_l2` | L2 classifier | `Natural`, `Production`, `Agroforest` (only for points where `class = 5`) |

To curate samples before training (one point per segment, conflict detection):

```bash
.venv/bin/python curate_samples_for_obia.py
```
