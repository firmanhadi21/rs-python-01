"""Filter SAM segments + add residual fill for L2 forest-subtype refinement.

Inputs:
  rasters/sam_output/sam_segments.gpkg
  rasters/sam_input_wz_dense_mask.tif

Output:
  outputs_10epoch_obia_v3/sam_l2_candidates.gpkg
  outputs_10epoch_obia_v3/sam_l2_candidates.qml

Filter rules:
  1. Drop polygons with area < MIN_AREA_HA (default 0.05 ha = 500 m²) — SAM noise.
  2. Clip every polygon to the L1=Dense raster (vector intersection with the
     Dense polygon dissolved from the mask). This keeps only the Dense portion
     of each segment, regardless of whether the original polygon spilled across
     L1 class boundaries.
  3. Drop the (now clipped) polygons whose remaining area < MIN_AREA_HA.
  4. **Residual fill**: any Dense pixel not covered by a SAM polygon is
     vectorised into one or more "residual" segments labelled
     `source = "residual"` so the user can label them too. SAM-derived
     polygons get `source = "sam"`.

Output schema:
  sam_id     running id (1..N)
  source     "sam" or "residual"
  area_ha    polygon area in hectares (post-clipping)
  perim_m    perimeter in metres
  compact    4πA / P²
  l2_class   empty (to be filled in QGIS as Natural / Production / Agroforest)
  l2_notes   empty
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path("/Users/firmanhadi/Works/Cisokan/2026/Apr26/process/CSK")
SAM_GPKG = ROOT / "rasters" / "sam_output" / "sam_segments.gpkg"
DENSE_MASK = ROOT / "rasters" / "sam_input_aoi_dense_mask.tif"
OUT_GPKG = ROOT / "outputs_10epoch_obia_v3" / "sam_l2_candidates.gpkg"
OUT_QML = ROOT / "outputs_10epoch_obia_v3" / "sam_l2_candidates.qml"

MIN_AREA_HA = 0.05  # 500 m² — drop SAM noise


def main() -> None:
    print(f"Reading {SAM_GPKG.name} ...")
    gdf = gpd.read_file(SAM_GPKG)
    if gdf.crs is None or gdf.crs.to_epsg() != 32748:
        gdf = gdf.set_crs(32748) if gdf.crs is None else gdf.to_crs(32748)
    n_raw = len(gdf)
    print(f"  {n_raw:,} raw SAM polygons")

    # 1. Pre-filter: tiny polygons
    pre = len(gdf)
    gdf = gdf[gdf.geometry.area >= MIN_AREA_HA * 10000].copy()
    print(f"  dropped {pre - len(gdf):,} polygons < {MIN_AREA_HA} ha")

    # 2. Build the L1=Dense polygon (dissolved from the mask raster).
    print(f"\nBuilding L1=Dense polygon from {DENSE_MASK.name} ...")
    with rasterio.open(DENSE_MASK) as src:
        dense_arr = src.read(1)
        transform = src.transform
        crs = src.crs
    polys = []
    for geom, val in rio_shapes(dense_arr, mask=(dense_arr == 1), transform=transform):
        polys.append(shape(geom))
    dense_union = unary_union(polys)
    dense_total_ha = dense_union.area / 10000
    print(f"  dense union: {dense_total_ha:.1f} ha")

    # 3. Clip every SAM polygon to Dense.
    print("Clipping SAM polygons to L1=Dense ...")
    gdf["geometry"] = gdf.geometry.intersection(dense_union)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    # 4. Drop tiny clipped polygons
    pre = len(gdf)
    gdf["area_ha"] = gdf.geometry.area / 10000
    gdf = gdf[gdf["area_ha"] >= MIN_AREA_HA].copy()
    print(f"  dropped {pre - len(gdf):,} polygons < {MIN_AREA_HA} ha after clipping")
    sam_covered = unary_union(gdf.geometry.values)
    sam_covered_ha = sam_covered.area / 10000
    print(f"  SAM coverage of Dense: {sam_covered_ha:.1f} ha "
          f"({100 * sam_covered_ha / dense_total_ha:.1f}%)")

    # 5. Residual: Dense − SAM-covered.
    print("\nBuilding residual segments (Dense not covered by SAM) ...")
    residual = dense_union.difference(sam_covered)
    if residual.is_empty:
        residual_polys = []
    else:
        if isinstance(residual, Polygon):
            residual_polys = [residual]
        else:
            residual_polys = list(residual.geoms)
        # drop tiny residual scraps
        residual_polys = [p for p in residual_polys if p.area >= MIN_AREA_HA * 10000]
    print(f"  residual polygons (>= {MIN_AREA_HA} ha): {len(residual_polys)}")
    residual_ha = sum(p.area for p in residual_polys) / 10000
    print(f"  residual area: {residual_ha:.1f} ha")

    # 6. Assemble final GeoDataFrame
    sam_part = gdf[["geometry"]].copy()
    sam_part["source"] = "sam"
    res_part = gpd.GeoDataFrame(
        {"source": ["residual"] * len(residual_polys)},
        geometry=residual_polys,
        crs=32748,
    )
    out = gpd.GeoDataFrame(
        gpd.pd.concat([sam_part, res_part], ignore_index=True), crs=32748
    )

    out["area_ha"] = out.geometry.area / 10000
    out["perim_m"] = out.geometry.length
    out["compact"] = 4 * np.pi * out.geometry.area / np.where(
        out["perim_m"] > 0, out["perim_m"] ** 2, np.nan
    )
    out = out.reset_index(drop=True)
    out["sam_id"] = np.arange(1, len(out) + 1)
    out["l2_class"] = ""
    out["l2_notes"] = ""

    cols = ["sam_id", "source", "area_ha", "perim_m", "compact",
            "l2_class", "l2_notes", "geometry"]
    out = out[cols]

    print(f"\nWriting {OUT_GPKG} ...")
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT_GPKG, driver="GPKG", layer="sam_l2_candidates")

    total_ha = out["area_ha"].sum()
    n_sam = (out["source"] == "sam").sum()
    n_res = (out["source"] == "residual").sum()
    print(f"  {len(out):,} polygons total ({n_sam:,} SAM + {n_res:,} residual)")
    print(f"  total area: {total_ha:.1f} ha "
          f"(coverage of Dense: {100 * total_ha / dense_total_ha:.1f}%)")
    print(f"  median area: {out['area_ha'].median():.3f} ha")
    print(f"  size distribution (ha):")
    for q in [0.05, 0.1, 0.5, 1, 5]:
        n = (out["area_ha"] >= q).sum()
        print(f"    ≥ {q} ha : {n:,}")

    qml = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <renderer-v2 type="categorizedSymbol" attr="l2_class" forceraster="0" symbollevels="0">
    <categories>
      <category render="true" value="" symbol="0" label="Unlabelled (empty)"/>
      <category render="true" value="Natural" symbol="1" label="Natural Forest"/>
      <category render="true" value="Production" symbol="2" label="Production Forest"/>
      <category render="true" value="Agroforest" symbol="3" label="Agroforest"/>
    </categories>
    <symbols>
      <symbol name="0" type="fill"><layer class="SimpleFill"><Option type="Map">
        <Option name="color" value="200,200,200,80" type="QString"/>
        <Option name="outline_color" value="120,120,120,255" type="QString"/>
        <Option name="outline_width" value="0.18" type="QString"/>
      </Option></layer></symbol>
      <symbol name="1" type="fill"><layer class="SimpleFill"><Option type="Map">
        <Option name="color" value="27,94,32,160" type="QString"/>
        <Option name="outline_color" value="0,40,0,255" type="QString"/>
        <Option name="outline_width" value="0.20" type="QString"/>
      </Option></layer></symbol>
      <symbol name="2" type="fill"><layer class="SimpleFill"><Option type="Map">
        <Option name="color" value="56,142,60,160" type="QString"/>
        <Option name="outline_color" value="0,40,0,255" type="QString"/>
        <Option name="outline_width" value="0.20" type="QString"/>
      </Option></layer></symbol>
      <symbol name="3" type="fill"><layer class="SimpleFill"><Option type="Map">
        <Option name="color" value="139,195,74,160" type="QString"/>
        <Option name="outline_color" value="0,40,0,255" type="QString"/>
        <Option name="outline_width" value="0.20" type="QString"/>
      </Option></layer></symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="sam_id" textColor="20,20,20,255" fontSize="7"/>
      <placement placement="0"/>
    </settings>
  </labeling>
</qgis>
'''
    OUT_QML.write_text(qml)
    print(f"\n✓ Open {OUT_GPKG.relative_to(ROOT)} in QGIS to label l2_class.")


if __name__ == "__main__":
    for k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA", "GDAL_DRIVER_PATH"):
        os.environ.pop(k, None)
    main()
