"""Option A: vectorize L1=Dense directly, subdivide with SAM as cut lines.

Produces polygons that match the L1=Dense raster footprint exactly. Internal
boundaries come from SAM segments (where SAM saw a canopy edge). Where SAM
did not segment, the Dense connected-component stays as a single polygon.

Pipeline:
  1. Vectorise the L1=Dense mask → one polygon per 4-connected component.
  2. Read raw SAM polygons (no pre-filtering by area — small SAM cuts are
     useful for subdivision).
  3. Build SAM cut-line union (boundary lines, not polygons).
  4. For each Dense component, use the SAM polygons as overlay cuts:
       parts = Dense_component intersect each SAM_polygon  → "sam" rows
       leftover = Dense_component - union(SAM_polygons)    → "residual" rows
  5. Drop pieces smaller than MIN_AREA_HA (default 0.05 ha = 500 m²).
  6. Write GeoPackage + QML (categorized by source).

Output schema:
  sam_id     running id (1..N)
  source     "sam" or "residual"
  area_ha    polygon area in hectares
  perim_m    perimeter (m)
  compact    4πA / P²
  l2_class   empty (fill in QGIS as Natural / Production / Agroforest)
  l2_notes   empty
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

ROOT = Path("/Users/firmanhadi/Works/Cisokan/2026/Apr26/process/CSK")
DENSE_MASK = ROOT / "rasters" / "sam_input_aoi_dense_mask.tif"
SAM_GPKG = ROOT / "rasters" / "sam_output" / "sam_segments.gpkg"
OUT_GPKG = ROOT / "outputs_10epoch_obia_v3" / "dense_polygons_subdivided.gpkg"
OUT_QML = ROOT / "outputs_10epoch_obia_v3" / "dense_polygons_subdivided.qml"

MIN_AREA_HA = 0.05  # 500 m²
MIN_AREA_M2 = MIN_AREA_HA * 10000


def _polygons_from(geom):
    """Yield only Polygon parts from a possibly Multi/Collection geometry."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            yield from _polygons_from(g)


def main() -> None:
    # 1. Vectorise Dense mask → connected components
    print(f"Vectorising {DENSE_MASK.name} ...")
    with rasterio.open(DENSE_MASK) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    dense_components = []
    for geom, val in rio_shapes(arr, mask=(arr == 1), transform=transform):
        p = shape(geom)
        if p.area >= 1:  # drop sub-pixel slivers
            dense_components.append(p)
    n_components_total = len(dense_components)
    dense_components = [p for p in dense_components if p.area >= MIN_AREA_M2]
    dense_total_ha = sum(p.area for p in dense_components) / 10000
    print(f"  {n_components_total:,} raw connected components")
    print(f"  {len(dense_components):,} components ≥ {MIN_AREA_HA} ha "
          f"(total {dense_total_ha:.1f} ha)")

    # Build sindex for Dense components
    dense_gdf = gpd.GeoDataFrame(
        {"dense_id": np.arange(1, len(dense_components) + 1)},
        geometry=dense_components,
        crs=32748,
    )
    dense_sindex = dense_gdf.sindex

    # 2. Read SAM polygons (raw, no pre-filter)
    print(f"\nReading {SAM_GPKG.name} ...")
    sam = gpd.read_file(SAM_GPKG)
    if sam.crs is None or sam.crs.to_epsg() != 32748:
        sam = sam.set_crs(32748) if sam.crs is None else sam.to_crs(32748)
    print(f"  {len(sam):,} raw SAM polygons")
    sam_sindex = sam.sindex

    # 3. Per-component subdivision
    print("\nSubdividing each Dense component by SAM ...")
    sam_rows = []        # (geometry, dense_id)
    residual_rows = []   # (geometry, dense_id)
    n_components_done = 0
    for dense_id, dpoly in zip(dense_gdf["dense_id"], dense_gdf.geometry):
        # Candidate SAM polys via spatial index
        cand_idx = list(sam_sindex.intersection(dpoly.bounds))
        if not cand_idx:
            # No SAM coverage at all — whole component is residual
            for p in _polygons_from(dpoly):
                if p.area >= MIN_AREA_M2:
                    residual_rows.append((p, int(dense_id)))
            n_components_done += 1
            continue
        cand = sam.iloc[cand_idx]
        sam_in_dense = []
        # Intersect each candidate SAM with this Dense component
        for spoly in cand.geometry.values:
            if not spoly.intersects(dpoly):
                continue
            inter = spoly.intersection(dpoly)
            for p in _polygons_from(inter):
                if p.area >= MIN_AREA_M2:
                    sam_rows.append((p, int(dense_id)))
                    sam_in_dense.append(p)
        # Residual = Dense component minus union(sam ∩ component)
        if sam_in_dense:
            sam_union_local = unary_union(sam_in_dense)
            res = dpoly.difference(sam_union_local)
        else:
            res = dpoly
        for p in _polygons_from(res):
            if p.area >= MIN_AREA_M2:
                residual_rows.append((p, int(dense_id)))
        n_components_done += 1
        if n_components_done % 500 == 0:
            print(f"  processed {n_components_done:,} / {len(dense_gdf):,} components")
    print(f"  ✓ {n_components_done:,} components processed")

    # 4. Build output GeoDataFrame
    print("\nAssembling output ...")
    sam_geoms = [g for g, _ in sam_rows]
    sam_dense_ids = [d for _, d in sam_rows]
    res_geoms = [g for g, _ in residual_rows]
    res_dense_ids = [d for _, d in residual_rows]

    sam_part = gpd.GeoDataFrame(
        {"source": ["sam"] * len(sam_geoms), "dense_id": sam_dense_ids},
        geometry=sam_geoms, crs=32748,
    )
    res_part = gpd.GeoDataFrame(
        {"source": ["residual"] * len(res_geoms), "dense_id": res_dense_ids},
        geometry=res_geoms, crs=32748,
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

    cols = ["sam_id", "source", "dense_id", "area_ha", "perim_m", "compact",
            "l2_class", "l2_notes", "geometry"]
    out = out[cols]

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUT_GPKG} ...")
    out.to_file(OUT_GPKG, driver="GPKG", layer="dense_polygons_subdivided")

    # 5. Stats
    n_sam = (out["source"] == "sam").sum()
    n_res = (out["source"] == "residual").sum()
    total_ha = out["area_ha"].sum()
    print(f"\n  {len(out):,} polygons total ({n_sam:,} SAM-cut + {n_res:,} residual)")
    print(f"  total area: {total_ha:.1f} ha "
          f"(Dense reference: {dense_total_ha:.1f} ha "
          f"→ {100 * total_ha / dense_total_ha:.1f}% — should be ~100%)")
    print(f"  median area: {out['area_ha'].median():.3f} ha")
    print(f"  size distribution (ha):")
    for q in [0.05, 0.1, 0.5, 1, 5, 10]:
        n = (out["area_ha"] >= q).sum()
        print(f"    ≥ {q:5.2f} ha : {n:,}")

    qml = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <renderer-v2 type="categorizedSymbol" attr="source" forceraster="0" symbollevels="0">
    <categories>
      <category render="true" value="sam" symbol="0" label="SAM-cut Dense polygon"/>
      <category render="true" value="residual" symbol="1" label="Dense component (no SAM cut)"/>
    </categories>
    <symbols>
      <symbol name="0" type="fill"><layer class="SimpleFill"><Option type="Map">
        <Option name="color" value="33,150,243,150" type="QString"/>
        <Option name="outline_color" value="13,71,161,255" type="QString"/>
        <Option name="outline_width" value="0.20" type="QString"/>
      </Option></layer></symbol>
      <symbol name="1" type="fill"><layer class="SimpleFill"><Option type="Map">
        <Option name="color" value="76,175,80,150" type="QString"/>
        <Option name="outline_color" value="27,94,32,255" type="QString"/>
        <Option name="outline_width" value="0.20" type="QString"/>
      </Option></layer></symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="sam_id" textColor="20,20,20,255" fontSize="6"/>
      <placement placement="0"/>
    </settings>
  </labeling>
</qgis>
'''
    OUT_QML.write_text(qml)
    print(f"\n✓ {OUT_GPKG.relative_to(ROOT)}")
    print(f"✓ {OUT_QML.relative_to(ROOT)} (categorized: blue=SAM-cut, green=Dense-only)")


if __name__ == "__main__":
    for k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA", "GDAL_DRIVER_PATH"):
        os.environ.pop(k, None)
    main()
