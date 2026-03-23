# Databricks notebook source
# MAGIC %pip install geopandas shapely rasterio pycountry folium plotly scikit-learn pyproj

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import folium as fl
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from shapely.geometry import Point
from shapely.wkt import loads as wkt_loads
from sklearn.cluster import KMeans
from shapely import wkt as shapely_wkt

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import udf

import re
print(f"Spark version: {spark.version}")

# COMMAND ----------

# CONFIGURATION

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/sgpbpi163/vgpbpi163"

COUNTRY_ISO3 = "ZMB"
POPULATION_YEAR = 2025

TRAVEL_API = ""  # "" for buffer, "osm", or "mapbox"

DISTANCE_METERS = 10000
dis_km = int(DISTANCE_METERS / 1000)
distance_name = f"{dis_km}km"

MAPBOX_ACCESS_TOKEN = ""
MAPBOX_MODE = "driving"

POTENTIAL_TYPE = "grid"  # "grid" or "kmeans"
GRID_SPACING = 0.03
N_CLUSTERS = 100

TARGET_NEW_FACILITIES = 50
H3_RESOLUTION = 8  # Must match extraction resolution

# Set to True to recompute cached results
FORCE_RECOMPUTE = False

# H3 resolution 8 edge length is ~461m
# k_rings = ceil(distance_meters / edge_length)
H3_EDGE_LENGTH_M = {4: 22606, 5: 8544, 6: 3229, 7: 1220, 8: 461, 9: 174, 10: 66}
K_RINGS = int(np.ceil(DISTANCE_METERS / H3_EDGE_LENGTH_M[H3_RESOLUTION]))

# Derived table names (input)
GADM_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_{COUNTRY_ISO3.lower()}"
FACILITIES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{COUNTRY_ISO3.lower()}"
POPULATION_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.population_{COUNTRY_ISO3.lower()}_{POPULATION_YEAR}"

# Derived table names (cached intermediate results)
POPULATION_AOI_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.population_aoi_{COUNTRY_ISO3.lower()}_{POPULATION_YEAR}_{distance_name}"
FACILITIES_H3_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.facilities_h3_{COUNTRY_ISO3.lower()}_{distance_name}"
FACILITIES_COVERAGE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.facilities_coverage_{COUNTRY_ISO3.lower()}_{distance_name}"
POTENTIAL_LOCATIONS_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.potential_locations_{COUNTRY_ISO3.lower()}_{distance_name}"
POTENTIAL_COVERAGE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.potential_coverage_{COUNTRY_ISO3.lower()}_{distance_name}"

# COMMAND ----------

print("Population AOI Table:", POPULATION_AOI_TABLE)
print("Facilities H3 Table:" , FACILITIES_H3_TABLE)
print("Facilities Coverage Table:", FACILITIES_COVERAGE_TABLE)
print("Potential Locations Table:", POTENTIAL_LOCATIONS_TABLE)
print("Potential Coverage Table:", POTENTIAL_COVERAGE_TABLE)

# COMMAND ----------

# SPATIAL UDFs

@udf(StringType())
def st_point_wkt(lon, lat):
    """Returns WKT string for a point."""
    if lon is None or lat is None:
        return None
    return Point(float(lon), float(lat)).wkt


print("Spatial UDFs registered.")

# COMMAND ----------

# HELPER FUNCTIONS

def table_exists(table_name: str) -> bool:
    """Check if UC table exists."""
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        return True
    except Exception:
        return False


def uc_table_to_gdf(table_name: str) -> gpd.GeoDataFrame:
    """Load Unity Catalog table as GeoDataFrame."""
    pdf = spark.table(table_name).toPandas()
    pdf["geometry"] = pdf["geometry_wkt"].apply(lambda w: wkt_loads(w) if w else None)
    pdf = pdf.drop(columns=["geometry_wkt"])
    return gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")

# COMMAND ----------

# LOAD EXTRACTED DATA FROM UC

print("Loading extracted data from Unity Catalog...")

selected_gadm_gdf = uc_table_to_gdf(GADM_TABLE)
print(f"  GADM boundaries: {len(selected_gadm_gdf)} features from {GADM_TABLE}")

facilities_gdf = uc_table_to_gdf(FACILITIES_TABLE)
print(f"  Health facilities: {len(facilities_gdf)} facilities from {FACILITIES_TABLE}")

boundary_wkt = selected_gadm_gdf.geometry.unary_union.wkt
aoi_union_geom = selected_gadm_gdf.geometry.unary_union

centroid = selected_gadm_gdf.iloc[0]["geometry"].centroid
CENTER_LAT = centroid.y
CENTER_LON = centroid.x
print(f"  Map center: lat={CENTER_LAT:.4f}, lon={CENTER_LON:.4f}")

# COMMAND ----------

# TRANSFORM: LOAD POPULATION AND FILTER TO AOI
# Long-running – takes 4 minutes to process

def load_population_aoi(
    source_table: str,
    output_table: str,
    boundary_wkt: str,
    h3_resolution: int,
    force: bool = False,
):
    """
    Loads population from UC table and filters to AOI using H3 index.
    Uses Photon-accelerated H3 functions for fast spatial filtering.
    Caches result to UC table.
    """
    if not force and table_exists(output_table):
        print(f"Population AOI already exists, loading: {output_table}")
        sdf = spark.table(output_table).cache()
        count = sdf.count()
        total = sdf.agg(F.sum("population")).collect()[0][0]
        print(f"  Loaded {count:,} pixels, population: {round(total):,}")
        return sdf

    print(f"Computing population AOI from: {source_table}")

    sdf = spark.table(source_table)
    total_rows = sdf.count()
    print(f"  Total pixels in table: {total_rows:,}")

    total_pop = sdf.agg(F.sum("population")).collect()[0][0]
    print(f"  Total population (country): {round(total_pop / 1_000_000, 2)} million")

    # Get H3 cells covering the AOI polygon (Photon-accelerated)
    print(f"  Computing H3 coverage at resolution {h3_resolution}...")
    aoi_h3_df = spark.sql(f"""
        SELECT explode(h3_polyfillash3('{boundary_wkt}', {h3_resolution})) as h3_index
    """)
    h3_count = aoi_h3_df.count()
    print(f"  H3 cells covering AOI: {h3_count:,}")

    # Filter population by H3 index (fast native join)
    sdf_aoi = sdf.join(F.broadcast(aoi_h3_df), on="h3_index", how="inner")

    aoi_count = sdf_aoi.count()
    print(f"  Population pixels in AOI: {aoi_count:,}")

    # Add ID column
    sdf_aoi = (
        sdf_aoi.withColumn("geom_wkt", st_point_wkt(F.col("xcoord"), F.col("ycoord")))
        .withColumn("row_id", F.monotonically_increasing_id())
        .withColumn("ID", F.concat(F.col("row_id").cast(StringType()), F.lit("_pop")))
    )

    aoi_total = sdf_aoi.agg(F.sum("population")).collect()[0][0]
    print(f"  Total population (AOI): {round(aoi_total):,}")

    thresholds = sdf_aoi.approxQuantile("population", [0.25, 0.5, 0.75], 0.01)
    sdf_aoi = sdf_aoi.withColumn(
        "opacity",
        F.when(F.col("population") <= thresholds[0], F.lit(0.1))
        .when(F.col("population") <= thresholds[1], F.lit(0.3))
        .when(F.col("population") <= thresholds[2], F.lit(0.6))
        .otherwise(F.lit(1.0)),
    )

    # Save to UC table
    sdf_aoi.write.mode("overwrite").saveAsTable(output_table)
    print(f"  Saved to: {output_table}")

    return spark.table(output_table).cache()


population_aoi_sdf = load_population_aoi(
    POPULATION_TABLE, POPULATION_AOI_TABLE, boundary_wkt, H3_RESOLUTION, FORCE_RECOMPUTE
)
population_aoi_sdf.count()
total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]

# COMMAND ----------

# TRANSFORM: HEALTH FACILITIES TO SPARK DATAFRAME

def facilities_gdf_to_spark(gdf: gpd.GeoDataFrame):
    """Converts GeoDataFrame to Spark DataFrame with WKT geometry."""
    gdf = gdf.copy()
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    gdf["geom_wkt"] = gdf.geometry.apply(lambda g: g.wkt)

    # Rename 'id' to 'osm_id' if present (avoid Spark case-insensitive collision with 'ID')
    if "id" in gdf.columns:
        gdf = gdf.rename(columns={"id": "osm_id"})

    cols = [c for c in gdf.columns if c != "geometry"]
    pdf = gdf[cols].copy()

    sdf = spark.createDataFrame(pdf)

    # Add ID column if not present (case-insensitive check)
    existing_cols_lower = [c.lower() for c in sdf.columns]
    if "id" not in existing_cols_lower:
        sdf = (
            sdf.withColumn("row_id", F.monotonically_increasing_id())
            .withColumn("ID", F.concat(F.col("row_id").cast(StringType()), F.lit("_current")))
        )

    return sdf.cache()


selected_hosp_sdf = facilities_gdf_to_spark(facilities_gdf)
print(f"Existing facilities loaded: {selected_hosp_sdf.count()}")

# COMMAND ----------

# TRANSFORM: GENERATE POTENTIAL FACILITY LOCATIONS

def generate_grid_in_polygon(spacing: float, geometry) -> pd.DataFrame:
    """Generates a regular point grid within the given geometry."""
    minx, miny, maxx, maxy = geometry.bounds
    x_coords = np.arange(np.floor(minx), np.ceil(maxx), spacing)
    y_coords = np.arange(np.floor(miny), np.ceil(maxy), spacing)
    mesh = np.meshgrid(x_coords, y_coords)
    pdf = pd.DataFrame({"longitude": mesh[0].flatten(), "latitude": mesh[1].flatten()})
    gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf.longitude, pdf.latitude), crs="EPSG:4326")
    gdf = gpd.clip(gdf, geometry).reset_index(drop=True)
    print(f"  Grid points: {len(gdf)}")
    return gdf[["longitude", "latitude"]]


def generate_kmeans(population_sdf, n_clusters: int) -> pd.DataFrame:
    """Runs KMeans on population coordinates to generate candidate locations."""
    sample_fraction = min(1.0, 500_000 / total_population)
    pop_sample_pdf = (
        population_sdf.select("xcoord", "ycoord")
        .sample(fraction=sample_fraction, seed=42)
        .toPandas()
    )
    coords = pop_sample_pdf[["xcoord", "ycoord"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_
    pdf = pd.DataFrame(centroids, columns=["longitude", "latitude"])
    print(f"  KMeans centroids: {len(pdf)}")
    return pdf


def locations_pdf_to_spark(pdf: pd.DataFrame, id_suffix: str = "_potential"):
    """Converts potential locations DataFrame to Spark."""
    sdf = spark.createDataFrame(pdf)
    sdf = (
        sdf.withColumn("geom_wkt", st_point_wkt(F.col("longitude"), F.col("latitude")))
        .withColumn("lon", F.col("longitude"))
        .withColumn("lat", F.col("latitude"))
        .withColumn("row_id", F.monotonically_increasing_id())
        .withColumn("ID", F.concat(F.col("row_id").cast(StringType()), F.lit(id_suffix)))
    )
    return sdf


print("Generating potential facility locations...")
if POTENTIAL_TYPE == "grid":
    potential_pdf = generate_grid_in_polygon(spacing=GRID_SPACING, geometry=aoi_union_geom)
else:
    potential_pdf = generate_kmeans(population_aoi_sdf, N_CLUSTERS)

potential_locations_sdf = locations_pdf_to_spark(potential_pdf).cache()
potential_locations_sdf.count()

# COMMAND ----------

# TRANSFORM: ADD H3 INDEX TO FACILITIES

def add_facility_h3_index(facilities_sdf, h3_resolution: int):
    """Adds H3 index to facilities based on their location."""
    return facilities_sdf.withColumn(
        "h3_index",
        F.expr(f"h3_longlatash3(lon, lat, {h3_resolution})")
    )


print(f"Adding H3 index to facilities (resolution {H3_RESOLUTION})...")
selected_hosp_sdf = add_facility_h3_index(selected_hosp_sdf, H3_RESOLUTION).cache()
potential_locations_sdf = add_facility_h3_index(potential_locations_sdf, H3_RESOLUTION).cache()

selected_hosp_sdf.count()
potential_locations_sdf.count()
print("  H3 indexes added.")

# COMMAND ----------

# TRANSFORM: COMPUTE POPULATION COVERAGE
# Long-running – takes 8 minutes to process

def _compute_coverage_h3_internal(facilities_sdf, population_sdf, h3_resolution: int, k_rings: int):
    """
    Computes which population points fall inside each facility's catchment using H3 grid rings.
    Uses distributed Spark joins instead of Python loops.

    k_rings: number of H3 rings around facility (determines catchment radius)
    """
    fac_count = facilities_sdf.count()
    pop_count = population_sdf.count()
    print(f"  Computing coverage: {fac_count} facilities x {pop_count:,} pop points (H3 k={k_rings})...")

    # Get H3 cells within k rings of each facility
    fac_h3_sdf = facilities_sdf.select(
        F.col("ID").alias("facility_ID"),
        F.explode(
            F.expr(f"h3_kring(h3_index, {k_rings})")
        ).alias("h3_index")
    )

    # Join facilities H3 cells with population H3 indexes
    coverage_sdf = fac_h3_sdf.join(
        population_sdf.select(
            F.col("ID").alias("pop_ID"),
            "h3_index",
            "population"
        ),
        on="h3_index",
        how="inner"
    ).drop("h3_index")

    # Aggregate coverage per facility
    facility_coverage_sdf = coverage_sdf.groupBy("facility_ID").agg(
        F.sum("population").alias("pop_with_access")
    )

    # Join back to facilities
    result_sdf = facilities_sdf.join(
        facility_coverage_sdf.withColumnRenamed("facility_ID", "ID"),
        on="ID",
        how="left"
    ).fillna({"pop_with_access": 0.0})

    # Flat coverage table (facility_ID, pop_ID pairs)
    flat_sdf = coverage_sdf.select("facility_ID", "pop_ID").distinct()

    return result_sdf, flat_sdf


def compute_coverage_h3(
    facilities_sdf,
    population_sdf,
    h3_resolution: int,
    k_rings: int,
    facilities_output_table: str,
    coverage_output_table: str,
    force: bool = False,
):
    """
    Wrapper for coverage computation with UC table caching.
    Coverage table is not cached - it's large and only used once for aggregation.
    """
    if not force and table_exists(facilities_output_table) and table_exists(coverage_output_table):
        print(f"  Loading from UC (lazy)...")
        fac_sdf = spark.table(facilities_output_table).cache()
        cov_sdf = spark.table(coverage_output_table)  # No cache - too large
        print(f"    Facilities: {fac_sdf.count()}")
        return fac_sdf, cov_sdf

    result_sdf, flat_sdf = _compute_coverage_h3_internal(
        facilities_sdf, population_sdf, h3_resolution, k_rings
    )

    # Save to UC tables
    result_sdf.write.mode("overwrite").saveAsTable(facilities_output_table)
    flat_sdf.write.mode("overwrite").saveAsTable(coverage_output_table)
    print(f"  Saved to: {facilities_output_table}, {coverage_output_table}")

    return spark.table(facilities_output_table).cache(), spark.table(coverage_output_table)


print(f"Computing coverage (H3 resolution={H3_RESOLUTION}, k_rings={K_RINGS}, ~{K_RINGS * H3_EDGE_LENGTH_M[H3_RESOLUTION]}m)...")
print("  Existing facilities...")
selected_hosp_sdf, hosp_coverage_sdf = compute_coverage_h3(
    selected_hosp_sdf, population_aoi_sdf, H3_RESOLUTION, K_RINGS,
    FACILITIES_H3_TABLE, FACILITIES_COVERAGE_TABLE, FORCE_RECOMPUTE
)

print("  Potential locations...")
potential_locations_sdf, potential_coverage_sdf = compute_coverage_h3(
    potential_locations_sdf, population_aoi_sdf, H3_RESOLUTION, K_RINGS,
    POTENTIAL_LOCATIONS_TABLE, POTENTIAL_COVERAGE_TABLE, FORCE_RECOMPUTE
)

# COMMAND ----------

display(potential_coverage_sdf)

# COMMAND ----------

# ANALYZE: CURRENT COVERAGE

existing_covered_ids_sdf = hosp_coverage_sdf.select("pop_ID").distinct()

pop_with_access_sdf = population_aoi_sdf.join(
    existing_covered_ids_sdf,
    population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
    "inner",
).drop("pop_ID")

pop_without_access_sdf = population_aoi_sdf.join(
    existing_covered_ids_sdf,
    population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
    "left_anti",
)

covered_pop_val = pop_with_access_sdf.agg(F.sum("population")).collect()[0][0]
current_access = round(covered_pop_val * 100 / total_population, 2)
print(f"Current population with access: {current_access}%")

# Maximum possible coverage
all_covered_ids_sdf = (
    hosp_coverage_sdf.select("pop_ID")
    .union(potential_coverage_sdf.select("pop_ID"))
    .distinct()
)

max_covered_pop = (
    population_aoi_sdf.join(
        all_covered_ids_sdf,
        population_aoi_sdf["ID"] == all_covered_ids_sdf["pop_ID"],
        "inner",
    )
    .agg(F.sum("population"))
    .collect()[0][0]
)

max_access_possible = round(max_covered_pop * 100 / total_population, 2)
print(f"Maximum access attainable: {max_access_possible}%")

# COMMAND ----------

# VISUALIZE: CURRENT COVERAGE MAP

_POP_VIZ_SAMPLE = 5_000

pop_with_access_pdf = pop_with_access_sdf.sample(
    fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_with_access_sdf.count())), seed=1
).toPandas()

pop_without_access_pdf = pop_without_access_sdf.sample(
    fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_without_access_sdf.count())), seed=2
).toPandas()

selected_hosp_pdf_viz = selected_hosp_sdf.select("ID", "lon", "lat", "pop_with_access").toPandas()

folium_map = fl.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=8, tiles="OpenStreetMap")
geo_adm = fl.GeoJson(
    data=selected_gadm_gdf.iloc[0]["geometry"].__geo_interface__,
    style_function=lambda x: {"color": "orange"},
)
geo_adm.add_to(folium_map)

for _, row in selected_hosp_pdf_viz.iterrows():
    fl.Marker([row["lat"], row["lon"]], icon=fl.Icon(color="blue")).add_to(folium_map)

for _, row in pop_without_access_pdf.iterrows():
    fl.CircleMarker(
        location=[row["ycoord"], row["xcoord"]],
        radius=5,
        color=None,
        fill=True,
        fill_color="red",
        fill_opacity=row["opacity"],
    ).add_to(folium_map)

for _, row in pop_with_access_pdf.iterrows():
    fl.CircleMarker(
        location=[row["ycoord"], row["xcoord"]],
        radius=5,
        color=None,
        fill=True,
        fill_color="green",
        fill_opacity=row["opacity"],
    ).add_to(folium_map)

folium_map

# COMMAND ----------

# OPTIMIZE: PREPARE INPUTS (AGGREGATED BY H3 CELL)

print("Preparing optimization inputs (aggregated by H3 cell)...")

# Aggregate population by H3 cell to reduce problem size
pop_by_h3_sdf = population_aoi_sdf.groupBy("h3_index").agg(
    F.sum("population").alias("population")
)
pop_h3_rows = pop_by_h3_sdf.collect()
w = {row["h3_index"]: float(row["population"]) for row in pop_h3_rows}
I = sorted(w.keys())
print(f"  Demand cells (H3): {len(I):,}")

# Facility IDs
hosp_id_rows = selected_hosp_sdf.select("ID").collect()
potential_id_rows = potential_locations_sdf.select("ID").collect()
J_existing = sorted(row["ID"] for row in hosp_id_rows)
J_potential = sorted(row["ID"] for row in potential_id_rows)
J = sorted(set(J_existing) | set(J_potential))
print(f"  Facilities: {len(J):,} (existing: {len(J_existing)}, potential: {len(J_potential)})")

# Coverage: which H3 cells are covered by which facilities
all_coverage_sdf = hosp_coverage_sdf.select("facility_ID", "pop_ID").union(
    potential_coverage_sdf.select("facility_ID", "pop_ID")
)

# Join coverage with population to get H3 index, then aggregate
coverage_h3_sdf = all_coverage_sdf.join(
    population_aoi_sdf.select("ID", "h3_index"),
    all_coverage_sdf["pop_ID"] == population_aoi_sdf["ID"],
    "inner"
).select("facility_ID", "h3_index").distinct()

JI_rows = (
    coverage_h3_sdf.groupBy("h3_index")
    .agg(F.collect_set("facility_ID").alias("fac_ids"))
    .collect()
)

JI = {row["h3_index"]: list(row["fac_ids"]) for row in JI_rows}
print(f"  Coverage pairs: {sum(len(v) for v in JI.values()):,}")

# COMMAND ----------

# OPTIMIZE: GREEDY MCLP (SCALABLE)

def solve_mclp_greedy(w, IJ, J_existing, J_potential, max_new_facilities):
    """
    Greedy Maximum Covering Location Problem.

    Args:
        w: dict of {h3_cell: population}
        IJ: dict of {h3_cell: [facility_ids that cover it]}
        J_existing: list of existing facility IDs
        J_potential: list of potential facility IDs
        max_new_facilities: maximum number of new facilities to add

    Returns list of results for p=1..max_new_facilities
    """
    # Build reverse index: facility -> set of H3 cells it covers
    facility_covers = {}
    for h3_cell, fac_list in IJ.items():
        for fac in fac_list:
            if fac not in facility_covers:
                facility_covers[fac] = set()
            facility_covers[fac].add(h3_cell)

    # Initialize with existing facilities
    selected = set(J_existing)
    covered_h3 = set()
    for fac in J_existing:
        if fac in facility_covers:
            covered_h3.update(facility_covers[fac])

    current_coverage = sum(w.get(h3, 0) for h3 in covered_h3)

    results = []
    candidates = set(J_potential) - selected

    for p in range(1, max_new_facilities + 1):
        best_fac = None
        best_gain = 0

        # Find facility with maximum marginal gain
        for fac in candidates:
            if fac not in facility_covers:
                continue
            new_cells = facility_covers[fac] - covered_h3
            gain = sum(w.get(h3, 0) for h3 in new_cells)
            if gain > best_gain:
                best_gain = gain
                best_fac = fac

        if best_fac is None or best_gain == 0:
            print(f"  No further improvement at p={p}. Stopping.")
            break

        # Add best facility
        selected.add(best_fac)
        candidates.remove(best_fac)
        covered_h3.update(facility_covers[best_fac])
        current_coverage += best_gain

        results.append({
            "p": p,
            "objective": current_coverage,
            "selected_facilities": list(selected),
            "covered_h3": list(covered_h3),
        })

        print(f"  p={p} | +{best_gain:.0f} | Total covered: {current_coverage:.0f} | Facilities: {len(selected)}")

    return results


print("Running greedy MCLP optimization...")
pareto_results = solve_mclp_greedy(w, JI, J_existing, J_potential, TARGET_NEW_FACILITIES + 5)

# COMMAND ----------

# VISUALIZE: PARETO FRONTIER

x_values = [len(J_existing) + item["p"] for item in pareto_results]
y_values = [round(100 * item["objective"] / total_population, 2) for item in pareto_results]

x_values.insert(0, len(J_existing))
y_values.insert(0, current_access)

fig = go.Figure(
    data=go.Scatter(x=x_values, y=y_values, mode="lines+markers", name="Pareto Frontier")
)
fig.update_layout(
    xaxis_title="Number of facilities (existing + new)",
    yaxis_title="Percentage of population with access",
    plot_bgcolor="white",
    yaxis=dict(range=[0, 100]),
    xaxis=dict(range=[0, len(J_potential) + len(J_existing)]),
    width=1200,
)
fig.add_vline(x=len(J_existing), line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(
    x=len(J_existing),
    y=current_access,
    text="Number of<br>existing facilities",
    showarrow=True,
    arrowhead=1,
    ax=90,
    ay=-30,
)
fig.add_hline(y=max_access_possible, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(
    x=len(J_existing) + 20,
    y=max_access_possible,
    text="Maximum access possible",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-30,
)
fig.show()

# COMMAND ----------

# VISUALIZE: OPTIMIZED RESULT FOR TARGET_NEW_FACILITIES

entry = next(item for item in pareto_results if item["p"] == TARGET_NEW_FACILITIES)
opened_ids = set(entry["selected_facilities"])
covered_h3_set = set(entry["covered_h3"])

new_facility_ids = opened_ids - set(J_existing)

new_fac_pdf = (
    potential_locations_sdf.filter(F.col("ID").isin(new_facility_ids))
    .select("ID", "lon", "lat")
    .toPandas()
)

# Join by H3 index to get covered/uncovered population
covered_h3_sdf = spark.createDataFrame(pd.DataFrame({"h3_covered": list(covered_h3_set)}))
pop_covered_sdf = population_aoi_sdf.join(
    covered_h3_sdf,
    population_aoi_sdf["h3_index"] == covered_h3_sdf["h3_covered"],
    "inner",
).drop("h3_covered")
pop_uncovered_sdf = population_aoi_sdf.join(
    covered_h3_sdf,
    population_aoi_sdf["h3_index"] == covered_h3_sdf["h3_covered"],
    "left_anti",
)

pop_covered_pdf = pop_covered_sdf.sample(
    fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_covered_sdf.count())), seed=3
).toPandas()
pop_uncovered_pdf = pop_uncovered_sdf.sample(
    fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_uncovered_sdf.count())), seed=4
).toPandas()

folium_map = fl.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=8, tiles="OpenStreetMap")
geo_adm = fl.GeoJson(
    data=selected_gadm_gdf.iloc[0]["geometry"].__geo_interface__,
    style_function=lambda x: {"color": "orange"},
)
geo_adm.add_to(folium_map)

for _, row in selected_hosp_pdf_viz.iterrows():
    fl.Marker([row["lat"], row["lon"]], icon=fl.Icon(color="blue")).add_to(folium_map)

for _, row in new_fac_pdf.iterrows():
    fl.Marker([row["lat"], row["lon"]], icon=fl.Icon(color="darkpurple")).add_to(folium_map)

for _, row in pop_uncovered_pdf.iterrows():
    fl.CircleMarker(
        location=[row["ycoord"], row["xcoord"]],
        radius=5,
        color=None,
        fill=True,
        fill_color="red",
        fill_opacity=row["opacity"],
    ).add_to(folium_map)

for _, row in pop_covered_pdf.iterrows():
    fl.CircleMarker(
        location=[row["ycoord"], row["xcoord"]],
        radius=5,
        color=None,
        fill=True,
        fill_color="green",
        fill_opacity=row["opacity"],
    ).add_to(folium_map)

print(f"Optimized result with {TARGET_NEW_FACILITIES} new facilities:")
print(f"  New facility IDs: {list(new_facility_ids)}")
print(f"  Coverage: {round(100 * entry['objective'] / total_population, 2)}%")

folium_map

# COMMAND ----------

new_fac_pdf

# COMMAND ----------

TARGET_ACCESS_RATE_PCT = 90.0

# UC table names
LGU_TABLE             = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_lgu_zambia"
LGU_ACCESSIBILITY_TABLE = (
    f"{UC_CATALOG}.{UC_SCHEMA}.lgu_accessibility_results_{COUNTRY_ISO3.lower()}_{distance_name}"
)


# COMMAND ----------

def sanitize_col_name(name: str) -> str:
    """
    Converts an LGU display name into a Delta-safe column name.
    Rules applied (in order):
      1. Strip leading/trailing whitespace.
      2. Replace any character that is not alphanumeric or underscore
         with an underscore  (covers spaces, commas, parens, etc.)
      3. Collapse consecutive underscores to a single one.
      4. Strip leading/trailing underscores.
      5. Prefix with 'lgu_' so the name never starts with a digit.
    Examples:
      "Kapiri Mposhi"  → "lgu_Kapiri_Mposhi"
      "Choma (East)"   → "lgu_Choma_East"
      "Lusaka"         → "lgu_Lusaka"
    """
    s = name.strip()
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return f"lgu_{s}"


print("Building H3 → LGU mapping via Photon polyfill ...")

lgu_raw_sdf = spark.table(LGU_TABLE).select("LGU", "geometry_wkt")

h3_lgu_sdf = (
    lgu_raw_sdf
    .select(
        F.col("LGU"),
        F.explode(
            F.expr(f"h3_polyfillash3(geometry_wkt, {H3_RESOLUTION})")
        ).alias("h3_index"),
    )
    .cache()
)

_lgu_h3_count = h3_lgu_sdf.count()

# Collect all raw LGU names and build the sanitization mapping
lgu_names_raw = sorted(
    row["LGU"] for row in h3_lgu_sdf.select("LGU").distinct().collect()
)

# name_map  : raw display name  → safe Delta column name
# name_map_r: safe column name  → raw display name  (for display)
name_map   = {lgu: sanitize_col_name(lgu) for lgu in lgu_names_raw}
name_map_r = {v: k for k, v in name_map.items()}
lgu_col_names = [name_map[lgu] for lgu in lgu_names_raw]   # safe names, same order

print(f"  {len(lgu_names_raw)} LGUs → {_lgu_h3_count:,} H3 cells mapped")
print(f"  Sample name mapping:")
for raw, safe in list(name_map.items())[:5]:
    print(f"    '{raw}'  →  '{safe}'")

# COMMAND ----------

print("Pre-aggregating population by (h3_index, LGU) ...")

# Join population AOI table with H3→LGU mapping
h3_lgu_pop_pdf = (
    population_aoi_sdf                       # h3_index, population, …
    .select("h3_index", "population")
    .join(h3_lgu_sdf, on="h3_index", how="inner")   # adds LGU column
    .groupBy("h3_index", "LGU")
    .agg(F.sum("population").alias("pop"))
    .toPandas()                                      # collected once; ~manageable size
)
print(f"  Pre-aggregated {len(h3_lgu_pop_pdf):,} (h3, LGU) rows")

# Per-LGU total population (denominator for % calculation)
lgu_total_pop: dict[str, float] = (
    h3_lgu_pop_pdf.groupby("LGU")["pop"].sum().to_dict()
)

# Lookup table for potential locations  (ID → lat / lon / h3_index)
potential_lookup = (
    potential_locations_sdf
    .select("ID", "lat", "lon", "h3_index")
    .toPandas()
    .set_index("ID")
)

print("Pre-computation complete. Ready for per-step accessibility calculation.")



# COMMAND ----------

print("Computing per-step LGU accessibility ...")
n_existing = len(J_existing)
result_rows = []

for idx, step in enumerate(pareto_results):
    p            = step["p"]
    covered_h3_set = set(step["covered_h3"])

    # ── Identify the single new facility added in this step ──────────────
    prev_selected = (
        set(pareto_results[idx - 1]["selected_facilities"])
        if idx > 0
        else set(J_existing)
    )
    new_fac_ids = set(step["selected_facilities"]) - prev_selected
    new_fac_id  = new_fac_ids.pop() if new_fac_ids else None

    # Retrieve location info from potential_locations lookup
    if new_fac_id and new_fac_id in potential_lookup.index:
        fac_lat = float(potential_lookup.at[new_fac_id, "lat"])
        fac_lon = float(potential_lookup.at[new_fac_id, "lon"])
        fac_h3  = str(potential_lookup.at[new_fac_id, "h3_index"])
    else:
        fac_lat = fac_lon = fac_h3 = None

    # ── National accessibility % ─────────────────────────────────────────
    national_access_pct = round(step["objective"] * 100.0 / total_population, 2)

    # ── Per-LGU accessibility (vectorised pandas, fast) ──────────────────
    # Filter the pre-aggregated (h3, LGU, pop) table to covered H3 cells only
    covered_mask  = h3_lgu_pop_pdf["h3_index"].isin(covered_h3_set)
    lgu_covered   = (
        h3_lgu_pop_pdf[covered_mask]
        .groupby("LGU")["pop"]
        .sum()
    )

    # ── Assemble result row ───────────────────────────────────────────────
    row = {
        "total_facilities":           n_existing + p,
        "new_facility":               new_fac_id,
        "lat":                        fac_lat,
        "lon":                        fac_lon,
        "h3_index":                   fac_h3,
        "total_population_access_pct": national_access_pct,
    }
    for lgu_raw in lgu_names_raw:
        safe_col = name_map[lgu_raw]                    # e.g. "lgu_Kapiri_Mposhi"
        total    = lgu_total_pop.get(lgu_raw, 0.0)
        covered  = float(lgu_covered.get(lgu_raw, 0.0))
        row[safe_col] = round(covered * 100.0 / total, 2) if total > 0 else 0.0

    result_rows.append(row)
    print(
        f"  Step {p:3d} | +1 facility ({new_fac_id}) "
        f"→ national {national_access_pct:.2f}%"
    )


# ── STEP: Identify how many facilities are needed to reach TARGET for all LGUs
result_pdf = pd.DataFrame(result_rows)

def all_lgus_above_target(row, col_names, target):
    return all(row[col] >= target for col in col_names)

target_row = result_pdf[
    result_pdf.apply(
        all_lgus_above_target, axis=1,
        col_names=lgu_col_names,
        target=TARGET_ACCESS_RATE_PCT,
    )
]
if not target_row.empty:
    first = target_row.iloc[0]
    print(
        f"✅  All {len(lgu_names_raw)} LGUs reach ≥{TARGET_ACCESS_RATE_PCT}% access at "
        f"{int(first['total_facilities'])} total facilities "
        f"({int(first['total_facilities']) - n_existing} new facilities needed)."
    )
else:
    last = result_pdf.iloc[-1]
    # Find how many LGUs are still below target
    below = [
        lgu_raw for lgu_raw, safe in name_map.items()
        if last[safe] < TARGET_ACCESS_RATE_PCT
    ]
    print(
        f"⚠️  After {TARGET_NEW_FACILITIES} new facilities, "
        f"{len(below)} LGUs are still below {TARGET_ACCESS_RATE_PCT}%:\n"
        f"  {below}\n"
        f"  Increase TARGET_NEW_FACILITIES and re-run solve_mclp_greedy."
    )
print("=" * 60)


# COMMAND ----------

# cell 1b — Spatial join: assign 'district' to each new facility in result_pdf

# ── Load district boundaries ──────────────────────────────────────────────────
print("Loading district boundaries ...")
boundaries_sdf = spark.table("prd_mega.sgpbpi163.gadm_boundaries_lgu_zambia")
boundaries_pdf = boundaries_sdf.select("LGU", "geometry_wkt").toPandas()

# Parse WKT strings into Shapely geometries (skip any nulls/malformed rows)
boundaries_pdf["geometry"] = boundaries_pdf["geometry_wkt"].apply(
    lambda w: shapely_wkt.loads(w) if isinstance(w, str) and w.strip() else None
)
boundaries_pdf = boundaries_pdf.dropna(subset=["geometry"]).reset_index(drop=True)

print(f"  Loaded {len(boundaries_pdf)} district polygons.")

# ── Point-in-polygon lookup ───────────────────────────────────────────────────
def find_district(lat, lon):
    """Return the LGU name whose polygon contains (lon, lat), else None."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    pt = Point(lon, lat)          # Shapely Point is (x=lon, y=lat)
    for _, row in boundaries_pdf.iterrows():
        if row["geometry"].contains(pt):
            return row["LGU"]
    return None                   # point falls outside all boundaries

print("Assigning district to each facility step ...")
result_pdf["district"] = result_pdf.apply(
    lambda r: find_district(r["lat"], r["lon"]), axis=1
)

# Quick sanity check
unmatched = result_pdf["district"].isna().sum()
if unmatched:
    print(f"  ⚠️  {unmatched} row(s) could not be matched to a district "
          f"(likely existing / null facilities).")
print("  ✅ 'district' column added.")
print(result_pdf[["total_facilities", "new_facility", "lat", "lon", "district"]].to_string())

# COMMAND ----------

print(f"\nWriting LGU accessibility results to: {LGU_ACCESSIBILITY_TABLE}")

# Enforce float type for all LGU columns (Spark requires uniform types)
for col in lgu_col_names:
    result_pdf[col] = result_pdf[col].astype(float)
result_pdf["lat"] = result_pdf["lat"].astype(float)
result_pdf["lon"] = result_pdf["lon"].astype(float)
result_pdf["total_population_access_pct"] = result_pdf["total_population_access_pct"].astype(float)

result_sdf = spark.createDataFrame(result_pdf)
result_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    LGU_ACCESSIBILITY_TABLE
)

print(
    f"Saved: {len(result_pdf)} rows × {len(result_pdf.columns)} columns\n"
    f"  Columns: total_facilities, new_facility, lat, lon, h3_index, "
    f"total_population_access_pct, [{len(lgu_col_names)} LGU columns]"
)
display(result_sdf)


# COMMAND ----------

display(result_sdf)

# COMMAND ----------

  front_cols = ["total_facilities", "new_facility", "district", "lat", "lon", "h3_index", "total_population_access_pct"]
  result_sdf = result_sdf.select(front_cols + lgu_col_names)

# COMMAND ----------

display(result_sdf)
