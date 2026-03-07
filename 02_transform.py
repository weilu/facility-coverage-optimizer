# Databricks notebook source
# MAGIC %pip install geopandas shapely rasterio pycountry gurobipy folium plotly scikit-learn pyproj

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
from pyproj import Transformer

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType, StructType, StructField
from pyspark.sql.functions import udf, pandas_udf

import gurobipy as gp
from gurobipy import GRB

print(f"Spark version: {spark.version}")

# COMMAND ----------

# CONFIGURATION

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"
COUNTRY_ISO3 = "ZMB"
POPULATION_YEAR = 2025

TRAVEL_API = ""  # "" for buffer, "osm", or "mapbox"
DISTANCE_METERS = 10000
MAPBOX_ACCESS_TOKEN = ""
MAPBOX_MODE = "driving"

POTENTIAL_TYPE = "grid"  # "grid" or "kmeans"
GRID_SPACING = 0.03
N_CLUSTERS = 100

TARGET_NEW_FACILITIES = 7

# Derived table names
GADM_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_{COUNTRY_ISO3.lower()}"
FACILITIES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{COUNTRY_ISO3.lower()}"
POPULATION_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.population_{COUNTRY_ISO3.lower()}_{POPULATION_YEAR}"

# COMMAND ----------

# SPATIAL UDFs

@udf(StringType())
def st_point_wkt(lon, lat):
    """Returns WKT string for a point."""
    if lon is None or lat is None:
        return None
    return Point(float(lon), float(lat)).wkt


@udf(BooleanType())
def st_within_wkt(point_wkt, polygon_wkt):
    """Returns True if point is within polygon."""
    if point_wkt is None or polygon_wkt is None:
        return False
    try:
        return wkt_loads(point_wkt).within(wkt_loads(polygon_wkt))
    except Exception:
        return False


@pandas_udf(StringType())
def st_buffer_meters(geom_wkt_series: pd.Series, distance_series: pd.Series) -> pd.Series:
    """Returns buffered geometry as WKT. Projects to Web Mercator for accurate distance."""
    from shapely.ops import transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    results = []
    for wkt, dist in zip(geom_wkt_series, distance_series):
        try:
            geom_3857 = transform(to_3857, wkt_loads(wkt))
            buffered_3857 = geom_3857.buffer(float(dist))
            buffered_4326 = transform(to_4326, buffered_3857)
            results.append(buffered_4326.wkt)
        except Exception:
            results.append(None)
    return pd.Series(results)


print("Spatial UDFs registered.")

# COMMAND ----------

# HELPER: LOAD GEODATAFRAME FROM UC TABLE

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

def load_population_aoi(table_name: str, boundary_wkt: str):
    """
    Loads population from UC table and filters to AOI.
    No raster processing needed - data already extracted to table.
    """
    print(f"Loading population from: {table_name}")

    sdf = spark.table(table_name)
    total_rows = sdf.count()
    print(f"  Total pixels in table: {total_rows:,}")

    sdf = (
        sdf.withColumn("geom_wkt", st_point_wkt(F.col("xcoord"), F.col("ycoord")))
        .withColumn("row_id", F.monotonically_increasing_id())
        .withColumn("ID", F.concat(F.col("row_id").cast(StringType()), F.lit("_pop")))
    )

    total_pop = sdf.agg(F.sum("population")).collect()[0][0]
    print(f"  Total population (country): {round(total_pop / 1_000_000, 2)} million")

    boundary_lit = F.lit(boundary_wkt)
    sdf_aoi = sdf.filter(st_within_wkt(F.col("geom_wkt"), boundary_lit))

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

    return sdf_aoi.cache()


population_aoi_sdf = load_population_aoi(POPULATION_TABLE, boundary_wkt)
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

# TRANSFORM: COMPUTE CATCHMENT AREAS

def compute_catchment_areas(facilities_sdf, distance_meters: int):
    """Adds catchment_area_wkt column using buffer."""
    dist_col = F.lit(float(distance_meters))
    return facilities_sdf.withColumn(
        "cachment_area_wkt",
        st_buffer_meters(F.col("geom_wkt"), dist_col),
    )


print(f"Computing catchment areas ({DISTANCE_METERS}m buffer)...")
selected_hosp_sdf = compute_catchment_areas(selected_hosp_sdf, DISTANCE_METERS).cache()
potential_locations_sdf = compute_catchment_areas(potential_locations_sdf, DISTANCE_METERS).cache()

selected_hosp_sdf.count()
potential_locations_sdf.count()
print("  Catchment areas computed.")

# COMMAND ----------

# TRANSFORM: COMPUTE POPULATION COVERAGE

def compute_coverage(facilities_sdf, population_sdf):
    """Computes which population points fall inside each facility's catchment."""
    fac_pdf = facilities_sdf.select("ID", "cachment_area_wkt").toPandas()
    fac_pdf = fac_pdf.dropna(subset=["cachment_area_wkt"])
    fac_shapes = {row["ID"]: wkt_loads(row["cachment_area_wkt"]) for _, row in fac_pdf.iterrows()}

    pop_pdf = population_sdf.select("ID", "xcoord", "ycoord", "population").toPandas()

    print(f"  Computing coverage: {len(fac_shapes)} facilities x {len(pop_pdf):,} pop points...")

    rows = []
    for fac_id, catchment in fac_shapes.items():
        mask = pop_pdf.apply(lambda r: Point(r["xcoord"], r["ycoord"]).within(catchment), axis=1)
        covered = pop_pdf[mask]
        rows.append({
            "facility_ID": fac_id,
            "covered_pop_ids": covered["ID"].tolist(),
            "pop_with_access": float(covered["population"].sum()),
        })

    coverage_pdf = pd.DataFrame(rows)
    coverage_sdf = spark.createDataFrame(coverage_pdf)

    result_sdf = facilities_sdf.join(
        coverage_sdf.withColumnRenamed("facility_ID", "ID"),
        on="ID",
        how="left",
    ).fillna({"pop_with_access": 0.0})

    flat_rows = [
        (fid, pid)
        for _, r in coverage_pdf.iterrows()
        for pid in r["covered_pop_ids"]
    ]
    flat_schema = StructType([
        StructField("facility_ID", StringType()),
        StructField("pop_ID", StringType()),
    ])
    flat_sdf = spark.createDataFrame(flat_rows, schema=flat_schema)

    return result_sdf, flat_sdf


print("Computing coverage for existing facilities...")
selected_hosp_sdf, hosp_coverage_sdf = compute_coverage(selected_hosp_sdf, population_aoi_sdf)

print("Computing coverage for potential locations...")
potential_locations_sdf, potential_coverage_sdf = compute_coverage(potential_locations_sdf, population_aoi_sdf)

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

_POP_VIZ_SAMPLE = 20_000

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

# OPTIMIZE: PREPARE GUROBI INPUTS

print("Collecting Gurobi inputs from Spark...")

w_rows = population_aoi_sdf.select("ID", "population").collect()
w = {row["ID"]: float(row["population"]) for row in w_rows}
I = sorted(w.keys())

hosp_id_rows = selected_hosp_sdf.select("ID").collect()
potential_id_rows = potential_locations_sdf.select("ID").collect()
J_existing = sorted(row["ID"] for row in hosp_id_rows)
J_potential = sorted(row["ID"] for row in potential_id_rows)
J = sorted(set(J_existing) | set(J_potential))

all_coverage_sdf = hosp_coverage_sdf.select("facility_ID", "pop_ID").union(
    potential_coverage_sdf.select("facility_ID", "pop_ID")
)

JI_rows = (
    all_coverage_sdf.groupBy("pop_ID")
    .agg(F.collect_list("facility_ID").alias("fac_ids"))
    .collect()
)

JI = {row["pop_ID"]: sorted(row["fac_ids"]) for row in JI_rows}

print(f"  Demand points (I): {len(I):,}")
print(f"  Facilities (J): {len(J):,}")
print(f"  Existing: {len(J_existing)}, Potential: {len(J_potential)}")

# COMMAND ----------

# OPTIMIZE: GUROBI MCLP

def model_max_covering_gurobi(w, I, J, JI, p, J_existing, model):
    """Gurobi Maximum Covering Location Problem (MCLP)."""
    model.remove(model.getVars())
    model.remove(model.getConstrs())

    x = model.addVars(J, vtype=GRB.BINARY, name="x")
    z = model.addVars(I, vtype=GRB.BINARY, name="z")

    model.setObjective(gp.quicksum(w[i] * z[i] for i in I), GRB.MAXIMIZE)

    for i in I:
        if JI.get(i):
            model.addConstr(z[i] <= gp.quicksum(x[j] for j in JI[i]), name=f"cover_{i}")
        else:
            model.addConstr(z[i] == 0, name=f"cover_{i}_zero")

    model.addConstr(gp.quicksum(x[j] for j in J) <= len(J_existing) + p, name="budget")

    for j in J_existing:
        model.addConstr(x[j] == 1, name=f"existing_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    x_sol = {j: x[j].X for j in J}
    z_sol = {i: z[i].X for i in I}

    selected_facilities = [j for j in J if x_sol[j] > 0.5]
    covered_demand = [i for i in I if z_sol[i] > 0.5]

    return model.ObjVal, selected_facilities, covered_demand


# Gurobi license params
params = {
    "WLSACCESSID": "REDACTED",
    "WLSSECRET": "REDACTED",
    "LICENSEID": 0,
}

env = gp.Env(params=params)
model = gp.Model(env=env)

pareto_gurobi = []
previous_obj = -1

print("Running Gurobi optimization...")
for p in range(1, len(J_potential) + 1):
    obj, selected_facilities, covered_demand = model_max_covering_gurobi(
        w, I, J, JI, p, J_existing, model
    )
    pareto_gurobi.append({
        "p": p,
        "objective": obj,
        "selected_facilities": selected_facilities,
        "covered_demand": covered_demand,
    })

    if round(obj) == round(previous_obj):
        print("  No further improvement. Stopping.")
        break

    previous_obj = obj
    print(f"  p={p} | Covered pop={obj:.0f} | Total facilities: {len(selected_facilities)}")

# COMMAND ----------

# VISUALIZE: PARETO FRONTIER

x_values = [len(J_existing) + item["p"] for item in pareto_gurobi]
y_values = [round(100 * item["objective"] / total_population, 2) for item in pareto_gurobi]

x_values.insert(0, len(J_existing))
y_values.insert(0, current_access)

fig = go.Figure(
    data=go.Scatter(x=x_values, y=y_values, mode="lines+markers", name="Gurobi Pareto Frontier")
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

entry = next(item for item in pareto_gurobi if item["p"] == TARGET_NEW_FACILITIES)
opened_ids = set(entry["selected_facilities"])
covered_ids_set = set(entry["covered_demand"])

new_facility_ids = opened_ids - set(J_existing)

new_fac_pdf = (
    potential_locations_sdf.filter(F.col("ID").isin(new_facility_ids))
    .select("ID", "lon", "lat")
    .toPandas()
)

covered_ids_sdf = spark.createDataFrame(pd.DataFrame({"pop_ID": list(covered_ids_set)}))
pop_covered_sdf = population_aoi_sdf.join(
    covered_ids_sdf,
    population_aoi_sdf["ID"] == covered_ids_sdf["pop_ID"],
    "inner",
).drop("pop_ID")
pop_uncovered_sdf = population_aoi_sdf.join(
    covered_ids_sdf,
    population_aoi_sdf["ID"] == covered_ids_sdf["pop_ID"],
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
