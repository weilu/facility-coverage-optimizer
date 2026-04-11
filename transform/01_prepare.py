# Databricks notebook source
# MAGIC %pip install "numpy<2" geopandas shapely scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Prepare data for optimization
#
# Dependencies:
#   - extract/02_population.py (Country-level population table)
#   - extract/03_boundaries.py (Province boundaries)
#   - extract/04_facilities.py (Health facilities)
#
# Outputs:
#   - Population AOI table (filtered by province boundary)
#   - Potential locations table (grid or kmeans)
#   - Facilities with H3 index
#
# This task loads extracted data, filters population to the AOI,
# and generates potential facility locations.

# COMMAND ----------

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# MAGIC %run "./config"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.env import get_spark, uc_table_to_gdf, table_exists
    from transform.config import (
        COUNTRY,
        COUNTRY_ISO3,
        POPULATION_YEAR,
        FORCE_RECOMPUTE,
        H3_RESOLUTION,
        POTENTIAL_TYPE,
        GRID_SPACING,
        N_CLUSTERS,
        get_transform_table_names,
        build_transform_combinations,
    )

# COMMAND ----------

spark = get_spark()

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


def generate_kmeans(population_sdf, n_clusters: int, total_population: float) -> pd.DataFrame:
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


def add_facility_h3_index(facilities_sdf, h3_resolution: int):
    """Adds H3 index to facilities based on their location."""
    return facilities_sdf.withColumn(
        "h3_index",
        F.expr(f"h3_longlatash3(lon, lat, {h3_resolution})")
    )

# COMMAND ----------

# EXECUTE TASK: Process each (province, distance) combination

transform_combinations = build_transform_combinations()
print(f"Will process {len(transform_combinations)} combination(s):")
for adm, dist in transform_combinations:
    region = adm if adm else "Country"
    print(f"  - {region} @ {int(dist/1000)}km")

# COMMAND ----------

for adm_level1, distance_meters in transform_combinations:
    print("\n" + "=" * 60)
    region_name = adm_level1 if adm_level1 else "Country"
    distance_name = f"{int(distance_meters / 1000)}km"
    print(f"PREPARING: {region_name} @ {distance_name}")
    print("=" * 60)

    tables = get_transform_table_names(
        COUNTRY, COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )

    # Load boundaries
    print(f"\nLoading boundaries from: {tables['boundaries']}")
    selected_boundary_gdf = uc_table_to_gdf(tables["boundaries"])
    print(f"  Boundaries: {len(selected_boundary_gdf)} features")

    boundary_wkt = selected_boundary_gdf.geometry.unary_union.wkt
    aoi_union_geom = selected_boundary_gdf.geometry.unary_union

    # Load and filter population to AOI
    print("\nFiltering population to AOI...")
    population_aoi_sdf = load_population_aoi(
        tables["population"], tables["population_aoi"],
        boundary_wkt, H3_RESOLUTION, FORCE_RECOMPUTE
    )
    total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]
    print(f"  Total population in AOI: {round(total_population):,}")

    # Load facilities
    print(f"\nLoading facilities from: {tables['facilities']}")
    facilities_gdf = uc_table_to_gdf(tables["facilities"])
    print(f"  Facilities: {len(facilities_gdf)}")

    selected_hosp_sdf = facilities_gdf_to_spark(facilities_gdf)
    print(f"  Converted to Spark: {selected_hosp_sdf.count()} facilities")

    # Generate potential locations
    print(f"\nGenerating potential facility locations ({POTENTIAL_TYPE})...")
    if POTENTIAL_TYPE == "grid":
        potential_pdf = generate_grid_in_polygon(spacing=GRID_SPACING, geometry=aoi_union_geom)
    else:
        potential_pdf = generate_kmeans(population_aoi_sdf, N_CLUSTERS, total_population)

    potential_locations_sdf = locations_pdf_to_spark(potential_pdf).cache()
    print(f"  Potential locations: {potential_locations_sdf.count()}")

    # Add H3 index to facilities
    print(f"\nAdding H3 index (resolution {H3_RESOLUTION})...")
    selected_hosp_sdf = add_facility_h3_index(selected_hosp_sdf, H3_RESOLUTION).cache()
    potential_locations_sdf = add_facility_h3_index(potential_locations_sdf, H3_RESOLUTION).cache()
    print("  H3 indexes added.")

    # Save potential locations to UC table
    print(f"\nSaving potential locations to: {tables['potential_locations']}")
    potential_locations_sdf.write.mode("overwrite").saveAsTable(tables["potential_locations"])

    # Save facilities with H3 to UC table
    print(f"Saving facilities with H3 to: {tables['facilities_h3']}")
    selected_hosp_sdf.write.mode("overwrite").saveAsTable(tables["facilities_h3"])

    print(f"\n  Completed: {region_name} @ {distance_name}")

# COMMAND ----------

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"Processed {len(transform_combinations)} combinations")
print("Next step: Run 02_coverage.py to compute coverage")
