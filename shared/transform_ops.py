# Databricks notebook source
# Transform operations extracted from the transform notebooks so they can be
# imported and unit/integration tested. Pure geometry helpers use numpy/pandas/
# geopandas; the H3 helpers rely on Databricks-native H3 SQL (h3_longlatash3,
# h3_kring) and are only meaningful on a cluster.

import numpy as np
import pandas as pd
import geopandas as gpd


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


def add_facility_h3_index(facilities_sdf, h3_resolution: int):
    """Adds H3 index to facilities based on their location."""
    # pyspark imported lazily so pure-Python consumers (and PySpark-free CI) can
    # import this module without the pyspark package.
    from pyspark.sql import functions as F

    return facilities_sdf.withColumn(
        "h3_index",
        F.expr(f"h3_longlatash3(lon, lat, {h3_resolution})")
    )


def compute_coverage_h3_internal(facilities_sdf, population_sdf, h3_resolution: int, k_rings: int):
    """
    Computes which population points fall inside each facility's catchment using H3 grid rings.
    Uses distributed Spark joins instead of Python loops.

    k_rings: number of H3 rings around facility (determines catchment radius)
    """
    # pyspark imported lazily (see add_facility_h3_index).
    from pyspark.sql import functions as F

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

    # pop_with_access survives prior writes; drop before re-join to avoid duplicate-column conflict on re-run
    result_sdf = facilities_sdf.drop("pop_with_access").join(
        facility_coverage_sdf.withColumnRenamed("facility_ID", "ID"),
        on="ID",
        how="left"
    ).fillna({"pop_with_access": 0.0})

    # Flat coverage table (facility_ID, pop_ID pairs)
    flat_sdf = coverage_sdf.select("facility_ID", "pop_ID").distinct()

    return result_sdf, flat_sdf
