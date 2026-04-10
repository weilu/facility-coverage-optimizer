# Databricks notebook source
# MAGIC %pip install "numpy<2" geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Compute population coverage
#
# Dependencies:
#   - transform/01_prepare.py (Population AOI, Facilities H3, Potential locations)
#
# Outputs:
#   - Facilities coverage table (facility_ID, pop_ID pairs)
#   - Potential coverage table (facility_ID, pop_ID pairs)
#
# This task computes which population points fall inside each facility's
# catchment area using H3 grid rings. Long-running task (~8 min per coverage).

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# MAGIC %run "./config"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.env import get_spark, table_exists
    from transform.config import (
        COUNTRY,
        COUNTRY_ISO3,
        POPULATION_YEAR,
        FORCE_RECOMPUTE,
        H3_RESOLUTION,
        H3_EDGE_LENGTH_M,
        get_k_rings,
        get_transform_table_names,
        build_transform_combinations,
    )

# COMMAND ----------

spark = get_spark()

# COMMAND ----------

# COVERAGE COMPUTATION FUNCTIONS

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
        print("  Loading from UC (lazy)...")
        fac_sdf = spark.table(facilities_output_table).cache()
        cov_sdf = spark.table(coverage_output_table)  # No cache - too large
        print(f"    Facilities: {fac_sdf.count()}")
        return fac_sdf, cov_sdf

    result_sdf, flat_sdf = _compute_coverage_h3_internal(
        facilities_sdf, population_sdf, h3_resolution, k_rings
    )

    # Save to UC tables
    result_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(facilities_output_table)
    flat_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(coverage_output_table)
    print(f"  Saved to: {facilities_output_table}, {coverage_output_table}")

    return spark.table(facilities_output_table).cache(), spark.table(coverage_output_table)

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
    print(f"COMPUTING COVERAGE: {region_name} @ {distance_name}")
    print("=" * 60)

    tables = get_transform_table_names(
        COUNTRY, COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )
    k_rings = get_k_rings(distance_meters, H3_RESOLUTION)

    print(f"  K_RINGS = {k_rings} (~{k_rings * H3_EDGE_LENGTH_M[H3_RESOLUTION]}m)")

    # Load prepared data
    print("\nLoading prepared data...")
    population_aoi_sdf = spark.table(tables["population_aoi"]).cache()
    selected_hosp_sdf = spark.table(tables["facilities_h3"]).cache()
    potential_locations_sdf = spark.table(tables["potential_locations"]).cache()

    print(f"  Population AOI: {population_aoi_sdf.count():,} pixels")
    print(f"  Facilities: {selected_hosp_sdf.count()}")
    print(f"  Potential locations: {potential_locations_sdf.count()}")

    # Compute coverage for existing facilities
    print("\nComputing coverage for existing facilities...")
    selected_hosp_sdf, hosp_coverage_sdf = compute_coverage_h3(
        selected_hosp_sdf, population_aoi_sdf, H3_RESOLUTION, k_rings,
        tables["facilities_h3"], tables["facilities_coverage"], FORCE_RECOMPUTE
    )

    # Compute coverage for potential locations
    print("\nComputing coverage for potential locations...")
    potential_locations_sdf, potential_coverage_sdf = compute_coverage_h3(
        potential_locations_sdf, population_aoi_sdf, H3_RESOLUTION, k_rings,
        tables["potential_locations"], tables["potential_coverage"], FORCE_RECOMPUTE
    )

    # Analyze current coverage
    print("\nAnalyzing current coverage...")
    total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]

    existing_covered_ids_sdf = hosp_coverage_sdf.select("pop_ID").distinct()

    pop_with_access_sdf = population_aoi_sdf.join(
        existing_covered_ids_sdf,
        population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
        "inner",
    ).drop("pop_ID")

    covered_pop_val = pop_with_access_sdf.agg(F.sum("population")).collect()[0][0]
    current_access = round(covered_pop_val * 100 / total_population, 2)
    print(f"  Current population with access: {current_access}%")

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
    print(f"  Maximum access attainable: {max_access_possible}%")

    print(f"\n  Completed: {region_name} @ {distance_name}")

# COMMAND ----------

print("\n" + "=" * 60)
print("COVERAGE COMPUTATION COMPLETE")
print("=" * 60)
print(f"Processed {len(transform_combinations)} combinations")
print("Next step: Run 03_optimize.py to run greedy MCLP")
