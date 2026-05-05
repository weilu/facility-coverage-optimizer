# Databricks notebook source
# MAGIC %pip install "numpy<2" geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Run optimization and compute LGU accessibility metrics
#
# Dependencies:
#   - transform/02_coverage.py (Coverage tables)
#   - extract/03_boundaries.py (LGU boundaries)
#
# Outputs:
#   - LGU accessibility results table (per-step metrics for each LGU)
#   - Base dashboard data table (aggregated metadata for the fronend data app)
#
# This task runs the greedy Maximum Covering Location Problem algorithm
# to determine optimal locations for new facilities, then computes
# per-district accessibility metrics at each optimization step.

# COMMAND ----------

import pandas as pd
from shapely.geometry import Point
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# MAGIC %run "../shared/core"

# COMMAND ----------

# MAGIC %run "./config"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.env import get_spark, table_exists
    from shared.core import sanitize_col_name, solve_mclp_greedy
    from transform.config import (
        COUNTRY,
        COUNTRY_ISO3,
        POPULATION_YEAR,
        FORCE_RECOMPUTE,
        H3_RESOLUTION,
        TARGET_NEW_FACILITIES,
        TARGET_ACCESS_RATE_PCT,
        BASE_DASHBOARD_TABLE,
        get_transform_table_names,
        build_transform_combinations,
    )


from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

# COMMAND ----------

spark = get_spark()

# COMMAND ----------

# HELPER FUNCTIONS


def find_district(lat, lon, boundaries_pdf):
    """Return the LGU name whose polygon contains (lon, lat), else None."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    pt = Point(lon, lat)  # Shapely Point is (x=lon, y=lat)
    for _, row in boundaries_pdf.iterrows():
        if row["geometry"].contains(pt):
            return row["LGU"]
    return None  # point falls outside all boundaries


_DASHBOARD_SCHEMA = StructType([
    StructField("country",               StringType(), True),
    StructField("province",              StringType(), True),
    StructField("year",                  LongType(),   True),
    StructField("central_lat",           DoubleType(), True),
    StructField("central_long",          DoubleType(), True),
    StructField("distance_km",           LongType(),   True),
    StructField("total_new_facilities",  LongType(),   True),
    StructField("current_access",        DoubleType(), True),
    StructField("geometry_wkt",          StringType(), True),
])

def save_dashboard_metadata(
    spark,
    table_name: str,
    country: str,
    province: str | None,
    year: int,
    central_lat: float,
    central_long: float,
    distance_km: int,
    total_new_facilities: int,
    current_access: float,
    geometry_wkt: str,
    first_write: bool = False,
):
    row = {
        "country":              country,
        "province":             province,           # None is fine — StringType is nullable
        "year":                 int(year),
        "central_lat":          float(central_lat),
        "central_long":         float(central_long),
        "distance_km":          int(distance_km),
        "total_new_facilities": int(total_new_facilities),
        "current_access":       float(current_access),
        "geometry_wkt":         geometry_wkt,
    }
    pdf = pd.DataFrame([row])

    # ── Fix: undo pandas 3.x StringDtype inference before Arrow sees it ──
    str_cols = pdf.select_dtypes(include="string").columns
    pdf[str_cols] = pdf[str_cols].astype(object)

    # ── Fix: pass explicit schema so Spark doesn't guess ──
    sdf = spark.createDataFrame(pdf, schema=_DASHBOARD_SCHEMA)

    mode = "overwrite" if first_write else "append"
    sdf.write.mode(mode).option("overwriteSchema", "true").saveAsTable(table_name)
    print(pdf)
    print(f"  Dashboard metadata saved ({mode}): {table_name}")

# COMMAND ----------

# EXECUTE TASK: Process each (province, distance) combination

transform_combinations = build_transform_combinations()
print(f"Will process {len(transform_combinations)} combination(s):")
for adm, dist in transform_combinations:
    region = adm if adm else "Country"
    print(f"  - {region} @ {int(dist/1000)}km")

# COMMAND ----------

# COMPUTATION & STORAGE

skip_dashboard = not FORCE_RECOMPUTE and table_exists(BASE_DASHBOARD_TABLE)
if skip_dashboard:
    print(f"Dashboard table already exists: {BASE_DASHBOARD_TABLE}")
    print("Set FORCE_RECOMPUTE = True to regenerate.")

for adm_level1, distance_meters in transform_combinations:
    print("\n" + "=" * 60)
    region_name = adm_level1 if adm_level1 else "Country"
    distance_name = f"{int(distance_meters / 1000)}km"
    print(f"COMPUTING: {region_name} @ {distance_name}")
    print("=" * 60)

    tables = get_transform_table_names(
        COUNTRY, COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )

    # Check if already computed
    skip_optimization = (
        not FORCE_RECOMPUTE and table_exists(tables["lgu_accessibility"])
    )

    if skip_optimization and skip_dashboard:
        print(f"LGU accessibility table already exists: {tables['lgu_accessibility']}")
        continue

    if skip_optimization:
        print(f"LGU accessibility table already exists: {tables['lgu_accessibility']}")
        result_pdf = spark.table(tables["lgu_accessibility"]).toPandas()
    else:
        # Load data
        print("\nLoading data...")
        population_aoi_sdf = spark.table(tables["population_aoi"]).cache()
        selected_hosp_sdf = spark.table(tables["facilities_h3"]).cache()
        potential_locations_sdf = spark.table(tables["potential_locations"]).cache()
        hosp_coverage_sdf = spark.table(tables["facilities_coverage"])
        potential_coverage_sdf = spark.table(tables["potential_coverage"])

        total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]

        # Build optimization inputs
        print("\nBuilding optimization inputs...")
        pop_by_h3_sdf = population_aoi_sdf.groupBy("h3_index").agg(
            F.sum("population").alias("population")
        )
        pop_h3_rows = pop_by_h3_sdf.collect()
        w = {row["h3_index"]: float(row["population"]) for row in pop_h3_rows}

        hosp_id_rows = selected_hosp_sdf.select("ID").collect()
        potential_id_rows = potential_locations_sdf.select("ID").collect()
        J_existing = sorted(row["ID"] for row in hosp_id_rows)
        J_potential = sorted(row["ID"] for row in potential_id_rows)

        all_coverage_sdf = hosp_coverage_sdf.select("facility_ID", "pop_ID").union(
            potential_coverage_sdf.select("facility_ID", "pop_ID")
        )

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

        # Run greedy MCLP
        print("\nRunning greedy MCLP optimization...")
        pareto_results = solve_mclp_greedy(w, JI, J_existing, J_potential, TARGET_NEW_FACILITIES + 5)

        # Build H3 -> LGU mapping
        print("\nBuilding H3 -> LGU mapping...")
        lgu_raw_sdf = spark.table(tables["lgu"]).select("LGU", "geometry_wkt")

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

        name_map = {lgu: sanitize_col_name(lgu) for lgu in lgu_names_raw}
        lgu_col_names = [name_map[lgu] for lgu in lgu_names_raw]

        print(f"  {len(lgu_names_raw)} LGUs -> {_lgu_h3_count:,} H3 cells mapped")

        # Pre-aggregate population by (h3_index, LGU)
        print("\nPre-aggregating population by (h3_index, LGU)...")
        h3_lgu_pop_pdf = (
            population_aoi_sdf
            .select("h3_index", "population")
            .join(h3_lgu_sdf, on="h3_index", how="inner")
            .groupBy("h3_index", "LGU")
            .agg(F.sum("population").alias("pop"))
            .toPandas()
        )
        print(f"  Pre-aggregated {len(h3_lgu_pop_pdf):,} (h3, LGU) rows")

        # Per-LGU total population (denominator for % calculation)
        lgu_total_pop = h3_lgu_pop_pdf.groupby("LGU")["pop"].sum().to_dict()

        # Lookup table for potential locations
        potential_lookup = (
            potential_locations_sdf
            .select("ID", "lat", "lon", "h3_index")
            .toPandas()
            .set_index("ID")
        )

        # Compute per-step LGU accessibility
        print("\nComputing per-step LGU accessibility...")
        n_existing = len(J_existing)
        result_rows = []

        for idx, step in enumerate(pareto_results):
            if step["p"] == 0:
                continue

            p = step["p"] - 1
            covered_h3_set = set(step["covered_h3"])

            # Identify the single new facility added in this step
            prev_selected = (
                set(pareto_results[idx - 1]["selected_facilities"])
                if idx > 0
                else set(J_existing)
            )
            new_fac_ids = set(step["selected_facilities"]) - prev_selected
            new_fac_id = new_fac_ids.pop() if new_fac_ids else None

            # Retrieve location info from potential_locations lookup
            if new_fac_id and new_fac_id in potential_lookup.index:
                fac_lat = float(potential_lookup.at[new_fac_id, "lat"])
                fac_lon = float(potential_lookup.at[new_fac_id, "lon"])
                fac_h3 = str(potential_lookup.at[new_fac_id, "h3_index"])
            else:
                fac_lat = fac_lon = fac_h3 = None

            # National accessibility %
            national_access_pct = round(step["objective"] * 100.0 / total_population, 2)

            # Per-LGU accessibility (vectorised pandas)
            covered_mask = h3_lgu_pop_pdf["h3_index"].isin(covered_h3_set)
            lgu_covered = (
                h3_lgu_pop_pdf[covered_mask]
                .groupby("LGU")["pop"]
                .sum()
            )

            # Assemble result row
            row = {
                "total_facilities": n_existing + p,
                "new_facility": new_fac_id,
                "lat": fac_lat,
                "lon": fac_lon,
                "h3_index": fac_h3,
                "total_population_access_pct": national_access_pct,
            }
            for lgu_raw in lgu_names_raw:
                safe_col = name_map[lgu_raw]
                total = lgu_total_pop.get(lgu_raw, 0.0)
                covered = float(lgu_covered.get(lgu_raw, 0.0))
                row[safe_col] = round(covered * 100.0 / total, 2) if total > 0 else 0.0

            result_rows.append(row)
            print(
                f"  Step {p:3d} | +1 facility ({new_fac_id}) "
                f"-> national {national_access_pct:.2f}%"
            )

        # Check how many facilities are needed to reach TARGET for all LGUs
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
                f"  All {len(lgu_names_raw)} LGUs reach >={TARGET_ACCESS_RATE_PCT}% access at "
                f"{int(first['total_facilities'])} total facilities "
                f"({int(first['total_facilities']) - n_existing} new facilities needed)."
            )
        else:
            try:
                last = result_pdf.iloc[-1]
            except IndexError:
                print(f"Skipping {adm_level1} @ {distance_meters}")
                continue

            below = [
                lgu_raw for lgu_raw, safe in name_map.items()
                if last[safe] < TARGET_ACCESS_RATE_PCT
            ]
            print(
                f"  After {TARGET_NEW_FACILITIES} new facilities, "
                f"{len(below)} LGUs are still below {TARGET_ACCESS_RATE_PCT}%:\n"
                f"  {below}\n"
                f"  Increase TARGET_NEW_FACILITIES and re-run."
            )

        # Spatial join: assign district to each new facility
        print("\nAssigning district to each facility step...")
        boundaries_sdf = spark.table(tables["lgu"])
        boundaries_pdf = boundaries_sdf.select("LGU", "geometry_wkt").toPandas()
        boundaries_pdf["geometry"] = boundaries_pdf["geometry_wkt"].apply(
            lambda w: shapely_wkt.loads(w) if isinstance(w, str) and w.strip() else None
        )
        boundaries_pdf = boundaries_pdf.dropna(subset=["geometry"]).reset_index(drop=True)

        result_pdf["district"] = result_pdf.apply(
            lambda r: find_district(r["lat"], r["lon"], boundaries_pdf), axis=1
        )

        unmatched = result_pdf["district"].isna().sum()
        if unmatched:
            print(f"  {unmatched} row(s) could not be matched to a district.")

        # Write to UC table
        print(f"\nWriting LGU accessibility results to: {tables['lgu_accessibility']}")

        # Enforce float type for all LGU columns
        for col in lgu_col_names:
            result_pdf[col] = result_pdf[col].astype(float)
        result_pdf["lat"] = result_pdf["lat"].astype(float)
        result_pdf["lon"] = result_pdf["lon"].astype(float)
        result_pdf["total_population_access_pct"] = result_pdf["total_population_access_pct"].astype(float)

        result_sdf = spark.createDataFrame(result_pdf)
        front_cols = ["total_facilities", "new_facility", "district", "lat", "lon", "h3_index", "total_population_access_pct"]
        result_sdf = result_sdf.select(front_cols + lgu_col_names)
        result_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            tables["lgu_accessibility"]
        )

        print(
            f"Saved: {len(result_pdf)} rows x {len(result_pdf.columns)} columns\n"
            f"  Columns: total_facilities, new_facility, lat, lon, h3_index, "
            f"total_population_access_pct, [{len(lgu_col_names)} LGU columns]"
        )

        # Release cached DataFrames to free driver memory
        population_aoi_sdf.unpersist()
        selected_hosp_sdf.unpersist()
        potential_locations_sdf.unpersist()
        h3_lgu_sdf.unpersist()

    # Save metadata to base_dashboard_data table
    if not skip_dashboard:
        print("\nSaving dashboard metadata...")
        boundaries_sdf = spark.table(tables["boundaries"])
        boundary_row = boundaries_sdf.select("geometry_wkt").limit(1).collect()
        
        boundry_basetable = boundary_row[0]["geometry_wkt"] if boundary_row else None
        
        # Parse WKT string → Shapely geometry, then get centroid
        geometry = shapely_wkt.loads(boundry_basetable)
        centroid = geometry.centroid
        print(f"Centroid: x={centroid.x}, y={centroid.y}")
        boundary_row = boundaries_sdf.select("geometry_wkt").limit(1).collect()

        boundry_basetable = boundary_row[0]["geometry_wkt"] if boundary_row else None

        # Read current access from first row of results
        try:
            # Find the first row where the new_facility identifier ends with "_current"
            mask = result_pdf["new_facility"].astype(str).str.endswith("_current")
            if mask.any():
                first_row = result_pdf[mask].iloc[0]
            else:
                first_row = result_pdf.iloc[0]
        except IndexError:
            print(f"Skipped {adm_level1} @ {distance_meters}")
            continue
        current_access_pct = float(first_row["total_population_access_pct"])
        distance_km = int(distance_meters / 1000)

        first_combination = (adm_level1, distance_meters) == transform_combinations[0]

        save_dashboard_metadata(
            spark=spark,
            table_name=BASE_DASHBOARD_TABLE,
            country= COUNTRY,
            province=adm_level1,
            year=POPULATION_YEAR,
            central_lat=centroid.y,
            central_long=centroid.x,
            distance_km=distance_km,
            total_new_facilities=len(result_pdf) - 1,
            current_access=round(current_access_pct, 2),
            geometry_wkt=boundry_basetable,
            first_write=first_combination,
        )

    print(f"\n  Completed: {region_name} @ {distance_name}")

# COMMAND ----------

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)
print(f"Processed {len(transform_combinations)} combinations")
print("Next step: Run 04_visualize.py to generate charts and maps")
