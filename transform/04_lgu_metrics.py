# Databricks notebook source
# MAGIC %pip install geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Compute LGU accessibility metrics
#
# Dependencies:
#   - transform/03_optimize.py (Pareto results)
#   - extract/03_boundaries.py (LGU boundaries)
#
# Outputs:
#   - LGU accessibility results table (per-step metrics for each LGU)
#
# This task computes per-district accessibility metrics at each step
# of the greedy MCLP optimization, enabling equity analysis.

# COMMAND ----------

import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt as shapely_wkt

from pyspark.sql import functions as F

# COMMAND ----------

# Import shared utilities and configuration
from shared.utils import get_spark, uc_table_to_gdf, table_exists
from transform.config import (
    COUNTRY,
    COUNTRY_ISO3,
    POPULATION_YEAR,
    FORCE_RECOMPUTE,
    H3_RESOLUTION,
    TARGET_NEW_FACILITIES,
    TARGET_ACCESS_RATE_PCT,
    get_k_rings,
    get_transform_table_names,
    build_transform_combinations,
)

spark = get_spark()

# COMMAND ----------

# HELPER FUNCTIONS

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


def find_district(lat, lon, boundaries_pdf):
    """Return the LGU name whose polygon contains (lon, lat), else None."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    pt = Point(lon, lat)  # Shapely Point is (x=lon, y=lat)
    for _, row in boundaries_pdf.iterrows():
        if row["geometry"].contains(pt):
            return row["LGU"]
    return None  # point falls outside all boundaries

# COMMAND ----------

# GREEDY MCLP FUNCTION (needed to regenerate results if not passed from previous task)

def solve_mclp_greedy(w, IJ, J_existing, J_potential, max_new_facilities):
    """
    Greedy Maximum Covering Location Problem.
    """
    facility_covers = {}
    for h3_cell, fac_list in IJ.items():
        for fac in fac_list:
            if fac not in facility_covers:
                facility_covers[fac] = set()
            facility_covers[fac].add(h3_cell)

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

    return results

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
    print(f"COMPUTING LGU METRICS: {region_name} @ {distance_name}")
    print("=" * 60)

    tables = get_transform_table_names(
        COUNTRY, COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )

    # Check if already computed
    if not FORCE_RECOMPUTE and table_exists(tables["lgu_accessibility"]):
        print(f"LGU accessibility table already exists: {tables['lgu_accessibility']}")
        print("Set FORCE_RECOMPUTE = True to regenerate.")
        continue

    # Load data
    print(f"\nLoading data...")
    population_aoi_sdf = spark.table(tables["population_aoi"]).cache()
    selected_hosp_sdf = spark.table(tables["facilities_h3"]).cache()
    potential_locations_sdf = spark.table(tables["potential_locations"]).cache()
    hosp_coverage_sdf = spark.table(tables["facilities_coverage"])
    potential_coverage_sdf = spark.table(tables["potential_coverage"])

    total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]

    # Regenerate optimization inputs
    print("\nRebuilding optimization inputs...")
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
    print("\nBuilding H3 → LGU mapping via Photon polyfill...")
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
    name_map_r = {v: k for k, v in name_map.items()}
    lgu_col_names = [name_map[lgu] for lgu in lgu_names_raw]

    print(f"  {len(lgu_names_raw)} LGUs → {_lgu_h3_count:,} H3 cells mapped")

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
        p = step["p"]
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
            f"→ national {national_access_pct:.2f}%"
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
            f"✅  All {len(lgu_names_raw)} LGUs reach ≥{TARGET_ACCESS_RATE_PCT}% access at "
            f"{int(first['total_facilities'])} total facilities "
            f"({int(first['total_facilities']) - n_existing} new facilities needed)."
        )
    else:
        last = result_pdf.iloc[-1]
        below = [
            lgu_raw for lgu_raw, safe in name_map.items()
            if last[safe] < TARGET_ACCESS_RATE_PCT
        ]
        print(
            f"⚠️  After {TARGET_NEW_FACILITIES} new facilities, "
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
        print(f"  ⚠️  {unmatched} row(s) could not be matched to a district.")

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
        f"Saved: {len(result_pdf)} rows × {len(result_pdf.columns)} columns\n"
        f"  Columns: total_facilities, new_facility, lat, lon, h3_index, "
        f"total_population_access_pct, [{len(lgu_col_names)} LGU columns]"
    )

    print(f"\n  Completed: {region_name} @ {distance_name}")

# COMMAND ----------

print("\n" + "=" * 60)
print("LGU METRICS COMPUTATION COMPLETE")
print("=" * 60)
print(f"Processed {len(transform_combinations)} combinations")
print("Transform pipeline complete!")
