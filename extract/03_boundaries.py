# Databricks notebook source
# MAGIC %pip install "numpy<2" geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Extract province and LGU boundaries
#
# Dependencies: 01b_download_wb.py (WB boundaries cached in Volume)
# Outputs:
#   - Province boundary UC tables (one per province or country-level)
#   - Country-level LGU boundaries UC table
#
# This task extracts boundaries for each configured province from the
# cached World Bank boundaries. It also extracts all LGU (district)
# boundaries for the country, which are used for equity metrics.

# COMMAND ----------

import geopandas as gpd

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# MAGIC %run "./config"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.env import (
        get_spark,
        gdf_to_uc_table,
        uc_table_to_gdf,
        table_exists,
    )
    from extract.config import (
        COUNTRY,
        ISO_3,
        FORCE_RECOMPUTE,
        INCLUDE_ADM_LEVEL0,
        ADM_LEVEL1_LIST,
        get_table_names,
        COUNTRY_LGU_TABLE,
        POPULATION_YEAR,
        load_cached_wb_boundaries,
        get_all_adm_level1_names,
    )

# COMMAND ----------

spark = get_spark()

# COMMAND ----------


def extract_boundaries(
    country_iso3: str,
    adm_level1: str | None,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Extracts World Bank Official Boundaries and saves to UC table.
    Returns the selected boundary GeoDataFrame.

    World Bank attributes:
    - ISO_A3: ISO 3-letter country code
    - NAM_0: Country name
    - NAM_1: Admin level 1 name (province/state)
    """
    if not force and table_exists(table_name):
        print(f"WB boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    if adm_level1 is not None:
        gdf = load_cached_wb_boundaries(admin_level=1)
        gdf_country = gdf[gdf["ISO_A3"] == country_iso3]
        selected = gdf_country[gdf_country["NAM_1"] == adm_level1]
        if selected.empty:
            available = sorted(gdf_country["NAM_1"].unique().tolist())
            raise ValueError(f"Admin1 '{adm_level1}' not found. Available: {available}")
    else:
        gdf = load_cached_wb_boundaries(admin_level=0)
        selected = gdf[gdf["ISO_A3"] == country_iso3]
        if selected.empty:
            raise ValueError(f"Country ISO3 '{country_iso3}' not found in WB boundaries")

    print(f"Selected {len(selected)} boundary feature(s)")
    gdf_to_uc_table(selected, table_name)
    return selected


def extract_boundaries_lgu(
    country_iso3: str,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Extracts World Bank Admin2 boundaries for LGU analysis.
    Normalizes to schema: province + LGU + geometry.
    """
    if not force and table_exists(table_name):
        print(f"WB LGU boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    gdf = load_cached_wb_boundaries(admin_level=2)
    gdf_country = gdf[gdf["ISO_A3"] == country_iso3]

    if gdf_country.empty:
        raise ValueError(f"No Admin2 boundaries found for ISO3 '{country_iso3}'")

    lgu_gdf = (
        gdf_country[["NAM_1", "NAM_2", "geometry"]]
        .copy()
        .rename(columns={"NAM_1": "province", "NAM_2": "LGU"})
        .reset_index(drop=True)
    )

    # Ensure WGS-84
    if lgu_gdf.crs is None or lgu_gdf.crs.to_epsg() != 4326:
        lgu_gdf = lgu_gdf.to_crs(epsg=4326)

    print(f"Extracted {len(lgu_gdf)} LGU boundaries (WB) | Uploading to: {table_name}")
    gdf_to_uc_table(lgu_gdf, table_name)
    return lgu_gdf

# COMMAND ----------

# EXECUTE TASK: Determine regions to process

print(f"Country: {COUNTRY} | ISO-3: {ISO_3}")

regions_to_process = []

# ADM0 (country-level)
if INCLUDE_ADM_LEVEL0:
    regions_to_process.append(None)

# ADM1 (provinces)
if ADM_LEVEL1_LIST == []:
    regions_to_process.extend(get_all_adm_level1_names(ISO_3))
else:
    regions_to_process.extend(ADM_LEVEL1_LIST)

print(f"Will process {len(regions_to_process)} region(s): {regions_to_process}")

# COMMAND ----------

# EXECUTE TASK: Extract province boundaries

extraction_results = []

for adm_level1 in regions_to_process:
    print("\n" + "=" * 60)
    print(f"PROCESSING: {adm_level1 if adm_level1 else 'ENTIRE COUNTRY'}")
    print("=" * 60)

    tables = get_table_names(COUNTRY, ISO_3, adm_level1, POPULATION_YEAR)
    boundaries_table = tables["boundaries"]

    print(f"  Boundaries Table: {boundaries_table}")

    selected_boundary_gdf = extract_boundaries(
        country_iso3=ISO_3,
        adm_level1=adm_level1,
        table_name=boundaries_table,
        force=FORCE_RECOMPUTE,
    )

    extraction_results.append({
        "adm_level1": adm_level1,
        "boundaries_table": boundaries_table,
    })

    print(f"  Completed: {adm_level1 if adm_level1 else 'ENTIRE COUNTRY'}")

# COMMAND ----------

# EXECUTE TASK: Extract LGU boundaries (country-level)

lgu_gdf = extract_boundaries_lgu(
    country_iso3=ISO_3,  # e.g. "ZMB" (resolved above)
    table_name=COUNTRY_LGU_TABLE,
    force=FORCE_RECOMPUTE,
)
print(f"LGU count: {len(lgu_gdf)}")
print(lgu_gdf[["LGU"]].to_string(max_rows=10))

# COMMAND ----------

# TASK SUMMARY

print("\n" + "=" * 60)
print("BOUNDARIES EXTRACTION COMPLETE")
print("=" * 60)
print(f"Country: {COUNTRY} ({ISO_3})")
print(f"LGU boundaries: {COUNTRY_LGU_TABLE} ({len(lgu_gdf)} LGUs)")
print(f"Province boundaries processed: {len(extraction_results)}")
for result in extraction_results:
    region = result["adm_level1"] if result["adm_level1"] else "Country"
    print(f"  - {region}: {result['boundaries_table']}")
print("=" * 60)
