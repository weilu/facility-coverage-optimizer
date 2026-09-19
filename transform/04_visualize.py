# Databricks notebook source
# MAGIC %pip install "numpy<2" geopandas shapely plotly folium pycountry

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Generate visualizations for optimization results
#
# Dependencies:
#   - transform/03_optimize.py (LGU accessibility results)
#   - transform/02_coverage.py (Coverage tables)
#
# Outputs:
#   - Pareto frontier charts (plotly)
#   - Coverage maps showing population with/without access (folium)
#
# This task is optional and can be skipped by setting ENABLE_VISUALIZATION=false
# in the databricks.yml task parameters.

# COMMAND ----------

import datetime

import plotly.graph_objects as go
import folium as fl

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# MAGIC %run "./config"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.env import get_spark, table_exists, uc_table_to_gdf
    from transform.config import (
        COUNTRY_ISO3,
        POPULATION_YEAR,
        ENABLE_VISUALIZATION,
        VIZ_SAMPLE_SIZE,
        get_transform_table_names,
        build_transform_combinations,
    )

# COMMAND ----------

# Check if visualization is enabled
if not ENABLE_VISUALIZATION:
    print("Visualization is disabled (ENABLE_VISUALIZATION=false)")
    print("Skipping all visualization tasks.")
    dbutils.notebook.exit("SKIPPED")

# COMMAND ----------

spark = get_spark()

# COMMAND ----------

# HELPER FUNCTIONS


def sample_dataframe(sdf, max_samples: int, seed: int = 1):
    """Sample a Spark DataFrame to limit visualization data size."""
    count = sdf.count()
    if count == 0:
        return sdf.toPandas()
    fraction = min(1.0, max_samples / count)
    return sdf.sample(fraction=fraction, seed=seed).toPandas()


def create_pareto_chart(
    result_pdf,
    n_existing: int,
    n_potential: int,
    current_access: float,
    max_access_possible: float,
    title: str,
):
    """Create Pareto frontier plotly chart."""
    result_sorted = result_pdf.sort_values("total_facilities")

    x_values = [n_existing] + result_sorted["total_facilities"].tolist()
    y_values = [current_access] + result_sorted["total_population_access_pct"].tolist()

    fig = go.Figure(
        data=go.Scatter(x=x_values, y=y_values, mode="lines+markers", name="Pareto Frontier")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Number of facilities (existing + new)",
        yaxis_title="Percentage of population with access",
        plot_bgcolor="white",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[0, n_potential + n_existing]),
        width=1200,
    )
    fig.add_vline(x=n_existing, line_width=3, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=n_existing,
        y=current_access,
        text="Number of<br>existing facilities",
        showarrow=True,
        arrowhead=1,
        ax=90,
        ay=-30,
    )
    fig.add_hline(y=max_access_possible, line_width=3, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=n_existing + 20,
        y=max_access_possible,
        text="Maximum access possible",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-30,
    )
    return fig


def create_coverage_map(
    boundary_gdf,
    facilities_pdf,
    pop_with_access_pdf,
    pop_without_access_pdf,
    center_lat: float,
    center_lon: float,
    title: str,
):
    """Create Folium map showing facility coverage."""
    folium_map = fl.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
    )

    # Add administrative boundary
    if not boundary_gdf.empty:
        geo_boundary = fl.GeoJson(
            data=boundary_gdf.iloc[0]["geometry"].__geo_interface__,
            style_function=lambda x: {"color": "orange", "fillOpacity": 0.1},
            name="Boundary",
        )
        geo_boundary.add_to(folium_map)

    # Add existing facilities
    for _, row in facilities_pdf.iterrows():
        fl.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.8,
        ).add_to(folium_map)

    # Add population WITHOUT access (red circles)
    for _, row in pop_without_access_pdf.iterrows():
        fl.CircleMarker(
            location=[row["ycoord"], row["xcoord"]],
            radius=3,
            color=None,
            fill=True,
            fill_color="red",
            fill_opacity=row.get("opacity", 0.5),
        ).add_to(folium_map)

    # Add population WITH access (green circles)
    for _, row in pop_with_access_pdf.iterrows():
        fl.CircleMarker(
            location=[row["ycoord"], row["xcoord"]],
            radius=3,
            color=None,
            fill=True,
            fill_color="green",
            fill_opacity=row.get("opacity", 0.5),
        ).add_to(folium_map)

    # Add title
    title_html = f'<h3 style="position:fixed;top:10px;left:50px;z-index:9999">{title}</h3>'
    folium_map.get_root().html.add_child(fl.Element(title_html))

    return folium_map

# COMMAND ----------

# DETERMINE WHICH COMBINATIONS TO VISUALIZE
#
# By default, only visualize combinations whose LGU accessibility table
# was updated after the notebook started (i.e. freshly computed).
# If nothing was updated, fall back to national-level results.

def _get_pipeline_start_time():
    """Get pipeline start time from the job parameter, or fall back to now."""
    try:
        # DAB passes {{job.start_time.iso_datetime}} (epoch_ms is not a valid ref).
        return datetime.datetime.fromisoformat(dbutils.widgets.get("PIPELINE_START_ISO"))
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc)


_job_start = _get_pipeline_start_time()

all_combinations = build_transform_combinations()


def _get_table_updated_at(table_name: str):
    """Return last-modified timestamp of a Delta table, or None."""
    try:
        rows = spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").collect()
        if rows:
            return rows[0]["timestamp"]
    except Exception:
        pass
    return None


def _select_viz_combinations(combinations, job_start):
    """Pick combinations whose results were freshly written after job_start."""
    fresh = []
    for adm, dist in combinations:
        tables = get_transform_table_names(
            COUNTRY_ISO3, adm, POPULATION_YEAR, dist
        )
        if not table_exists(tables["lgu_accessibility"]):
            continue
        ts = _get_table_updated_at(tables["lgu_accessibility"])
        if ts is not None and ts.replace(tzinfo=None) >= job_start.replace(tzinfo=None):
            fresh.append((adm, dist))

    if fresh:
        return fresh

    # Nothing freshly updated — fall back to national-level at all distances
    national = []
    for adm, dist in combinations:
        if adm is None:
            tables = get_transform_table_names(
                COUNTRY_ISO3, adm, POPULATION_YEAR, dist
            )
            if table_exists(tables["lgu_accessibility"]):
                national.append((adm, dist))

    if national:
        return national

    # Last resort: first combination that exists
    for adm, dist in combinations:
        tables = get_transform_table_names(
            COUNTRY_ISO3, adm, POPULATION_YEAR, dist
        )
        if table_exists(tables["lgu_accessibility"]):
            return [(adm, dist)]

    return []


transform_combinations = _select_viz_combinations(all_combinations, _job_start)
print(f"Will visualize {len(transform_combinations)} of {len(all_combinations)} combination(s):")
for adm, dist in transform_combinations:
    region = adm if adm else "Country"
    print(f"  - {region} @ {int(dist/1000)}km")

# COMMAND ----------

# PARETO FRONTIER CHARTS

for adm_level1, distance_meters in transform_combinations:
    region_name = adm_level1 if adm_level1 else "Country"
    distance_name = f"{int(distance_meters / 1000)}km"

    tables = get_transform_table_names(
        COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )

    # Check if results exist
    if not table_exists(tables["lgu_accessibility"]):
        print(f"Skipping {region_name} @ {distance_name}: results table not found")
        continue

    print(f"\nGenerating Pareto frontier: {region_name} @ {distance_name}")

    # Load required data
    result_sdf = spark.table(tables["lgu_accessibility"])
    population_aoi_sdf = spark.table(tables["population_aoi"])
    selected_hosp_sdf = spark.table(tables["facilities_h3"])
    potential_locations_sdf = spark.table(tables["potential_locations"])
    hosp_coverage_sdf = spark.table(tables["facilities_coverage"])
    potential_coverage_sdf = spark.table(tables["potential_coverage"])

    total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]
    n_existing = selected_hosp_sdf.count()
    n_potential = potential_locations_sdf.count()

    # Calculate current access (existing facilities only)
    existing_covered_ids_sdf = hosp_coverage_sdf.select("pop_ID").distinct()
    covered_pop_val = (
        population_aoi_sdf.join(
            existing_covered_ids_sdf,
            population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
            "inner",
        )
        .agg(F.sum("population"))
        .collect()[0][0]
    )
    current_access = round(covered_pop_val * 100 / total_population, 2)

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

    # Build plot data from results table
    result_pdf = result_sdf.select("total_facilities", "total_population_access_pct").toPandas()

    # Create and display Pareto chart
    fig = create_pareto_chart(
        result_pdf,
        n_existing,
        n_potential,
        current_access,
        max_access_possible,
        f"Pareto Frontier: {region_name} @ {distance_name}",
    )
    fig.show()

# COMMAND ----------

# COVERAGE MAPS (Folium)

for adm_level1, distance_meters in transform_combinations:
    region_name = adm_level1 if adm_level1 else "Country"
    distance_name = f"{int(distance_meters / 1000)}km"

    tables = get_transform_table_names(
        COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )

    # Check if required tables exist
    if not table_exists(tables["lgu_accessibility"]):
        print(f"Skipping {region_name} @ {distance_name}: results table not found")
        continue

    print(f"\nGenerating coverage map: {region_name} @ {distance_name}")

    # Load data
    population_aoi_sdf = spark.table(tables["population_aoi"])
    selected_hosp_sdf = spark.table(tables["facilities_h3"])
    hosp_coverage_sdf = spark.table(tables["facilities_coverage"])

    # Load boundary
    boundary_gdf = uc_table_to_gdf(tables["boundaries"])
    centroid = boundary_gdf.iloc[0]["geometry"].centroid
    center_lat = centroid.y
    center_lon = centroid.x

    # Identify covered vs uncovered population
    existing_covered_ids_sdf = hosp_coverage_sdf.select("pop_ID").distinct()

    pop_with_access_sdf = population_aoi_sdf.join(
        existing_covered_ids_sdf,
        population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
        "inner",
    ).select("xcoord", "ycoord", "opacity")

    pop_without_access_sdf = population_aoi_sdf.join(
        existing_covered_ids_sdf,
        population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
        "left_anti",
    ).select("xcoord", "ycoord", "opacity")

    # Scale sample size down for country-level (many more facilities + points)
    n_facilities = selected_hosp_sdf.count()
    viz_samples = min(VIZ_SAMPLE_SIZE, max(1000, 10_000 - n_facilities * 3))

    pop_with_access_pdf = sample_dataframe(pop_with_access_sdf, viz_samples, seed=1)
    pop_without_access_pdf = sample_dataframe(pop_without_access_sdf, viz_samples, seed=2)
    facilities_pdf = selected_hosp_sdf.select("ID", "lat", "lon").toPandas()

    print(f"  Sampled {len(pop_with_access_pdf)} covered, {len(pop_without_access_pdf)} uncovered points")

    # Create and display map
    folium_map = create_coverage_map(
        boundary_gdf,
        facilities_pdf,
        pop_with_access_pdf,
        pop_without_access_pdf,
        center_lat,
        center_lon,
        f"Coverage: {region_name} @ {distance_name}",
    )
    display(folium_map)

# COMMAND ----------

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print(f"Generated visualizations for {len(transform_combinations)} of {len(all_combinations)} combinations")
