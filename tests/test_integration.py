"""
Integration tests using local PySpark.

These tests verify Spark-based transformations work correctly with a local
Spark session. They test the core logic without requiring Databricks.

Requirements:
- pyspark
- geopandas
- shapely
"""

import tempfile
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

# Skip all tests if pyspark is not installed
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType


@pytest.fixture(scope="module")
def spark():
    """Create a local Spark session for testing."""
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("pimpam-integration-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPopulationDataProcessing:
    """Integration tests for population data processing."""

    def test_create_population_dataframe(self, spark):
        """Test creating a population DataFrame with coordinates."""
        # Simulate population data from raster
        data = [
            (28.0, -15.0, 100.0),
            (28.1, -15.1, 200.0),
            (28.2, -15.2, 150.0),
            (28.3, -15.3, 50.0),
        ]
        schema = StructType([
            StructField("xcoord", DoubleType(), False),
            StructField("ycoord", DoubleType(), False),
            StructField("population", DoubleType(), False),
        ])

        sdf = spark.createDataFrame(data, schema)

        assert sdf.count() == 4
        total_pop = sdf.agg(F.sum("population")).collect()[0][0]
        assert total_pop == 500.0

    def test_population_aggregation(self, spark):
        """Test aggregating population by region."""
        data = [
            ("region_a", 100.0),
            ("region_a", 200.0),
            ("region_b", 150.0),
            ("region_b", 50.0),
            ("region_c", 75.0),
        ]
        schema = StructType([
            StructField("region", StringType(), False),
            StructField("population", DoubleType(), False),
        ])

        sdf = spark.createDataFrame(data, schema)

        aggregated = sdf.groupBy("region").agg(
            F.sum("population").alias("total_pop"),
            F.count("*").alias("count"),
        ).orderBy("region")

        result = aggregated.collect()
        assert len(result) == 3
        assert result[0]["region"] == "region_a"
        assert result[0]["total_pop"] == 300.0
        assert result[1]["total_pop"] == 200.0
        assert result[2]["total_pop"] == 75.0

    def test_population_percentile_calculation(self, spark):
        """Test calculating population percentiles for visualization."""
        # Create population data with varying densities
        data = [(float(i), float(i * 10)) for i in range(1, 101)]
        schema = StructType([
            StructField("id", DoubleType(), False),
            StructField("population", DoubleType(), False),
        ])

        sdf = spark.createDataFrame(data, schema)

        # Calculate quartiles
        thresholds = sdf.approxQuantile("population", [0.25, 0.5, 0.75], 0.01)

        # Should be approximately [250, 500, 750]
        assert len(thresholds) == 3
        assert thresholds[0] < thresholds[1] < thresholds[2]
        assert 200 < thresholds[0] < 300  # 25th percentile
        assert 450 < thresholds[1] < 550  # 50th percentile
        assert 700 < thresholds[2] < 800  # 75th percentile


class TestFacilitiesProcessing:
    """Integration tests for facility data processing."""

    def test_facilities_to_spark(self, spark):
        """Test converting GeoDataFrame facilities to Spark DataFrame."""
        # Create sample facilities GeoDataFrame
        facilities_gdf = gpd.GeoDataFrame({
            "osm_id": [1, 2, 3],
            "name": ["Hospital A", "Clinic B", "Hospital C"],
            "geometry": [
                Point(28.0, -15.0),
                Point(28.1, -15.1),
                Point(28.2, -15.2),
            ],
        }, crs="EPSG:4326")

        # Convert to pandas with WKT
        facilities_pdf = facilities_gdf.copy()
        facilities_pdf["lon"] = facilities_pdf.geometry.x
        facilities_pdf["lat"] = facilities_pdf.geometry.y
        facilities_pdf["geom_wkt"] = facilities_pdf.geometry.apply(lambda g: g.wkt)
        facilities_pdf = facilities_pdf.drop(columns=["geometry"])

        # Convert to Spark
        sdf = spark.createDataFrame(facilities_pdf)

        assert sdf.count() == 3
        assert "lon" in sdf.columns
        assert "lat" in sdf.columns
        assert "geom_wkt" in sdf.columns

    def test_add_facility_ids(self, spark):
        """Test adding monotonic IDs to facilities."""
        data = [
            ("Hospital A", 28.0, -15.0),
            ("Clinic B", 28.1, -15.1),
        ]
        schema = StructType([
            StructField("name", StringType(), False),
            StructField("lon", DoubleType(), False),
            StructField("lat", DoubleType(), False),
        ])

        sdf = spark.createDataFrame(data, schema)

        sdf_with_id = (
            sdf.withColumn("row_id", F.monotonically_increasing_id())
            .withColumn("ID", F.concat(F.col("row_id").cast(StringType()), F.lit("_current")))
        )

        result = sdf_with_id.select("ID").collect()
        assert len(result) == 2
        assert result[0]["ID"].endswith("_current")
        assert result[1]["ID"].endswith("_current")


class TestCoverageComputation:
    """Integration tests for coverage computation logic."""

    def test_coverage_join(self, spark):
        """Test coverage computation using joins."""
        # Create facilities with their covered regions
        facilities = [
            ("fac_1", ["h3_a", "h3_b"]),
            ("fac_2", ["h3_b", "h3_c"]),
        ]
        fac_schema = StructType([
            StructField("facility_ID", StringType(), False),
            StructField("covered_h3", StringType(), False),
        ])

        # Explode to get facility -> h3 pairs
        fac_data = []
        for fac_id, h3_list in facilities:
            for h3 in h3_list:
                fac_data.append((fac_id, h3))

        fac_sdf = spark.createDataFrame(fac_data, fac_schema)

        # Create population by h3
        pop_data = [
            ("h3_a", 100.0),
            ("h3_b", 200.0),
            ("h3_c", 150.0),
            ("h3_d", 50.0),  # Not covered by any facility
        ]
        pop_schema = StructType([
            StructField("h3_index", StringType(), False),
            StructField("population", DoubleType(), False),
        ])
        pop_sdf = spark.createDataFrame(pop_data, pop_schema)

        # Join to compute coverage
        coverage_sdf = fac_sdf.join(
            pop_sdf,
            fac_sdf["covered_h3"] == pop_sdf["h3_index"],
            "inner"
        )

        # Aggregate coverage per facility
        facility_coverage = (
            coverage_sdf
            .groupBy("facility_ID")
            .agg(F.sum("population").alias("pop_covered"))
            .orderBy("facility_ID")
        )

        result = facility_coverage.collect()
        assert len(result) == 2
        assert result[0]["facility_ID"] == "fac_1"
        assert result[0]["pop_covered"] == 300.0  # h3_a (100) + h3_b (200)
        assert result[1]["facility_ID"] == "fac_2"
        assert result[1]["pop_covered"] == 350.0  # h3_b (200) + h3_c (150)

    def test_uncovered_population(self, spark):
        """Test identifying uncovered population."""
        # Covered h3 cells
        covered_h3 = ["h3_a", "h3_b"]
        covered_sdf = spark.createDataFrame(
            [(h3,) for h3 in covered_h3],
            ["h3_covered"]
        )

        # All population
        pop_data = [
            ("h3_a", 100.0),
            ("h3_b", 200.0),
            ("h3_c", 150.0),
            ("h3_d", 50.0),
        ]
        pop_sdf = spark.createDataFrame(pop_data, ["h3_index", "population"])

        # Find uncovered using left_anti join
        uncovered_sdf = pop_sdf.join(
            covered_sdf,
            pop_sdf["h3_index"] == covered_sdf["h3_covered"],
            "left_anti"
        )

        uncovered_pop = uncovered_sdf.agg(F.sum("population")).collect()[0][0]
        assert uncovered_pop == 200.0  # h3_c (150) + h3_d (50)


class TestGridGeneration:
    """Integration tests for potential location generation."""

    def test_grid_points_generation(self):
        """Test generating grid points within a polygon."""
        # Create a simple square polygon
        polygon = Polygon([
            (27.9, -15.1),
            (28.1, -15.1),
            (28.1, -14.9),
            (27.9, -14.9),
            (27.9, -15.1),
        ])

        # Generate grid
        spacing = 0.05
        minx, miny, maxx, maxy = polygon.bounds
        x_coords = np.arange(np.floor(minx * 100) / 100, np.ceil(maxx * 100) / 100, spacing)
        y_coords = np.arange(np.floor(miny * 100) / 100, np.ceil(maxy * 100) / 100, spacing)
        mesh = np.meshgrid(x_coords, y_coords)
        pdf = pd.DataFrame({"longitude": mesh[0].flatten(), "latitude": mesh[1].flatten()})

        # Create GeoDataFrame and clip to polygon
        gdf = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf.longitude, pdf.latitude),
            crs="EPSG:4326"
        )
        gdf_clipped = gpd.clip(gdf, polygon)

        # Should have multiple points within the polygon
        assert len(gdf_clipped) > 0
        # Use intersects instead of contains to handle boundary points
        assert all(polygon.intersects(p) for p in gdf_clipped.geometry)


class TestOptimizationPreparation:
    """Integration tests for MCLP optimization input preparation."""

    def test_aggregate_coverage_by_facility(self, spark):
        """Test aggregating coverage sets by facility."""
        # Coverage pairs (facility_ID, h3_index)
        coverage_data = [
            ("fac_1", "h3_a"),
            ("fac_1", "h3_b"),
            ("fac_2", "h3_b"),
            ("fac_2", "h3_c"),
            ("fac_3", "h3_a"),
        ]
        coverage_sdf = spark.createDataFrame(coverage_data, ["facility_ID", "h3_index"])

        # Aggregate by h3_index to get list of facilities covering each cell
        h3_to_fac = (
            coverage_sdf
            .groupBy("h3_index")
            .agg(F.collect_set("facility_ID").alias("fac_ids"))
        )

        result = {row["h3_index"]: row["fac_ids"] for row in h3_to_fac.collect()}

        assert set(result["h3_a"]) == {"fac_1", "fac_3"}
        assert set(result["h3_b"]) == {"fac_1", "fac_2"}
        assert set(result["h3_c"]) == {"fac_2"}


class TestDataFrameConversions:
    """Integration tests for DataFrame conversions."""

    def test_pandas_to_spark_roundtrip(self, spark):
        """Test converting pandas DataFrame to Spark and back."""
        pdf = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [1.1, 2.2, 3.3],
        })

        sdf = spark.createDataFrame(pdf)
        pdf_back = sdf.toPandas()

        pd.testing.assert_frame_equal(
            pdf.sort_values("id").reset_index(drop=True),
            pdf_back.sort_values("id").reset_index(drop=True),
        )

    def test_geodataframe_wkt_roundtrip(self, spark):
        """Test converting GeoDataFrame with WKT geometry."""
        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "name": ["point1", "point2"],
            "geometry": [Point(28.0, -15.0), Point(28.1, -15.1)],
        }, crs="EPSG:4326")

        # Convert to pandas with WKT
        pdf = pd.DataFrame(gdf.drop(columns=["geometry"]))
        pdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt)

        # Convert to Spark and back
        sdf = spark.createDataFrame(pdf)
        pdf_back = sdf.toPandas()

        # Verify WKT strings are preserved
        assert pdf_back["geometry_wkt"].iloc[0] == "POINT (28 -15)"
        assert pdf_back["geometry_wkt"].iloc[1] == "POINT (28.1 -15.1)"
