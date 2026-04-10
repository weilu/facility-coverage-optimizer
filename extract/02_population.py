# Databricks notebook source
# MAGIC %pip install shapely rasterio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Convert WorldPop raster to Unity Catalog table
#
# Dependencies: 01a_download_worldpop.py (WorldPop raster file)
# Outputs: Country-level population UC table with H3 indices
#
# This task reads the WorldPop raster in chunks and saves all populated
# pixels to a Unity Catalog table. Each pixel gets an H3 index for fast
# spatial filtering in the transform phase.
#
# Long-running task: ~7 minutes for Zambia

# COMMAND ----------

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType

# COMMAND ----------

# Import shared utilities and configuration
from shared.env import get_spark, table_exists
from extract.config import (
    COUNTRY,
    ISO_3,
    FORCE_RECOMPUTE,
    RASTER_PATH,
    COUNTRY_POPULATION_TABLE,
)

spark = get_spark()

# COMMAND ----------

def extract_population_chunked(
    raster_path: str,
    table_name: str,
    chunk_size: int = 1024,
    h3_resolution: int = 8,
    force: bool = False,
) -> int:
    """
    Reads WorldPop raster in chunks and saves populated pixels to UC table.
    Uses windowed reading to avoid loading entire raster into memory.
    Adds H3 index for fast spatial filtering.
    Returns total number of populated pixels.
    """
    if not force and table_exists(table_name):
        count = spark.table(table_name).count()
        print(f"Population table already exists: {table_name} ({count:,} rows)")
        return count

    print(f"Processing raster in chunks: {raster_path}")
    print(f"  H3 resolution: {h3_resolution}")

    schema = StructType([
        StructField("xcoord", DoubleType(), False),
        StructField("ycoord", DoubleType(), False),
        StructField("population", DoubleType(), False),
    ])

    total_pixels = 0
    first_chunk = True

    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        transform_affine = src.transform

        n_row_chunks = (height + chunk_size - 1) // chunk_size
        n_col_chunks = (width + chunk_size - 1) // chunk_size
        total_chunks = n_row_chunks * n_col_chunks

        print(f"  Raster size: {width} x {height}")
        print(f"  Chunk size: {chunk_size} x {chunk_size}")
        print(f"  Total chunks: {total_chunks}")

        chunk_num = 0
        for row_off in range(0, height, chunk_size):
            for col_off in range(0, width, chunk_size):
                chunk_num += 1

                win_height = min(chunk_size, height - row_off)
                win_width = min(chunk_size, width - col_off)
                window = Window(col_off, row_off, win_width, win_height)

                data = src.read(1, window=window)

                rows, cols = np.where(data > 0)
                if len(rows) == 0:
                    continue

                values = data[rows, cols].astype(float)

                abs_rows = rows + row_off
                abs_cols = cols + col_off
                x_coords, y_coords = rasterio.transform.xy(
                    transform_affine, abs_rows, abs_cols, offset="center"
                )

                pdf = pd.DataFrame({
                    "xcoord": np.array(x_coords, dtype=float),
                    "ycoord": np.array(y_coords, dtype=float),
                    "population": values,
                })

                sdf = spark.createDataFrame(pdf, schema=schema)

                # Add H3 index (Photon-accelerated)
                sdf = sdf.withColumn(
                    "h3_index",
                    F.expr(f"h3_longlatash3(xcoord, ycoord, {h3_resolution})")
                )

                if first_chunk:
                    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
                    first_chunk = False
                else:
                    sdf.write.mode("append").saveAsTable(table_name)

                total_pixels += len(rows)

                if chunk_num % 50 == 0 or chunk_num == total_chunks:
                    print(f"  Processed chunk {chunk_num}/{total_chunks}, pixels so far: {total_pixels:,}")

    print(f"Population table saved: {table_name} ({total_pixels:,} rows)")
    return total_pixels

# COMMAND ----------

# EXECUTE TASK

print(f"Country: {COUNTRY} | ISO-3: {ISO_3}")
print(f"Raster path: {RASTER_PATH}")
print(f"Output table: {COUNTRY_POPULATION_TABLE}")

total_pixels = extract_population_chunked(
    raster_path=RASTER_PATH,
    table_name=COUNTRY_POPULATION_TABLE,
    chunk_size=1024,
    force=FORCE_RECOMPUTE,
)

print(f"\nTask complete. Population table: {COUNTRY_POPULATION_TABLE}")
print(f"Total populated pixels: {total_pixels:,}")
