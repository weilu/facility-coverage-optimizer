# Databricks notebook source
# MAGIC %run "../facility-coverage-optimizer/extract/01a_download_worldpop"

# COMMAND ----------

# MAGIC %run "../facility-coverage-optimizer/extract/01b_download_wb"

# COMMAND ----------

# MAGIC %run "../facility-coverage-optimizer/extract/02_population"

# COMMAND ----------

# MAGIC %run "../facility-coverage-optimizer/extract/03_boundaries"

# COMMAND ----------

# MAGIC %run "../facility-coverage-optimizer/extract/04_facilities"

# COMMAND ----------

# DBTITLE 1,Prepare for Transform
# MAGIC %run "../facility-coverage-optimizer/transform/01_prepare"

# COMMAND ----------

# MAGIC %run "../facility-coverage-optimizer/transform/02_coverage"

# COMMAND ----------

# MAGIC %run "../facility-coverage-optimizer/transform/03_optimize"

# COMMAND ----------

print("=== DONE ===")

# COMMAND ----------

# # Get list of tables starting with "potential_locations_" or "potential_coverage_"
# tables_to_drop = spark.sql("""
#   SELECT table_name 
#   FROM prd_mega.information_schema.tables
#   WHERE table_catalog = 'prd_mega' AND table_schema = 'sgpbpi163'
#     AND (table_name LIKE 'potential_locations_%' OR table_name LIKE 'potential_coverage_%')
# """).collect()

# # Drop each table
# for row in tables_to_drop:
#     table_name = row.table_name
#     print(f"Dropping table: {table_name}")
#     spark.sql(f"DROP TABLE IF EXISTS prd_mega.sgpbpi163.{table_name}")

# print(f"\nDropped {len(tables_to_drop)} table(s).")
# print(len(tables_to_drop))
