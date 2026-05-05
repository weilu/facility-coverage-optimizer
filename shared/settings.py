# Databricks notebook source
# Shared application settings
# Imported by extract/config.py and transform/config.py

# Unity Catalog
UC_CATALOG = "prd_mega"
UC_SCHEMA_DEFAULT = "sgpbpi163"

def _get_uc_schema() -> str:
    """Get UC_SCHEMA from dbutils widget (Databricks) or use default (local)."""
    try:
        return dbutils.widgets.get("UC_SCHEMA")
    except:
        return UC_SCHEMA_DEFAULT

UC_SCHEMA = _get_uc_schema()

# Country settings
COUNTRY = "Serbia"
ISO_2 = "RS"
ISO_3 = "SRB"
POPULATION_YEAR = 2025
