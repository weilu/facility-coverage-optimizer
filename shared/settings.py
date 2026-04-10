# Databricks notebook source
# Shared application settings
# Imported by extract/config.py and transform/config.py

import os

# Unity Catalog
UC_CATALOG = "prd_mega"
UC_SCHEMA = os.getenv("UC_SCHEMA", "sgpbpi163")

# Country settings
COUNTRY = "Zambia"
ISO_2 = "ZM"
ISO_3 = "ZMB"
POPULATION_YEAR = 2025
