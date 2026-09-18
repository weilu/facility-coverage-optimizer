# Databricks notebook source
# Shared application settings
# Imported by extract/config.py and transform/config.py

import pycountry

# Unity Catalog
UC_CATALOG = "prd_mega"
UC_SCHEMA_DEFAULT = "sgpbpi163"


def _get_widget(name: str, default: str) -> str:
    # dbutils is absent off-cluster (local/CI/wheel) — check globals() rather than
    # bare-reference it, so importing this module there falls back to the default.
    if "dbutils" not in globals():
        return default
    return dbutils.widgets.get(name)


def resolve_iso2(iso3: str) -> str:
    """ISO 3166-1 alpha-2 for an alpha-3 code (standard codes, unlike WB admin0)."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    return country.alpha_2


def resolve_country_name(iso3: str) -> str:
    # Display/logging only. base_dashboard_data.country uses the WB NAM_0 (set in
    # 03_optimize) since pycountry diverges (e.g. "Yemen" vs "Republic of Yemen").
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    return getattr(country, "common_name", None) or country.name


def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("true", "1", "yes")


def _get_bool_widget(name: str, default: bool) -> bool:
    if "dbutils" not in globals():  # not on Databricks (local/CI or imported wheel)
        return default
    return _parse_bool(dbutils.widgets.get(name))


UC_SCHEMA = _get_widget("UC_SCHEMA", UC_SCHEMA_DEFAULT)

# "schema/volume" for file caches; its own widget since the volume name isn't
# derivable from UC_SCHEMA.
UC_VOLUME = _get_widget("UC_VOLUME", "sgpbpi163/vgpbpi163")

# Country settings — derived from a single COUNTRY_ISO3 widget (default Laos)
ISO_3 = _get_widget("COUNTRY_ISO3", "LAO").upper()
ISO_2 = resolve_iso2(ISO_3)
COUNTRY = resolve_country_name(ISO_3)
POPULATION_YEAR = int(_get_widget("POPULATION_YEAR", "2025"))

# Run-control (job parameters), consolidated from extract/config.py + transform/config.py
FORCE_RECOMPUTE = _get_bool_widget("FORCE_RECOMPUTE", False)
INCLUDE_ADM_LEVEL0 = _get_bool_widget("INCLUDE_ADM_LEVEL0", True)

# H3 resolution — single source; extraction and transform must match.
H3_RESOLUTION = 8
