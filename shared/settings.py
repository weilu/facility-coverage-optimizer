# Databricks notebook source
# Shared application settings
# Imported by extract/config.py and transform/config.py

import pycountry

# Unity Catalog
UC_CATALOG = "prd_mega"
UC_SCHEMA_DEFAULT = "sgpbpi163"


def _get_widget(name: str, default: str) -> str:
    """Read a Databricks widget, falling back to a default off-cluster."""
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


def resolve_iso2(iso3: str) -> str:
    """ISO 3166-1 alpha-2 for an alpha-3 code (standard codes, unlike WB admin0)."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    return country.alpha_2


def resolve_country_name(iso3: str) -> str:
    """Human-readable country name (display/logging only; table names use ISO3)."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    return getattr(country, "common_name", None) or country.name


UC_SCHEMA = _get_widget("UC_SCHEMA", UC_SCHEMA_DEFAULT)

# Country settings — derived from a single COUNTRY_ISO3 widget (default Laos)
ISO_3 = _get_widget("COUNTRY_ISO3", "LAO").upper()
ISO_2 = resolve_iso2(ISO_3)
COUNTRY = resolve_country_name(ISO_3)
POPULATION_YEAR = int(_get_widget("POPULATION_YEAR", "2025"))
