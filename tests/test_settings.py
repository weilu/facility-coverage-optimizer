"""Tests for country/ISO derivation and widget fallbacks in shared/settings.py."""

import pytest

from shared.settings import resolve_iso2, resolve_country_name

# The 36 countries already present in prd_mega.sgpbpi163.
ALL_ISO3 = [
    "AFG", "BEN", "BFA", "BGD", "CIV", "CMR", "DJI", "ETH", "GAB", "GIN",
    "GMB", "GNQ", "HTI", "IND", "JPN", "KHM", "LAO", "MLI", "MWI", "NER",
    "NGA", "NPL", "PAK", "PSE", "ROU", "SDN", "SEN", "SOM", "SRB", "SSD",
    "SYR", "TCD", "TGO", "UZB", "YEM", "ZMB",
]


class TestResolveIso2:
    def test_all_36_resolve(self):
        for iso3 in ALL_ISO3:
            assert len(resolve_iso2(iso3)) == 2

    @pytest.mark.parametrize("iso3,iso2", [("PSE", "PS"), ("SRB", "RS"), ("YEM", "YE")])
    def test_standard_codes_for_wb_oddities(self, iso3, iso2):
        # WB admin0 uses non-standard alpha-2 (GZ/YF/RY); OSM needs the real codes.
        assert resolve_iso2(iso3) == iso2

    def test_case_insensitive(self):
        assert resolve_iso2("lao") == "LA"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_iso2("ZZZ")


class TestResolveCountryName:
    def test_returns_nonempty_for_all(self):
        for iso3 in ALL_ISO3:
            assert resolve_country_name(iso3)
