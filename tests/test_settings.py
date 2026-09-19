import json
import pathlib
import re

import pytest

from shared.settings import resolve_iso2, resolve_country_name, _parse_bool


@pytest.fixture(scope="module")
def default_countries():
    # Read the batch country list from databricks.yml so this never drifts from the
    # actual default. Not bundled in the wheel, so the on-cluster (wheel) run skips.
    yml = pathlib.Path(__file__).resolve().parents[1] / "databricks.yml"
    if not yml.exists():
        pytest.skip("databricks.yml not available (wheel-installed run)")
    m = re.search(r"countries:.*?default:\s*'(\[.*?\])'", yml.read_text(), re.S)
    assert m, "countries variable default not found in databricks.yml"
    return json.loads(m.group(1))


class TestResolveIso2:
    def test_all_default_countries_resolve(self, default_countries):
        for iso3 in default_countries:
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
    def test_returns_nonempty_for_all(self, default_countries):
        for iso3 in default_countries:
            assert resolve_country_name(iso3)


class TestParseBool:
    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("True", True), ("1", True), ("yes", True),
        ("false", False), ("no", False), ("", False),
    ])
    def test_parses_truthy_strings(self, raw, expected):
        assert _parse_bool(raw) is expected
