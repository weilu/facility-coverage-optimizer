# Databricks notebook source
# DAB test gate. This cluster doesn't mount /Workspace to the driver, so pytest
# can't discover files there; instead install the wheel (tests bundled) from the UC
# Volume (dist_dir, per target) and run `pytest --pyargs tests` from site-packages.

# COMMAND ----------

import sys
import subprocess

dist_dir = (dbutils.widgets.get("dist_dir") or "").strip()
if not dist_dir:
    raise RuntimeError("'dist_dir' is not set — expected from the DAB job's base_parameters.")

# Install the project wheel (pulls its own deps: geopandas/shapely/pycountry/…) plus pytest.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--find-links", dist_dir,
     "health-facility-location-optimizer", "pytest"],
    check=True,
)
# Force the package itself to the just-published wheel (version is static, so a
# plain install would keep a previously-installed copy); deps are already satisfied.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--find-links", dist_dir,
     "--force-reinstall", "--no-deps", "health-facility-location-optimizer"],
    check=True,
)

# COMMAND ----------

import io
import contextlib

import pytest

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    # --pyargs discovers the installed `tests` package; no workspace-file enumeration.
    exit_code = pytest.main(["-p", "no:cacheprovider", "-rA", "--pyargs", "tests"])
_output = _buf.getvalue()
print(_output)  # full output to the driver log
if exit_code != 0:
    raise RuntimeError(f"Test gate failed: pytest exit code {exit_code}\n{_output[-3500:]}")

print("All tests passed — pipeline may proceed.")
