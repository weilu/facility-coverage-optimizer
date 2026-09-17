# Databricks notebook source
# DAB test gate. Runs the full pytest suite (including the @pytest.mark.databricks
# H3 tests) on the cluster. The repo root is supplied by the DAB as the
# `repo_root` base_parameter (${workspace.file_path}) — the deploy-time-known
# bundle root — rather than guessed from the notebook path.

# COMMAND ----------

# MAGIC %pip install pytest pycountry

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass  # not on Databricks

# COMMAND ----------

import os
import sys
import io
import contextlib

import pytest

_repo_root = (dbutils.widgets.get("repo_root") or "").strip()
if not _repo_root:
    raise RuntimeError(
        "'repo_root' is not set — expected from the DAB job's base_parameters "
        "(repo_root: ${workspace.file_path})."
    )
os.chdir(_repo_root)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
print(f"Test gate repo root: {_repo_root}")

# COMMAND ----------

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    exit_code = pytest.main(["tests", "-p", "no:cacheprovider", "-rA"])
_output = _buf.getvalue()
print(_output)  # full output to the driver log
if exit_code != 0:
    # Surface the tail via the run's error so it's visible without cluster logs.
    raise RuntimeError(f"Test gate failed: pytest exit code {exit_code}\n{_output[-3500:]}")

print("All tests passed — pipeline may proceed.")
