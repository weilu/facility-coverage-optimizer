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

# With source: GIT, Databricks checks the repo out to a local driver path and the
# task's CWD is the notebook's directory (…/tests). Derive the repo root: use CWD
# if it already contains tests/, else its parent. (A repo_root widget, if passed,
# still wins — e.g. for a bundle-deploy setup.)
def _resolve_repo_root() -> str:
    try:
        widget = (dbutils.widgets.get("repo_root") or "").strip()
    except Exception:
        widget = ""
    if widget:
        return widget
    cwd = os.getcwd()
    return cwd if os.path.isdir(os.path.join(cwd, "tests")) else os.path.dirname(cwd)


_repo_root = _resolve_repo_root()
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
