# Databricks notebook source
# Pipeline test gate. Runs the full suite (including the `databricks`-marked H3
# tests) in-process on the cluster so those tests use the cluster's native Spark
# session and real H3 SQL. Raises on any failure so the pipeline job does not
# proceed with broken code.
#
# Wired into databricks.yml as the first task of the pipeline job (added when the
# extract+transform jobs are merged).

# COMMAND ----------

# MAGIC %pip install pytest pycountry

# COMMAND ----------

import os
import sys

import pytest

# Make the repo root (containing shared/, transform/, tests/) importable and the
# working directory. On Databricks the notebook CWD is not the repo, so derive the
# root from this notebook's own workspace path (this file lives at <root>/tests/run_tests).
def _repo_root() -> str:
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        nb_path = ctx.notebookPath().get()  # e.g. /Users/<user>/<repo>/tests/run_tests
        return "/Workspace" + os.path.dirname(os.path.dirname(nb_path))
    except Exception:
        # Local fallback: climb from CWD until shared/ and tests/ are found.
        root = os.getcwd()
        for _ in range(5):
            if os.path.isdir(os.path.join(root, "shared")) and os.path.isdir(os.path.join(root, "tests")):
                return root
            root = os.path.dirname(root)
        return os.getcwd()


_root = _repo_root()
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)
print(f"Test gate repo root: {_root}")

# COMMAND ----------

# Run everything (no `-m 'not databricks'`) — the cluster is where H3 tests run.
import io
import contextlib

_tests_dir = os.path.join(_root, "tests")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    # Diagnostics: confirm the driver can enumerate the workspace dir (/Workspace
    # FUSE can list empty even when files are readable/importable by path).
    print("DIAG isdir(tests):", os.path.isdir(_tests_dir))
    try:
        print("DIAG listdir(tests):", sorted(os.listdir(_tests_dir))[:20])
    except Exception as e:
        print("DIAG listdir(tests) FAILED:", e)
    try:
        import shared.core  # noqa: F401
        print("DIAG import shared.core: OK")
    except Exception as e:
        print("DIAG import shared.core FAILED:", type(e).__name__, e)

    exit_code = pytest.main([_tests_dir, "-p", "no:cacheprovider", "-rA"])
_output = _buf.getvalue()
print(_output)  # full output to the driver log / cell
if exit_code != 0:
    # Surface the tail via the run's error so it's visible without cluster logs.
    raise RuntimeError(f"Test gate failed: pytest exit code {exit_code}\n{_output[-3500:]}")

print("All tests passed — pipeline may proceed.")
