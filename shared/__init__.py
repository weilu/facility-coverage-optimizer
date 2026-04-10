# Shared utilities for the pipeline
#
# Module structure:
#   - shared.settings: Application constants (catalog, schema, country)
#   - shared.core: Pure functions (no Spark dependencies, unit-testable)
#   - shared.env: Environment detection and storage backends
#
# Usage:
#   from shared.settings import UC_CATALOG, UC_SCHEMA, COUNTRY
#   from shared.core import solve_mclp_greedy, get_k_rings
#   from shared.env import get_spark, table_exists, get_storage_backend
