"""
Bootstrap for Databricks notebooks — Free Edition & paid workspaces.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_notebook(project_root: Path | None = None) -> dict[str, str]:
    root = project_root or Path.cwd()
    if root.name == "notebooks":
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.databricks.config import catalog, load_databricks_config, schema
    from src.databricks.mlflow_setup import configure_mlflow
    from src.databricks.runtime import get_dbutils, is_databricks
    from src.databricks.tier import free_tier_defaults, get_storage_backend, is_free_edition

    ctx: dict[str, str] = {"mode": "local", "edition": "local"}

    if not is_databricks():
        return ctx

    # Auto-enable free-tier profile on Free Edition workspaces
    if not os.environ.get("FMG_DATABRICKS_TIER"):
        try:
            tags = (
                __import__("src.databricks.runtime", fromlist=["get_spark"])
                .get_spark()
                .conf.get("spark.databricks.clusterUsageTags.edition", "")
            )
            if "FREE" in str(tags).upper():
                os.environ["FMG_DATABRICKS_TIER"] = "free"
        except Exception:
            pass

    defaults = free_tier_defaults()
    dbu = get_dbutils()
    if dbu is not None:
        dbu.widgets.dropdown("edition", "free" if is_free_edition() else "standard", ["free", "standard"], "Edition")
        dbu.widgets.text("uc_catalog", catalog(), "Catalog")
        dbu.widgets.text("uc_schema", schema(), "Schema")
        default_rows = "100000" if is_free_edition() else os.environ.get("FMG_ROW_COUNT", "5000000")
        dbu.widgets.text("row_count", default_rows, "Rows (synthetic / sample)")
        groq_default = "false" if is_free_edition() else "true"
        dbu.widgets.dropdown("use_groq", groq_default, ["true", "false"], "GROQ FE")

        if dbu.widgets.get("edition") == "free":
            os.environ["FMG_DATABRICKS_TIER"] = "free"
        os.environ["FMG_UC_CATALOG"] = dbu.widgets.get("uc_catalog")
        os.environ["FMG_UC_SCHEMA"] = dbu.widgets.get("uc_schema")
        os.environ["FMG_ROW_COUNT"] = dbu.widgets.get("row_count")

        if dbu.widgets.get("use_groq") == "false":
            os.environ["GROQ_DISABLED"] = "1"
        else:
            from src.groq_credentials import get_groq_api_key

            if not get_groq_api_key():
                try:
                    os.environ["GROQ_API_KEY"] = dbu.secrets.get(scope="fmg", key="groq_api_key")
                except Exception:
                    pass

    if is_free_edition() and defaults.get("groq_enabled") is False:
        os.environ["GROQ_DISABLED"] = "1"

    from src.groq_credentials import get_groq_api_key

    if get_groq_api_key() and os.environ.get("GROQ_DISABLED", "").lower() not in ("1", "true", "yes"):
        os.environ.pop("GROQ_DISABLED", None)

    configure_mlflow()
    backend = get_storage_backend()

    if backend == "uc_delta":
        from src.databricks.uc_io import ensure_catalog_objects

        ensure_catalog_objects()
        spark = __import__("src.databricks.runtime", fromlist=["get_spark"]).get_spark()
        spark.sql(f"USE CATALOG {catalog()}")
        spark.sql(f"USE SCHEMA {schema()}")
    elif backend == "dbfs_delta":
        spark = __import__("src.databricks.runtime", fromlist=["get_spark"]).get_spark()
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog()}.{schema()}")

    ctx = {
        "mode": "databricks",
        "edition": "free" if is_free_edition() else "standard",
        "storage": backend,
        "catalog": catalog(),
        "schema": schema(),
        "experiment": load_databricks_config()["mlflow"]["experiment_name"],
        "max_sample_rows": str(defaults.get("max_sample_rows", 100_000)),
    }
    return ctx
