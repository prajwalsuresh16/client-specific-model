"""Databricks edition detection — Free Edition vs paid workspace."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

StorageBackend = Literal["parquet", "uc_delta", "dbfs_delta"]


def is_free_edition() -> bool:
    if os.environ.get("FMG_DATABRICKS_TIER", "").lower() in ("free", "community", "ce"):
        return True
    if os.environ.get("DATABRICKS_EDITION", "").upper() in ("FREE", "COMMUNITY"):
        return True
    return False


def _load_merged_dbx_config() -> dict:
    from src.databricks.config import load_databricks_config

    base = load_databricks_config()
    if not is_free_edition():
        return base
    path = os.environ.get(
        "FMG_DATABRICKS_CONFIG",
        str(__import__("pathlib").Path(__file__).resolve().parents[2] / "config" / "databricks_free.yaml"),
    )
    import yaml

    with open(path, encoding="utf-8") as f:
        free = yaml.safe_load(f)
    merged = {**base, **free}
    merged["unity_catalog"] = {**base.get("unity_catalog", {}), **free.get("unity_catalog", {})}
    merged["mlflow"] = {**base.get("mlflow", {}), **free.get("mlflow", {})}
    merged["groq"] = {**base.get("groq", {}), **free.get("groq", {})}
    return merged


@lru_cache(maxsize=1)
def get_storage_backend() -> StorageBackend:
    from src.databricks.runtime import is_databricks

    if not is_databricks():
        return "parquet"

    cfg = _load_merged_dbx_config()
    primary = cfg.get("storage", {}).get("primary", "uc_delta")

    if primary == "dbfs_delta":
        return "dbfs_delta"

    if primary == "uc_delta":
        try:
            from src.databricks.runtime import get_spark

            spark = get_spark()
            spark.sql("SELECT 1")
            if cfg.get("unity_catalog", {}).get("auto_detect_catalog", False):
                cat = spark.sql("SELECT current_catalog()").collect()[0][0]
                sch = spark.sql("SELECT current_schema()").collect()[0][0]
                os.environ.setdefault("FMG_UC_CATALOG", cat)
                os.environ.setdefault("FMG_UC_SCHEMA", sch)
            return "uc_delta"
        except Exception:
            return "dbfs_delta"

    return "parquet"


def use_unity_catalog_storage() -> bool:
    return get_storage_backend() == "uc_delta"


def use_dbfs_delta_storage() -> bool:
    return get_storage_backend() == "dbfs_delta"


def free_tier_defaults() -> dict:
    cfg = _load_merged_dbx_config()
    return {
        "max_sample_rows": int(cfg.get("job", {}).get("max_sample_rows", 100_000)),
        "groq_enabled": bool(cfg.get("groq", {}).get("enabled", False)),
        "register_models": bool(cfg.get("mlflow", {}).get("register_models", True)),
        "use_uc_registry": bool(cfg.get("mlflow", {}).get("use_uc_registry", False)),
    }
