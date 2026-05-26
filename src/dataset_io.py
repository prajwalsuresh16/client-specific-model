"""Unified I/O: local Parquet | Unity Catalog Delta | DBFS Delta (Free Edition)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import resolve_path
from src.databricks.runtime import is_databricks
from src.databricks.tier import get_storage_backend, use_dbfs_delta_storage, use_unity_catalog_storage
from src.io_utils import ensure_dir, read_parquet_dir, write_parquet_partitioned


def _logical_name(name: str) -> str:
    return name.replace(".parquet", "")


def read_dataset(layer: str, name: str, sample_n: int | None = None) -> pd.DataFrame:
    logical = _logical_name(name)
    backend = get_storage_backend()

    if backend == "uc_delta" and is_databricks():
        from src.databricks.uc_io import read_table

        return read_table(logical, sample_n=sample_n)

    if backend == "dbfs_delta" and is_databricks():
        from src.databricks.delta_io import read_delta

        return read_delta(logical, sample_n=sample_n)

    base = resolve_path("data", "regional_bank", layer)
    path = base / name if name.endswith(".parquet") else base / f"{name}.parquet"
    if path.exists():
        return read_parquet_dir(path, sample_n=sample_n)
    return read_parquet_dir(base / logical, sample_n=sample_n)


def write_dataset(df: pd.DataFrame, layer: str, name: str) -> Path:
    logical = _logical_name(name)
    backend = get_storage_backend()

    if backend == "uc_delta" and is_databricks():
        from src.databricks.runtime import set_task_value
        from src.databricks.uc_io import write_table

        fqn = write_table(df, logical)
        set_task_value(logical, fqn)
        return Path(fqn.replace(".", "/"))

    if backend == "dbfs_delta" and is_databricks():
        from src.databricks.delta_io import register_dbfs_as_table, write_delta
        from src.databricks.runtime import set_task_value

        path = write_delta(df, logical)
        try:
            fqn = register_dbfs_as_table(logical)
            set_task_value(logical, fqn)
        except Exception:
            set_task_value(logical, path)
        return Path(path.replace("dbfs:", "/dbfs/"))

    base = ensure_dir(resolve_path("data", "regional_bank", layer))
    return write_parquet_partitioned(df, base, logical)


def read_bronze(table_dir: str, sample_n: int | None = None) -> pd.DataFrame:
    mapping = {
        "bronze_sd": "bronze_sd",
        "bronze_stat_promo": "bronze_stat_promo",
        "bronze_stat_membership": "bronze_stat_membership",
        "bronze_stat_demo": "bronze_stat_demo",
        "bronze_mrgal": "bronze_mrgal",
        "bronze_responders": "bronze_responders",
        "bronze_so": "bronze_so",
    }
    logical = mapping.get(table_dir, table_dir)
    backend = get_storage_backend()

    if backend == "uc_delta" and is_databricks():
        from src.databricks.uc_io import read_table

        return read_table(logical, sample_n=sample_n)

    if backend == "dbfs_delta" and is_databricks():
        from src.databricks.delta_io import read_delta

        return read_delta(logical, sample_n=sample_n)

    return read_parquet_dir(resolve_path("data", "regional_bank", "bronze", table_dir), sample_n=sample_n)


def read_bronze_prior(table_dir: str, sample_n: int | None = None) -> pd.DataFrame:
    prior_map = {"bronze_sd": "bronze_prior_sd", "bronze_responders": "bronze_responders"}
    logical = prior_map.get(table_dir, f"bronze_prior_{table_dir.replace('bronze_', '')}")
    backend = get_storage_backend()

    if backend == "uc_delta" and is_databricks():
        from src.databricks.uc_io import read_table

        try:
            return read_table(logical, sample_n=sample_n)
        except Exception:
            return read_table(table_dir, sample_n=sample_n)

    if backend == "dbfs_delta" and is_databricks():
        from src.databricks.delta_io import read_delta

        try:
            return read_delta(logical, sample_n=sample_n)
        except Exception:
            return read_delta(table_dir, sample_n=sample_n)

    return read_parquet_dir(resolve_path("data", "regional_bank", "bronze_prior", table_dir), sample_n=sample_n)


# Backward-compatible helpers
def use_unity_catalog() -> bool:
    return use_unity_catalog_storage()
