"""Delta I/O via Unity Catalog or DBFS paths (Free Edition friendly)."""

from __future__ import annotations

import pandas as pd

from src.databricks.config import load_databricks_config, table_fqn
from src.databricks.runtime import get_spark
from src.databricks.tier import _load_merged_dbx_config, use_dbfs_delta_storage


def _dbfs_path(logical_name: str) -> str:
    cfg = _load_merged_dbx_config()
    base = cfg.get("storage", {}).get("dbfs_base", "dbfs:/FileStore/fmg/regional_bank")
    return f"{base}/{logical_name}"


def write_delta(df: pd.DataFrame, logical_name: str) -> str:
    spark = get_spark()
    path = _dbfs_path(logical_name)
    spark.createDataFrame(df).write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    return path


def read_delta(logical_name: str, sample_n: int | None = None) -> pd.DataFrame:
    spark = get_spark()
    path = _dbfs_path(logical_name)
    sdf = spark.read.format("delta").load(path)
    if sample_n:
        total = sdf.count()
        if total > sample_n:
            frac = sample_n / total
            sdf = sdf.sample(withReplacement=False, fraction=frac, seed=42)
    return sdf.toPandas()


def register_dbfs_as_table(logical_name: str) -> str:
    """Register DBFS Delta path as UC external table for SQL / dashboards."""
    if not use_dbfs_delta_storage():
        return table_fqn(logical_name)
    spark = get_spark()
    path = _dbfs_path(logical_name)
    fqn = table_fqn(logical_name)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {fqn} USING DELTA LOCATION '{path}'")
    return fqn
