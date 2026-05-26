"""Unity Catalog Delta read/write (replaces local Parquet on Databricks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.databricks.config import table_fqn, volume_path
from src.databricks.runtime import get_spark, is_databricks, use_unity_catalog


def write_table(df: pd.DataFrame, logical_name: str, mode: str = "overwrite") -> str:
    fqn = table_fqn(logical_name)
    spark = get_spark()
    spark_df = spark.createDataFrame(df)
    (
        spark_df.write.format("delta")
        .mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(fqn)
    )
    return fqn


def read_table(logical_name: str, sample_n: int | None = None) -> pd.DataFrame:
    fqn = table_fqn(logical_name)
    spark = get_spark()
    sdf = spark.table(fqn)
    if sample_n:
        frac = min(1.0, sample_n / max(sdf.count(), 1))
        if frac < 1.0:
            sdf = sdf.sample(withReplacement=False, fraction=frac, seed=42)
    return sdf.toPandas()


def write_volume_file(local_path: Path, volume_subpath: str) -> str:
    dest = volume_path(volume_subpath)
    dbu = __import__("src.databricks.runtime", fromlist=["get_dbutils"]).get_dbutils()
    if dbu is not None:
        dbu.fs.cp(f"file:{local_path}", dest, recurse=True)
    return dest


def ensure_catalog_objects() -> list[str]:
    """DDL for catalog, schema, volume — run on cluster or SQL warehouse."""
    from src.databricks.config import catalog as cat, load_databricks_config, schema as sch

    cfg = load_databricks_config()
    vol = cfg["unity_catalog"].get("bronze_volume") or ""
    stmts = [
        f"CREATE CATALOG IF NOT EXISTS {cat()}",
        f"CREATE SCHEMA IF NOT EXISTS {cat()}.{sch()}",
    ]
    if vol:
        stmts.append(f"CREATE VOLUME IF NOT EXISTS {cat()}.{sch()}.{vol}")
    if is_databricks():
        spark = get_spark()
        for s in stmts:
            try:
                spark.sql(s)
            except Exception:
                if "VOLUME" in s:
                    continue
                raise
    return stmts
