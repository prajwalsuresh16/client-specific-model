"""Databricks Feature Store — publish indexed features for training/scoring."""

from __future__ import annotations

import pandas as pd

from src.databricks.config import load_databricks_config, table_fqn
from src.databricks.runtime import is_databricks, use_unity_catalog


def publish_feature_table(
    df: pd.DataFrame,
    logical_name: str,
    primary_keys: list[str] | None = None,
) -> str | None:
    if not is_databricks():
        return None

    from src.databricks.tier import is_free_edition

    cfg = load_databricks_config()
    if is_free_edition() or not cfg.get("feature_store", {}).get("enabled", True):
        return None

    from src.databricks.tier import use_unity_catalog_storage

    if not use_unity_catalog_storage():
        return None

    from databricks.feature_engineering import FeatureEngineeringClient

    prefix = cfg["feature_store"]["feature_table_prefix"]
    fs_logical = f"{logical_name}_{prefix}"
    fqn = table_fqn(fs_logical)

    fe = FeatureEngineeringClient()
    keys = primary_keys or [c for c in ("bpid", "indiv_id", "campaign_id") if c in df.columns]
    spark = __import__("src.databricks.runtime", fromlist=["get_spark"]).get_spark()
    spark_df = spark.createDataFrame(df)

    try:
        fe.create_table(
            name=fqn,
            primary_keys=keys,
            df=spark_df,
            description=f"FMG indexed features — {logical_name}",
        )
    except Exception:
        from src.databricks.uc_io import write_table

        write_table(df, fs_logical)
    return fqn
