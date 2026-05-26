"""Sync local/volume Parquet bronze partitions into Unity Catalog Delta tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import resolve_path
from src.databricks.tier import get_storage_backend
from src.io_utils import read_parquet_dir


BRONZE_ENTITIES = [
    "bronze_sd",
    "bronze_stat_promo",
    "bronze_stat_membership",
    "bronze_stat_demo",
    "bronze_mrgal",
    "bronze_responders",
]


def sync_parquet_dir_to_table(local_dir: Path, logical_name: str, sample_n: int | None = None) -> str:
    df = read_parquet_dir(local_dir, sample_n=sample_n)
    backend = get_storage_backend()
    if backend == "uc_delta":
        from src.databricks.uc_io import write_table

        return write_table(df, logical_name)
    if backend == "dbfs_delta":
        from src.databricks.delta_io import register_dbfs_as_table, write_delta

        path = write_delta(df, logical_name)
        try:
            return register_dbfs_as_table(logical_name)
        except Exception:
            return path
    return str(local_dir)


def sync_all_bronze_to_delta(include_prior: bool = True) -> list[str]:
    out: list[str] = []
    root = resolve_path("data", "regional_bank", "bronze")
    for entity in BRONZE_ENTITIES:
        path = root / entity
        if path.exists():
            out.append(sync_parquet_dir_to_table(path, entity))
    if include_prior:
        prior_root = resolve_path("data", "regional_bank", "bronze_prior")
        prior_map = {
            "bronze_sd": "bronze_prior_sd",
            "bronze_responders": "bronze_responders",
        }
        for folder, logical in prior_map.items():
            path = prior_root / folder
            if path.exists():
                out.append(sync_parquet_dir_to_table(path, logical))
    return out
