"""Load Databricks platform config (Unity Catalog, warehouse, MLflow)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_databricks_config() -> dict[str, Any]:
    path = _ROOT / "config" / "databricks.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    uc = cfg.setdefault("unity_catalog", {})
    if os.environ.get("FMG_UC_CATALOG"):
        uc["catalog"] = os.environ["FMG_UC_CATALOG"]
    if os.environ.get("FMG_UC_SCHEMA"):
        uc["schema"] = os.environ["FMG_UC_SCHEMA"]

    wh = cfg.setdefault("sql_warehouse", {})
    if os.environ.get("FMG_SQL_WAREHOUSE_ID"):
        wh["warehouse_id"] = os.environ["FMG_SQL_WAREHOUSE_ID"]

    return cfg


def catalog() -> str:
    return load_databricks_config()["unity_catalog"]["catalog"]


def schema() -> str:
    return load_databricks_config()["unity_catalog"]["schema"]


def table_fqn(logical_name: str) -> str:
    dbx = load_databricks_config()
    short = dbx["tables"].get(logical_name, logical_name)
    return f"{catalog()}.{schema()}.{short}"


def volume_path(subpath: str = "") -> str:
    dbx = load_databricks_config()
    uc = dbx["unity_catalog"]
    base = f"/Volumes/{uc['catalog']}/{uc['schema']}/{uc['bronze_volume']}"
    return f"{base}/{subpath}".rstrip("/") if subpath else base
