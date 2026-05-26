"""Detect Databricks runtime and expose Spark / dbutils / task values."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def is_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def use_unity_catalog() -> bool:
    if os.environ.get("FMG_FORCE_LOCAL_IO", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("FMG_USE_UNITY_CATALOG", "").lower() in ("1", "true", "yes"):
        return True
    if is_databricks():
        from src.databricks.tier import get_storage_backend

        return get_storage_backend() == "uc_delta"
    return False


def get_spark() -> Any:
    if not is_databricks():
        raise RuntimeError("Spark is only available on Databricks runtime")
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def get_dbutils() -> Any:
    if not is_databricks():
        return None
    try:
        from pyspark.dbutils import DBUtils

        return DBUtils(get_spark())
    except ImportError:
        import IPython

        return IPython.get_ipython().user_ns["dbutils"]


@lru_cache(maxsize=1)
def workspace_user() -> str:
    if not is_databricks():
        return os.environ.get("USER", "local")
    dbu = get_dbutils()
    if dbu is None:
        return "unknown"
    try:
        return dbu.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except Exception:
        return os.environ.get("DATABRICKS_USER", "unknown")


def set_task_value(key: str, value: str) -> None:
    if not is_databricks():
        os.environ[f"FMG_TASK_{key.upper()}"] = value
        return
    dbu = get_dbutils()
    if dbu is not None:
        dbu.jobs.taskValues.set(key=key, value=value)


def get_task_value(key: str, default: str = "") -> str:
    if not is_databricks():
        return os.environ.get(f"FMG_TASK_{key.upper()}", default)
    dbu = get_dbutils()
    if dbu is None:
        return default
    try:
        return dbu.jobs.taskValues.get(taskKey=os.environ.get("FMG_TASK_KEY", ""), key=key, default=default)
    except Exception:
        try:
            return dbu.jobs.taskValues.get(key=key, default=default)
        except Exception:
            return default


def get_widget(name: str, default: str = "") -> str:
    if not is_databricks():
        return os.environ.get(name.upper(), default)
    dbu = get_dbutils()
    if dbu is None:
        return default
    try:
        return dbu.widgets.get(name)
    except Exception:
        return default
