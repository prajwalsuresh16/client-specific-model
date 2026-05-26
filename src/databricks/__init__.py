"""Databricks platform adapters (Unity Catalog, MLflow, Feature Store, Jobs)."""

from src.databricks.runtime import is_databricks, use_unity_catalog

__all__ = ["is_databricks", "use_unity_catalog"]
