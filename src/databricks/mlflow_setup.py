"""MLflow — workspace tracking on Free Edition; UC registry when available."""

from __future__ import annotations

import os
from typing import Any

import mlflow

from src.databricks.config import catalog, load_databricks_config, schema
from src.databricks.runtime import is_databricks
from src.databricks.tier import free_tier_defaults, is_free_edition


def configure_mlflow(experiment_name: str | None = None) -> str:
    dbx = load_databricks_config()
    exp = experiment_name or dbx["mlflow"]["experiment_name"]
    defaults = free_tier_defaults()

    if is_databricks():
        use_uc = defaults.get("use_uc_registry", not is_free_edition())
        if use_uc:
            try:
                mlflow.set_registry_uri("databricks-uc")
            except Exception:
                mlflow.set_registry_uri("databricks")
        else:
            mlflow.set_registry_uri("databricks")

        exp_id = dbx["mlflow"].get("experiment_id")
        if exp_id:
            mlflow.set_experiment(experiment_id=exp_id)
        else:
            try:
                mlflow.set_experiment(exp)
            except Exception:
                mlflow.create_experiment(exp)
                mlflow.set_experiment(exp)
    else:
        tracking = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(exp.split("/")[-1])

    return exp


def registered_model_name(tier: str, algorithm: str) -> str:
    prefix = load_databricks_config()["mlflow"]["model_registry_prefix"]
    if is_free_edition() and not free_tier_defaults().get("use_uc_registry", False):
        return f"{prefix}_{tier}_{algorithm}"
    return f"{catalog()}.{schema()}.{prefix}_{tier}_{algorithm}"


def log_sklearn_model_to_uc(
    model: Any,
    artifact_path: str,
    tier: str,
    algorithm: str,
    metrics: dict[str, float],
    params: dict[str, str | int | float],
) -> str:
    configure_mlflow()
    for k, v in params.items():
        mlflow.log_param(k, v)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    import mlflow.sklearn as mlflow_sklearn

    name = registered_model_name(tier, algorithm)
    register = free_tier_defaults().get("register_models", True) and is_databricks()

    if register and not is_free_edition():
        mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=name,
        )
    elif register and is_free_edition():
        try:
            mlflow_sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=name,
            )
        except Exception:
            mlflow_sklearn.log_model(model, artifact_path=artifact_path)
            name = artifact_path
    else:
        mlflow_sklearn.log_model(model, artifact_path=artifact_path)
        name = artifact_path
    return name
