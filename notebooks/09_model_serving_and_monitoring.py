# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Unity Catalog Model Serving + Monitoring
# MAGIC Registers champion model from MLflow UC and optionally deploys a Model Serving endpoint.

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook

ctx = bootstrap_notebook(root)
print("Context:", ctx)

# COMMAND ----------
from src.databricks.config import load_databricks_config, table_fqn
from src.databricks.runtime import is_databricks, set_task_value
from src.dataset_io import read_dataset

from src.databricks.tier import is_free_edition

if not is_databricks():
    dbutils.notebook.exit("Local mode — skip serving deployment")  # type: ignore[name-defined]

if is_free_edition():
    dbutils.notebook.exit("Free Edition — model serving skipped (enable in paid workspace or if quota allows)")  # type: ignore[name-defined]

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

dbx = load_databricks_config()
rec = read_dataset("gold", "gold_auto_recommendation").iloc[0]
tier_algo = rec["recommended_model_a"].replace("prob_", "").split("_", 1)
if len(tier_algo) == 2:
    tier, algo = tier_algo
else:
    tier, algo = "client_product", "xgboost"

from src.databricks.mlflow_setup import registered_model_name

model_name = registered_model_name(tier, algo)
client = MlflowClient(registry_uri="databricks-uc")

versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    raise RuntimeError(f"No UC model versions for {model_name}")

latest = max(versions, key=lambda v: int(v.version))
client.set_registered_model_alias(model_name, "Champion", latest.version)
set_task_value("champion_model", model_name)
set_task_value("champion_version", str(latest.version))

# COMMAND ----------
# Model Serving endpoint (Databricks Model Serving)
endpoint_name = dbx["model_serving"]["endpoint_name"]
if dbx.get("model_serving", {}).get("enabled", True):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

    w = WorkspaceClient()
    try:
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_models=[
                    ServedModelInput(
                        model_name=model_name,
                        model_version=latest.version,
                        workload_size=dbx["model_serving"].get("workload_size", "Small"),
                        scale_to_zero_enabled=dbx["model_serving"].get("scale_to_zero", True),
                    )
                ]
            ),
        )
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise
    set_task_value("serving_endpoint", endpoint_name)

# COMMAND ----------
# Quality monitor stub — log SO table metrics to MLflow
with mlflow.start_run(run_name="post_scoring_monitor"):
    so = read_dataset("gold", "gold_so_output", sample_n=100_000)
    mail_pct = (so["keep_flag"] == "Y").mean()
    mlflow.log_metric("mail_pct", float(mail_pct))
    mlflow.log_metric("so_row_count", float(len(so)))
    mlflow.set_tag("source_table", table_fqn("gold_so_output"))

print("Champion:", model_name, "v", latest.version)
