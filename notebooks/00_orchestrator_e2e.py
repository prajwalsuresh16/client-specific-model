# Databricks notebook source
# MAGIC %md
# MAGIC # FMG E2E Orchestrator (recommended for **Free Edition**)
# MAGIC Single serverless run: synthetic → steps 01–08 → MLflow → SQL KPIs.
# MAGIC Saves job-task quota vs multi-task workflow.
# MAGIC
# MAGIC **Free Edition tips:** keep `row_count` ≤ 100k; leave GROQ off (rule-based FE).

# COMMAND ----------
# MAGIC %pip install -r ../requirements-databricks.txt -q

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook

ctx = bootstrap_notebook(root)
print("Runtime context:", ctx)

# COMMAND ----------
import os

os.environ["FMG_DATABRICKS_TIER"] = ctx.get("edition", "free")
sample_n = int(ctx.get("max_sample_rows", 100_000))

# COMMAND ----------
# MAGIC %md ## 00 — Synthetic bronze (+ prior)
from src.synthetic_data import generate_regional_bank

paths = generate_regional_bank()
display(list(paths.items()))  # noqa: F821 — Databricks display

# COMMAND ----------
if ctx.get("storage") in ("uc_delta", "dbfs_delta"):
    from src.databricks.sync_bronze import sync_all_bronze_to_delta

    synced = sync_all_bronze_to_delta()
    print("Synced Delta tables:", synced)

# COMMAND ----------
# MAGIC %md ## 01 — MRGAL
from src.pipeline import step01_mrgal

step01_mrgal.run()

# COMMAND ----------
# MAGIC %md ## 02 — Sample prep
from src.pipeline import step02_sample_prep

step02_sample_prep.run()

# COMMAND ----------
# MAGIC %md ## 03 — Modeling (MLflow)
import mlflow
from src.pipeline import step03_modeling

with mlflow.start_run(run_name="fmg_e2e_orchestrator"):
    mlflow.set_tag("edition", ctx.get("edition", "unknown"))
    mlflow.log_param("sample_n", sample_n)
    out03 = step03_modeling.run(sample_n=sample_n)
    mlflow.log_param("predictions_path", str(out03))

# COMMAND ----------
# MAGIC %md ## 04–08 — Validation through SO
from src.pipeline import step04_validation, step05_recommendation, step06_rank_mix, step07_lol, step08_so_output

step04_validation.run(sample_n=sample_n)
step05_recommendation.run()
step06_rank_mix.run(sample_n=sample_n)
step07_lol.run()
so_path = step08_so_output.run()
print("SO output:", so_path)

# COMMAND ----------
# MAGIC %md ## SQL KPIs (SQL Editor / `%sql` — works on Free Edition)
from src.databricks.config import catalog, schema, table_fqn

spark.sql(f"USE CATALOG {catalog()}")  # noqa: F821
spark.sql(f"USE SCHEMA {schema()}")  # noqa: F821

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Mail keep rate (requires gold_so_output registered as Delta table)
# MAGIC SELECT
# MAGIC   campaign_id,
# MAGIC   COUNT(*) AS sd_rows,
# MAGIC   SUM(CASE WHEN keep_flag = 'Y' THEN 1 ELSE 0 END) AS mail_count,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN keep_flag = 'Y' THEN 1 ELSE 0 END) / COUNT(*), 2) AS mail_pct
# MAGIC FROM gold_so_output
# MAGIC GROUP BY campaign_id;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT combo_id, model_a, model_b, ROUND(auc, 4) AS auc, segment_id
# MAGIC FROM gold_validation_results
# MAGIC WHERE segment_id = 'ALL'
# MAGIC ORDER BY auc DESC
# MAGIC LIMIT 10;

# COMMAND ----------
display(spark.table(table_fqn("gold_list_of_lists")).limit(50))  # noqa: F821
