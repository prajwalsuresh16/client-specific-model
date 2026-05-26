# Databricks notebook source
# MAGIC %md # 00 — Synthetic Regional Bank Bronze (+ prior campaign)
# MAGIC Writes Parquet locally; on Databricks syncs to **Unity Catalog Delta** + **Volume**.

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook

ctx = bootstrap_notebook(root)

# COMMAND ----------
from src.synthetic_data import generate_regional_bank

paths = generate_regional_bank()
for k, v in paths.items():
    print(k, v)

# COMMAND ----------
if ctx.get("mode") == "databricks":
    from src.databricks.sync_bronze import sync_all_bronze_to_delta
    from src.databricks.runtime import set_task_value

    tables = sync_all_bronze_to_delta()
    for t in tables:
        print("Delta table:", t)
    set_task_value("bronze_sync_complete", "true")
