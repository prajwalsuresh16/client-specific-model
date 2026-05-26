# Databricks notebook source
# MAGIC %md # 01 — Auto MRGAL creation (Silver Delta table)

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook
from src.databricks.runtime import set_task_value

ctx = bootstrap_notebook(root)

# COMMAND ----------
from src.pipeline.step01_mrgal import run

paths = run()
for k, v in paths.items():
    print(k, v)
    if ctx.get("mode") == "databricks":
        set_task_value(k, str(v))
