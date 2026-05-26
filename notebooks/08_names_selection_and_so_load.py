# Databricks notebook source
# MAGIC %md # 08 — Names Selection + SO Output (100% SD parity)

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook
from src.databricks.runtime import set_task_value

ctx = bootstrap_notebook(root)

# COMMAND ----------
from src.pipeline.step08_so_output import run

out = run()
print(out)
if ctx.get("mode") == "databricks":
    set_task_value("gold_so_output", str(out))
