# Databricks notebook source
# MAGIC %md # 02 — Auto Sample Prep (prior labeled / current scoring)

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook

bootstrap_notebook(root)

# COMMAND ----------
from src.pipeline.step02_sample_prep import run

print(run())
