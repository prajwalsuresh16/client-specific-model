# Databricks notebook source
# MAGIC %md # 07 — Auto List-of-Lists (segment × decile)

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook

bootstrap_notebook(root)

# COMMAND ----------
from src.pipeline.step07_lol import run

print(run())
