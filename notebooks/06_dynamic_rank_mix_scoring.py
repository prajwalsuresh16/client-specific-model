# Databricks notebook source
# MAGIC %md # 06 — Dynamic Rank-Mix Scoring

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook
from src.databricks.runtime import get_widget

bootstrap_notebook(root)

# COMMAND ----------
from src.pipeline.step06_rank_mix import run

sample_n = int(get_widget("sample_n", "50000") or "50000")
print(run(sample_n=sample_n))
