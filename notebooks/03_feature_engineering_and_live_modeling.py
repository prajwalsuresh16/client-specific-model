# Databricks notebook source
# MAGIC %md # 03 — Indexing + GROQ FE + Live Modeling
# MAGIC **MLflow** (UC registry) · **Feature Store** · cluster secrets for `GROQ_API_KEY`

# COMMAND ----------
import sys
from pathlib import Path

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(root))

from src.databricks.bootstrap import bootstrap_notebook

bootstrap_notebook(root)

# COMMAND ----------
from src.pipeline.step03_modeling import run

# Widget override for dev sampling
from src.databricks.runtime import get_widget

sample_n = int(get_widget("sample_n", "50000") or "50000")
print(run(sample_n=sample_n))
