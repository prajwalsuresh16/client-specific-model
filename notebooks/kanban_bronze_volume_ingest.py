# Databricks notebook source
# MAGIC %md
# MAGIC # Kanban bronze ingest — UC Volume → Delta
# MAGIC Reads CSV/Parquet from intake volume path and writes bronze Delta tables.

# COMMAND ----------

import re
from pathlib import PurePosixPath

from pyspark.sql import functions as F

dbutils.widgets.text("campaign_month", "202605")
dbutils.widgets.text("campaign_id", "0")
dbutils.widgets.text("uc_catalog", "fmg_datahub")
dbutils.widgets.text("uc_schema", "regional_bank")
dbutils.widgets.text("volume_path", "")

month = dbutils.widgets.get("campaign_month")
catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("uc_schema")
volume_path = dbutils.widgets.get("volume_path").rstrip("/")

BRONZE_LOGICAL = [
    "bronze_sd",
    "bronze_stat_promo",
    "bronze_stat_membership",
    "bronze_stat_demo",
    "bronze_responders",
    "bronze_prior_sd",
    "bronze_so",
]


def resolve_logical(name: str) -> str | None:
    base = name.lower()
    if not base.endswith((".csv", ".parquet")):
        return None
    stem = PurePosixPath(base).stem
    for hint in sorted(BRONZE_LOGICAL, key=len, reverse=True):
        if hint in stem or stem == hint:
            return hint
    return None


# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

files = dbutils.fs.ls(volume_path) if volume_path else []
loaded = []

for f in files:
    logical = resolve_logical(f.name)
    if not logical:
        continue
    path = f.path
    if f.name.lower().endswith(".csv"):
        df = spark.read.option("header", True).csv(path)
    else:
        df = spark.read.parquet(path)
    if "indiv_id" not in df.columns:
        if "bpid" in df.columns:
            df = df.withColumn("indiv_id", F.col("bpid"))
        else:
            for alt in ("individual_id", "inidivid_id", "individ_id"):
                if alt in df.columns:
                    df = df.withColumn("indiv_id", F.col(alt))
                    break
    target = f"{catalog}.{schema}.{logical}"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
    loaded.append((logical, df.count()))

# COMMAND ----------

display(spark.createDataFrame(loaded, ["table", "row_count"]))
