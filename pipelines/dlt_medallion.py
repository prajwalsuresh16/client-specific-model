# Databricks Lakeflow / Delta Live Tables — bronze → silver MRGAL refresh
# Deploy via Asset Bundle resource: resources.pipelines.fmg_medallion

import dlt
from pyspark.sql import functions as F


def _table(name: str) -> str:
    import os

    cat = os.environ.get("FMG_UC_CATALOG", "fmg_datahub")
    sch = os.environ.get("FMG_UC_SCHEMA", "regional_bank")
    return f"{cat}.{sch}.{name}"


@dlt.table(
    name="dlt_bronze_sd",
    comment="DLT view of bronze SD for medallion pipeline",
)
def dlt_bronze_sd():
    return spark.table(_table("bronze_sd"))


@dlt.table(
    name="dlt_silver_mrgal_live",
    comment="DLT silver MRGAL — mirrors batch step01 output",
)
def dlt_silver_mrgal_live():
    sd = dlt.read("dlt_bronze_sd")
    promo = spark.table(_table("bronze_stat_promo"))
    keys = ["bpid", "indiv_id", "campaign_id", "client_id", "cut_date", "product_code"]
    out = sd
    for stat, suffix in [(promo, "_promo")]:
        stat_cols = [c for c in stat.columns if c not in keys]
        renamed = [F.col(c).alias(f"{c}{suffix}" if c in sd.columns else c) for c in stat_cols]
        part = stat.select(*[F.col(k) for k in keys], *renamed)
        out = out.join(part, on=keys, how="left")
    return out
