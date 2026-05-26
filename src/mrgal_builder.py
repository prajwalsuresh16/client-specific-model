"""Shared MRGAL (merge-all) construction from bronze SD + STAT tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.schemas import KEY_COLUMNS

MRGAL_KEYS = list(KEY_COLUMNS)


def _read_bronze_table(bronze_root: Path, table_name: str, sample_n: int | None, *, prior: bool = False) -> pd.DataFrame:
    from src.databricks.runtime import is_databricks, use_unity_catalog
    from src.dataset_io import read_bronze, read_bronze_prior

    if use_unity_catalog() and is_databricks():
        return (read_bronze_prior if prior else read_bronze)(table_name, sample_n=sample_n)

    from src.io_utils import read_parquet_dir

    path = bronze_root / table_name
    return read_parquet_dir(path, sample_n=sample_n)


def apply_marketable_universe_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Suppress ineligible records (legacy MDM / marketable universe)."""
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if "eligible_to_market_flag" in out.columns:
        mask &= out["eligible_to_market_flag"].fillna(0).astype(int) == 1
    for suppress_col in (
        "deceased_suppressed_flag",
        "bad_address_flag",
        "dnc_flag",
        "existing_product_suppressed_flag",
        "max_coverage_reached_flag",
    ):
        if suppress_col in out.columns:
            mask &= out[suppress_col].fillna(0).astype(int) == 0
    out["marketable_flag"] = mask.astype(np.int8)
    return out


def dedupe_mrgal(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    subset = subset or ["bpid", "campaign_id", "client_id", "product_code"]
    present = [c for c in subset if c in df.columns]
    if not present:
        return df
    return df.sort_values("marketable_flag" if "marketable_flag" in df.columns else present[0], ascending=False).drop_duplicates(
        subset=present, keep="first"
    )


def build_mrgal_from_bronze(
    bronze_root: Path,
    sample_n: int | None = None,
    merge_reference_flags: Path | None = None,
) -> pd.DataFrame:
    sd = _read_bronze_table(bronze_root, "bronze_sd", sample_n)
    promo = _read_bronze_table(bronze_root, "bronze_stat_promo", sample_n)
    member = _read_bronze_table(bronze_root, "bronze_stat_membership", sample_n)
    demo = _read_bronze_table(bronze_root, "bronze_stat_demo", sample_n)

    keys = MRGAL_KEYS
    mrgal = sd.copy()

    for stat_df, suffix in [(promo, "_promo"), (member, "_mem"), (demo, "_demo")]:
        stat_cols = [c for c in stat_df.columns if c not in keys]
        renamed = {c: f"{c}{suffix}" if c in mrgal.columns else c for c in stat_cols}
        part = stat_df[keys + stat_cols].rename(columns=renamed)
        mrgal = mrgal.merge(part, on=keys, how="left", suffixes=("", "_dup"))

    mrgal = mrgal.loc[:, ~mrgal.columns.str.endswith("_dup")]

    if merge_reference_flags is not None and merge_reference_flags.exists():
        from src.io_utils import read_parquet_dir

        ref = read_parquet_dir(merge_reference_flags, sample_n=sample_n)
        flag_cols = [c for c in ref.columns if c.endswith("_flag") or "news" in c or c == "prior_promo_depth_score"]
        flag_cols = [c for c in flag_cols if c not in keys]
        if flag_cols:
            mrgal = mrgal.merge(ref[keys + flag_cols], on=keys, how="left", suffixes=("", "_ref"))

    mrgal = apply_marketable_universe_rules(mrgal)
    return mrgal
