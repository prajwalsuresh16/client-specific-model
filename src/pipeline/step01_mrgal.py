"""Step 1 — Auto MRGAL: SD + STAT → wide merge-all; current + prior (Delta on Databricks)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import load_settings, resolve_path
from src.dataset_io import read_dataset, write_dataset
from src.mrgal_builder import build_mrgal_from_bronze, dedupe_mrgal


def _finalize_mrgal(df: pd.DataFrame, cfg: dict, label: str) -> pd.DataFrame:
    dedupe_keys = cfg.get("mrgal", {}).get("dedupe_keys", ["bpid", "campaign_id", "client_id", "product_code"])
    out = dedupe_mrgal(df, subset=dedupe_keys)
    if cfg.get("mrgal", {}).get("filter_to_marketable_only", False):
        out = out[out["marketable_flag"] == 1].copy()
    out["mrgal_source"] = label
    return out


def _build_and_write(
    bronze_root: Path,
    out_name: str,
    cfg: dict,
    sample_n: int | None,
    flag_ref: Path | None,
    *,
    prior: bool = False,
) -> Path:
    mrgal = build_mrgal_from_bronze(bronze_root, sample_n=sample_n, merge_reference_flags=flag_ref)
    if prior:
        for col in ("campaign_id", "cut_date"):
            if col in mrgal.columns and col == "campaign_id":
                mrgal[col] = cfg["campaign"]["prior_campaign_id"]
            if col == "cut_date":
                mrgal[col] = cfg["campaign"].get("prior_cut_date", cfg["campaign"]["cut_date"])
    mrgal = _finalize_mrgal(mrgal, cfg, label=out_name)
    return write_dataset(mrgal, "silver", out_name)


def run(sample_n: int | None = None) -> dict[str, Path]:
    cfg = load_settings()
    bronze = resolve_path("data", "regional_bank", "bronze")
    bronze_prior = resolve_path("data", "regional_bank", "bronze_prior")
    flag_ref = bronze / "bronze_mrgal"

    paths: dict[str, Path] = {}
    paths["silver_mrgal"] = _build_and_write(
        bronze,
        "silver_mrgal",
        cfg,
        sample_n,
        flag_ref if flag_ref.exists() or list(flag_ref.glob("*.parquet")) else None,
    )

    if bronze_prior.exists() and any(bronze_prior.glob("**/*.parquet")):
        prior_ref = bronze_prior / "bronze_mrgal"
        paths["silver_mrgal_prior"] = _build_and_write(
            bronze_prior,
            "silver_mrgal_prior",
            cfg,
            sample_n,
            prior_ref if prior_ref.exists() or list(prior_ref.glob("*.parquet")) else None,
            prior=True,
        )
    else:
        prior = read_dataset("silver", "silver_mrgal", sample_n=sample_n).copy()
        prior["campaign_id"] = cfg["campaign"]["prior_campaign_id"]
        prior["cut_date"] = cfg["campaign"].get("prior_cut_date", cfg["campaign"]["cut_date"])
        prior["mrgal_source"] = "silver_mrgal_prior_fallback"
        paths["silver_mrgal_prior"] = write_dataset(prior, "silver", "silver_mrgal_prior")

    qc = pd.DataFrame(
        [
            {
                "table": k,
                "row_count": len(read_dataset("silver", k, sample_n=10_000)),
                "column_count": len(read_dataset("silver", k, sample_n=1_000).columns),
            }
            for k in paths
        ]
    )
    write_dataset(qc, "silver", "silver_mrgal_qc")
    return paths
