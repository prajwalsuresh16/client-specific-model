"""
Chunked synthetic data generation for Regional Bank.

Writes parquet partitions under data/regional_bank/bronze/.
Default row_count from settings.yaml (5_000_000); override with FMG_ROW_COUNT.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.config_loader import load_settings, resolve_path
from src.schemas import (
    REGIONAL_BANK_MRGL_COLUMNS,
    REGIONAL_BANK_RESPONDER_COLUMNS,
    REGIONAL_BANK_SD_COLUMNS,
    REGIONAL_BANK_STAT_COLUMNS,
    assert_min_columns,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _bpid_series(n: int, start_index: int, prefix: str) -> np.ndarray:
    return np.array([f"{prefix}_{i:010d}" for i in range(start_index, start_index + n)], dtype=object)


def _base_keys(
    n: int,
    cfg: dict,
    rng: np.random.Generator,
    client_id: str,
    start_index: int = 0,
    *,
    prior: bool = False,
) -> dict:
    camp = cfg["campaign"]
    return {
        "bpid": _bpid_series(n, start_index, client_id[:3]),
        "campaign_id": np.full(n, camp["prior_campaign_id"] if prior else camp["campaign_id"], dtype=object),
        "client_id": np.full(n, client_id, dtype=object),
        "cut_date": np.full(n, camp.get("prior_cut_date", camp["cut_date"]) if prior else camp["cut_date"], dtype=object),
        "product_code": np.full(n, camp["product_code"], dtype=object),
    }


def _fill_sd_block(n: int, rng: np.random.Generator) -> dict:
    age = rng.integers(18, 85, size=n)
    return {
        "eligible_to_market_flag": rng.choice([0, 1], size=n, p=[0.05, 0.95]).astype(np.int8),
        "deceased_suppressed_flag": rng.choice([0, 1], size=n, p=[0.99, 0.01]).astype(np.int8),
        "bad_address_flag": rng.choice([0, 1], size=n, p=[0.97, 0.03]).astype(np.int8),
        "dnc_flag": rng.choice([0, 1], size=n, p=[0.92, 0.08]).astype(np.int8),
        "existing_product_suppressed_flag": rng.choice([0, 1], size=n, p=[0.85, 0.15]).astype(np.int8),
        "household_id": np.array([f"HH_{x}" for x in rng.integers(0, n // 2 + 1, size=n)], dtype=object),
        "state_code": rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=n),
        "zip_code": rng.integers(10000, 99999, size=n),
        "age": age,
        "gender_code": rng.choice(["M", "F", "U"], size=n, p=[0.48, 0.48, 0.04]),
        "income_band": rng.integers(1, 10, size=n),
        "marital_status_code": rng.choice(["S", "M", "D", "W"], size=n),
        "homeowner_flag": rng.choice([0, 1], size=n, p=[0.4, 0.6]).astype(np.int8),
        "insured_flag": rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype(np.int8),
        "max_coverage_reached_flag": rng.choice([0, 1], size=n, p=[0.9, 0.1]).astype(np.int8),
    }


def _fill_infobase(n: int, rng: np.random.Generator) -> dict:
    return {f"infobase_attr_{i:03d}": rng.normal(0, 1, size=n).astype(np.float32) for i in range(1, 16)}


def _fill_stat_block(n: int, rng: np.random.Generator) -> dict:
    from src.schemas import STAT_COMMON

    out: dict = {}
    for col in STAT_COMMON:
        if "flag" in col:
            out[col] = rng.choice([0, 1], size=n).astype(np.int8)
        elif "code" in col or col.endswith("_band"):
            out[col] = rng.integers(1, 8, size=n)
        elif "days" in col or "months" in col or "count" in col:
            out[col] = rng.integers(0, 500, size=n)
        else:
            out[col] = rng.normal(0, 1, size=n).astype(np.float32)
    return out


def _fill_mrgal_flags(n: int, rng: np.random.Generator) -> dict:
    news_p = rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype(np.int8)
    news_d = rng.choice([0, 1], size=n, p=[0.8, 0.2]).astype(np.int8)
    base = np.where((news_p == 0) & (news_d == 0), 1, 0).astype(np.int8)
    return {
        "news_p_flag": news_p,
        "news_d_flag": news_d,
        "base_name_flag": base,
        "prior_promo_depth_score": rng.uniform(0, 1, size=n).astype(np.float32),
    }


def _fill_regional_extras(n: int, rng: np.random.Generator) -> dict:
    return {
        "regional_branch_code": rng.integers(100, 999, size=n),
        "digital_banking_user_flag": rng.choice([0, 1], size=n, p=[0.35, 0.65]).astype(np.int8),
        "cd_balance_band": rng.integers(1, 6, size=n),
        "mortgage_holder_flag": rng.choice([0, 1], size=n).astype(np.int8),
        "auto_loan_flag": rng.choice([0, 1], size=n).astype(np.int8),
    }


def _fill_responders(n: int, rng: np.random.Generator, rate: float) -> dict:
    from src.schemas import RESPONDER_COLS

    flag = rng.choice([0, 1], size=n, p=[1 - rate, rate]).astype(np.int8)
    premium = np.where(flag == 1, rng.uniform(50, 400, size=n), 0.0).astype(np.float32)
    base = {
        "responder_flag": flag,
        "response_date": np.array([("2026-02-15" if f else None) for f in flag], dtype=object),
        "premium_amount": premium,
        "policy_issued_flag": np.where(flag == 1, 1, 0).astype(np.int8),
        "channel_code": rng.choice(["DM", "EM", "TM"], size=n),
        "response_lag_days": np.where(flag == 1, rng.integers(7, 90, size=n), 0),
        "coverage_amount_band": rng.integers(1, 8, size=n),
        "payment_mode_code": rng.choice(["M", "Q", "A"], size=n),
        "cancel_within_30d_flag": rng.choice([0, 1], size=n, p=[0.95, 0.05]).astype(np.int8),
        "claim_filed_flag": rng.choice([0, 1], size=n, p=[0.99, 0.01]).astype(np.int8),
        "multi_product_responder_flag": rng.choice([0, 1], size=n, p=[0.9, 0.1]).astype(np.int8),
        "campaign_touch_index": rng.integers(1, 6, size=n),
        "mail_piece_version": rng.choice(["A", "B", "C"], size=n),
        "creative_test_cell": rng.integers(1, 5, size=n),
        "underwriting_decision_code": rng.choice(["A", "D", "R"], size=n),
        "agent_channel_flag": rng.choice([0, 1], size=n).astype(np.int8),
        "digital_response_flag": rng.choice([0, 1], size=n).astype(np.int8),
        "inbound_call_flag": rng.choice([0, 1], size=n, p=[0.92, 0.08]).astype(np.int8),
        "apps_started_count": rng.integers(0, 3, size=n),
        "apps_completed_count": np.where(flag == 1, 1, 0).astype(np.int8),
        "household_responder_flag": rng.choice([0, 1], size=n).astype(np.int8),
        "ltv_score_at_response": rng.uniform(0, 1, size=n).astype(np.float32),
        "risk_score_at_response": rng.uniform(0, 1, size=n).astype(np.float32),
        "discount_applied_flag": rng.choice([0, 1], size=n).astype(np.int8),
        "renewal_probability_score": rng.uniform(0, 1, size=n).astype(np.float32),
        "cross_sell_eligible_flag": rng.choice([0, 1], size=n).astype(np.int8),
        "actual_pppm": np.where(flag == 1, premium / 12.0, 0.0).astype(np.float32),
        "actual_response_decile": rng.integers(1, 11, size=n),
        "sales_book_actualization_pct": np.where(flag == 1, rng.uniform(0.85, 1.0, size=n), 0.0).astype(np.float32),
        "maturity_days": np.where(flag == 1, rng.integers(60, 120, size=n), 0),
    }
    return {k: base[k] for k in RESPONDER_COLS}


def _chunk_ranges(total: int, chunk: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def generate_regional_bank(cfg: dict | None = None, *, include_prior: bool | None = None) -> dict[str, Path]:
    cfg = cfg or load_settings()
    n = int(cfg["synthetic"]["row_count"])
    paths = _generate_client("REGIONAL_BANK", n, cfg, bronze_subdir="bronze")
    if include_prior is None:
        include_prior = bool(cfg.get("synthetic", {}).get("generate_prior_campaign", True))
    if include_prior:
        prior_paths = _generate_client("REGIONAL_BANK", n, cfg, bronze_subdir="bronze_prior", prior=True)
        paths.update({f"prior_{k}": v for k, v in prior_paths.items()})
    return paths


def _generate_client(
    client_id: str,
    row_count: int,
    cfg: dict,
    *,
    bronze_subdir: str = "bronze",
    prior: bool = False,
) -> dict[str, Path]:
    chunk = int(cfg["synthetic"]["chunk_size"])
    seed = int(cfg["synthetic"]["random_seed"])
    rate = float(
        cfg["synthetic"].get("prior_responder_positive_rate", cfg["synthetic"]["responder_positive_rate"])
        if prior
        else cfg["synthetic"]["responder_positive_rate"]
    )

    sd_cols = REGIONAL_BANK_SD_COLUMNS
    stat_cols = REGIONAL_BANK_STAT_COLUMNS
    mrgal_cols = REGIONAL_BANK_MRGL_COLUMNS
    resp_cols = REGIONAL_BANK_RESPONDER_COLUMNS

    for label, cols in [("sd", sd_cols), ("stat", stat_cols), ("mrgal", mrgal_cols), ("responders", resp_cols)]:
        assert_min_columns(cols, 30, label)

    root = resolve_path("data", "regional_bank", bronze_subdir)
    paths: dict[str, Path] = {}

    def _chunk_builder(builder, cols, out_name: str, base_seed: int):
        out_dir = root / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, (start, end) in enumerate(_chunk_ranges(row_count, chunk)):
            df = builder(end - start, base_seed + idx, start)[cols]
            df.to_parquet(out_dir / f"part_{idx:05d}.parquet", index=False)
        return out_dir

    def build_sd(n: int, s: int, start: int = 0) -> pd.DataFrame:
        rng = _rng(s)
        data = _base_keys(n, cfg, rng, client_id, start_index=start, prior=prior)
        data.update(_fill_sd_block(n, rng))
        data.update(_fill_infobase(n, rng))
        data.update(_fill_regional_extras(n, rng))
        return pd.DataFrame(data)

    paths["sd"] = _chunk_builder(build_sd, sd_cols, "bronze_sd", seed)

    for stat_name in ("bronze_stat_promo", "bronze_stat_membership", "bronze_stat_demo"):
        sn = stat_name

        def build_stat(n: int, s: int, start: int = 0, _sn: str = sn) -> pd.DataFrame:
            rng = _rng(s + hash(_sn) % 1000)
            data = _base_keys(n, cfg, rng, client_id, start_index=start, prior=prior)
            data.update(_fill_stat_block(n, rng))
            return pd.DataFrame(data)

        paths[stat_name] = _chunk_builder(build_stat, stat_cols, stat_name, seed + 1)

    def build_mrgal(n: int, s: int, start: int = 0) -> pd.DataFrame:
        rng = _rng(s)
        data = _base_keys(n, cfg, rng, client_id, start_index=start, prior=prior)
        data.update(_fill_sd_block(n, rng))
        data.update(_fill_stat_block(n, rng))
        data.update(_fill_infobase(n, rng))
        data.update(_fill_mrgal_flags(n, rng))
        data.update(_fill_regional_extras(n, rng))
        return pd.DataFrame(data)

    paths["mrgal"] = _chunk_builder(build_mrgal, mrgal_cols, "bronze_mrgal", seed + 2)

    def build_resp(n: int, s: int, start: int = 0) -> pd.DataFrame:
        rng = _rng(s)
        data = _base_keys(n, cfg, rng, client_id, start_index=start, prior=prior)
        data.update(_fill_responders(n, rng, rate))
        return pd.DataFrame(data)

    paths["responders"] = _chunk_builder(build_resp, resp_cols, "bronze_responders", seed + 3)

    manifest = {
        "client_id": client_id,
        "row_count": row_count,
        "bronze_subdir": bronze_subdir,
        "prior_campaign": prior,
        "campaign_id": cfg["campaign"]["prior_campaign_id"] if prior else cfg["campaign"]["campaign_id"],
        "files": {k: str(v) for k, v in paths.items()},
        "column_counts": {
            "sd": len(sd_cols),
            "stat": len(stat_cols),
            "mrgal": len(mrgal_cols),
            "responders": len(resp_cols),
        },
        "sd_columns": sd_cols,
    }
    manifest_name = "schema_manifest_prior.json" if prior else "schema_manifest.json"
    (root / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
