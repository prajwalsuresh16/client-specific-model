"""Step 2 — Prior labeled train/val; current scoring (Unity Catalog gold tables on Databricks)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_settings, resolve_path
from src.dataset_io import read_bronze_prior, read_dataset, write_dataset
from src.id_keys import KEY_COLUMNS


def _resample_balanced(df: pd.DataFrame, target: str, max_rows: int, seed: int) -> pd.DataFrame:
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df.head(max_rows)
    n_pos = min(len(pos), max_rows // 2)
    n_neg = min(len(neg), max_rows - n_pos)
    rng = np.random.default_rng(seed)
    pos_s = pos.sample(n=n_pos, random_state=int(rng.integers(0, 2**31)))
    neg_s = neg.sample(n=n_neg, random_state=int(rng.integers(0, 2**31)))
    return pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1, random_state=seed)


def _attach_responders(mrgal: pd.DataFrame, responders: pd.DataFrame) -> pd.DataFrame:
    keys = list(KEY_COLUMNS)
    rcols = keys + ["responder_flag", "premium_amount"]
    labeled = mrgal.merge(responders[rcols].drop_duplicates(subset=keys), on=keys, how="left")
    if labeled["responder_flag"].notna().sum() < 10:
        by_bpid = responders.groupby("bpid", as_index=False).agg(
            responder_flag=("responder_flag", "max"),
            premium_amount=("premium_amount", "mean"),
        )
        labeled = mrgal.drop(columns=[c for c in ("responder_flag", "premium_amount") if c in mrgal.columns]).merge(
            by_bpid, on="bpid", how="left"
        )
    labeled["responder_flag"] = labeled["responder_flag"].fillna(0).astype(int)
    return labeled


def run(sample_n: int | None = None) -> dict[str, Path]:
    cfg = load_settings()

    try:
        prior = read_dataset("silver", "silver_mrgal_prior", sample_n=sample_n)
    except FileNotFoundError:
        prior = read_dataset("silver", "silver_mrgal", sample_n=sample_n)
        prior["campaign_id"] = cfg["campaign"]["prior_campaign_id"]

    current = read_dataset("silver", "silver_mrgal", sample_n=sample_n)

    try:
        responders = read_bronze_prior("bronze_responders", sample_n=sample_n)
    except FileNotFoundError:
        from src.dataset_io import read_bronze

        responders = read_bronze("bronze_responders", sample_n=sample_n)

    labeled_prior = _attach_responders(prior, responders)
    allow_synth = bool(cfg.get("sample_prep", {}).get("allow_synthetic_responders", False)) or os.environ.get(
        "FMG_ALLOW_SYNTHETIC_RESPONDERS", ""
    ).lower() in ("1", "true", "yes")

    min_resp = int(cfg.get("sample_prep", {}).get("min_responders_for_training", 50))
    if labeled_prior["responder_flag"].sum() < min_resp:
        if not allow_synth:
            raise ValueError(
                f"Only {labeled_prior['responder_flag'].sum()} responders in prior sample "
                f"(need {min_resp}). Generate bronze_prior or set FMG_ALLOW_SYNTHETIC_RESPONDERS=true."
            )
        rng = np.random.default_rng(cfg["synthetic"]["random_seed"])
        n_pos = max(min_resp, int(len(labeled_prior) * float(cfg["synthetic"].get("prior_responder_positive_rate", 0.012))))
        pos_idx = rng.choice(labeled_prior.index, size=min(n_pos, len(labeled_prior)), replace=False)
        labeled_prior.loc[pos_idx, "responder_flag"] = 1
        if "premium_amount" in labeled_prior.columns:
            labeled_prior.loc[pos_idx, "premium_amount"] = rng.uniform(50, 400, size=len(pos_idx))

    scoring = current.copy()
    scoring["responder_flag"] = np.nan

    max_train = sample_n or int(cfg.get("sample_prep", {}).get("max_training_rows", 500_000))
    max_train = min(max_train, len(labeled_prior))
    train_pool = _resample_balanced(labeled_prior, "responder_flag", max_train, cfg["synthetic"]["random_seed"])
    if train_pool["responder_flag"].nunique() < 2:
        raise ValueError("Prior training sample must contain both responders and non-responders.")

    holdout = float(cfg["modeling"]["holdout_fraction"])
    try:
        from sklearn.model_selection import train_test_split

        training, validation = train_test_split(
            train_pool,
            test_size=holdout,
            random_state=42,
            stratify=train_pool["responder_flag"],
        )
    except ValueError:
        train_pool = train_pool.sample(frac=1, random_state=42)
        cut = int(len(train_pool) * (1 - holdout))
        training = train_pool.iloc[:cut]
        validation = train_pool.iloc[cut:]

    meta = pd.DataFrame(
        [
            {
                "prior_campaign_id": cfg["campaign"]["prior_campaign_id"],
                "current_campaign_id": cfg["campaign"]["campaign_id"],
                "prior_rows": len(prior),
                "current_rows": len(current),
                "train_rows": len(training),
                "validation_rows": len(validation),
                "scoring_rows": len(scoring),
                "prior_responders": int(labeled_prior["responder_flag"].sum()),
            }
        ]
    )

    return {
        "gold_training_sample": write_dataset(training, "gold", "gold_training_sample"),
        "gold_validation_sample": write_dataset(validation, "gold", "gold_validation_sample"),
        "gold_scoring_population": write_dataset(scoring, "gold", "gold_scoring_population"),
        "gold_sample_prep_meta": write_dataset(meta, "gold", "gold_sample_prep_meta"),
    }
