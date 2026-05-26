"""Step 6 — Dynamic rank-mix scoring: probability → rank → average rank."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config_loader import load_settings, resolve_path
from src.dataset_io import read_dataset, write_dataset
from src.metrics import rank_mix_score
from src.segments import enrich_segment_columns


def run(sample_n: int | None = None) -> Path:
    load_settings()

    rec = read_dataset("gold", "gold_auto_recommendation").iloc[0]
    scoring = read_dataset("gold", "gold_scoring_population", sample_n=sample_n)
    preds = read_dataset("gold", "gold_model_predictions", sample_n=sample_n)

    df = scoring.merge(preds, on="bpid", how="left")
    df = enrich_segment_columns(df)

    m1 = rec["recommended_model_a"]
    m2 = rec["recommended_model_b"]
    if m1 not in df.columns or m2 not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith("prob_") and c != "prob_default"]
        m1, m2 = prob_cols[0], prob_cols[1]

    default = df["prob_default"] if "prob_default" in df.columns else 0
    s1 = df[m1].fillna(default)
    s2 = df[m2].fillna(default)
    df["rank_m1"] = s1.rank(ascending=False, method="average")
    df["rank_m2"] = s2.rank(ascending=False, method="average")
    df["rank_mix_score"] = rank_mix_score(s1.values, s2.values)

    out = write_dataset(df, "gold", "gold_scored_records")

    qc = (
        df.groupby("segment_id", observed=True)
        .agg(n=("bpid", "count"), mean_rank_mix=("rank_mix_score", "mean"))
        .reset_index()
    )
    write_dataset(qc, "gold", "gold_qc_segment_report")
    return out
