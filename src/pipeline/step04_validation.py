"""Step 4 — Auto validation: rank-based combos, AUC, PPPM, KS, orthogonality, per-segment."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_settings, resolve_path
from src.dataset_io import read_dataset, write_dataset
from src.metrics import (
    decile_lift,
    expected_pppm_score,
    ks_statistic,
    passes_orthogonality,
    pppm_rank_correlation,
    rank_mix_score,
    safe_auc,
)
from src.segments import enrich_segment_columns


def _prob_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("prob_") and c != "prob_default")


def _combo_score(df: pd.DataFrame, col_a: str, col_b: str, use_rank: bool) -> np.ndarray:
    a = df[col_a].fillna(df.get("prob_default", 0)).values
    b = df[col_b].fillna(df.get("prob_default", 0)).values
    if use_rank:
        return rank_mix_score(a, b)
    return (a + b) / 2.0


def _evaluate_combo(
    y: np.ndarray,
    premium: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "auc": safe_auc(y, scores),
        "pppm_score": expected_pppm_score(y, premium, scores),
        "pppm_corr": pppm_rank_correlation(y, premium, scores),
        "ks": ks_statistic(y, scores),
    }


def run(sample_n: int | None = 50000) -> Path:
    cfg = load_settings()
    val_cfg = cfg.get("validation", {})

    val = read_dataset("gold", "gold_validation_sample", sample_n=sample_n)
    preds = read_dataset("gold", "gold_model_predictions", sample_n=sample_n)

    df = val.merge(preds, on="bpid", how="inner", suffixes=("", "_pred"))
    df = enrich_segment_columns(df)

    y = df["responder_flag"].astype(int).values
    premium = df["premium_amount"].fillna(0).values if "premium_amount" in df.columns else np.zeros(len(df))
    prob_cols = _prob_cols(df)
    if len(prob_cols) < 2:
        raise RuntimeError("Need at least two prob_* columns from step03.")

    use_rank = bool(val_cfg.get("use_rank_for_combo_validation", True))
    max_corr = float(val_cfg.get("orthogonality_max_correlation", 0.85))
    max_combos = int(cfg["modeling"]["max_model_combinations"])
    seg_cap = int(val_cfg.get("segment_combo_cap", 200))

    pairs = []
    for a, b in itertools.combinations(prob_cols, 2):
        if passes_orthogonality(df[a].values, df[b].values, max_corr):
            pairs.append((a, b))
        if len(pairs) >= max_combos:
            break

    if not pairs:
        pairs = list(itertools.combinations(prob_cols, 2))[:max_combos]

    rows = []
    for i, (a, b) in enumerate(pairs):
        cs = _combo_score(df, a, b, use_rank)
        metrics = _evaluate_combo(y, premium, cs)
        rows.append(
            {
                "combo_id": f"combo_{i:04d}",
                "model_a": a,
                "model_b": b,
                "segment_id": "ALL",
                "validation_mode": "rank_mix" if use_rank else "avg_probability",
                **metrics,
            }
        )

    seg_col = "segment_id"
    for seg, grp in df.groupby(seg_col, observed=True):
        ys = grp["responder_flag"].astype(int).values
        prem = grp["premium_amount"].fillna(0).values if "premium_amount" in grp.columns else np.zeros(len(grp))
        for i, (a, b) in enumerate(pairs[: min(seg_cap, len(pairs))]):
            cs = _combo_score(grp, a, b, use_rank)
            metrics = _evaluate_combo(ys, prem, cs)
            rows.append(
                {
                    "combo_id": f"combo_{i:04d}_{seg}",
                    "model_a": a,
                    "model_b": b,
                    "segment_id": str(seg),
                    "validation_mode": "rank_mix" if use_rank else "avg_probability",
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    out = write_dataset(results, "gold", "gold_validation_results")

    if len(results) > 0:
        best = results.loc[results["segment_id"] == "ALL"].nlargest(1, "auc")
        if not best.empty:
            row = best.iloc[0]
            lift = decile_lift(y, _combo_score(df, row["model_a"], row["model_b"], use_rank))
            write_dataset(lift, "gold", "gold_gain_table")

    return out
