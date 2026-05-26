"""Step 5 — Auto recommendations with AUC/PPPM tradeoff and optional prior-campaign stability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_settings
from src.dataset_io import read_dataset, write_dataset


def _normalize(series: pd.Series) -> pd.Series:
    if series.isna().all():
        return series.fillna(0)
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def run() -> Path:
    cfg = load_settings()
    val_cfg = cfg.get("validation", {})

    results = read_dataset("gold", "gold_validation_results")
    overall = results[results["segment_id"] == "ALL"].copy()
    if overall.empty:
        raise RuntimeError("No validation results; run step04 first.")
    overall["auc"] = overall["auc"].fillna(0.5)
    overall["pppm_score"] = overall.get("pppm_score", pd.Series(0, index=overall.index)).fillna(0)

    w_auc = 0.7
    w_pppm = 0.3
    overall["pppm_corr"] = overall["pppm_corr"].fillna(0)
    overall["pppm_score"] = overall.get("pppm_score", pd.Series(np.nan, index=overall.index)).fillna(0)

    overall["score"] = (
        w_auc * _normalize(overall["auc"])
        + w_pppm * (_normalize(overall["pppm_score"]) * 0.5 + overall["pppm_corr"].clip(-1, 1) * 0.5)
    )

    # Stability: prefer combos that also scored well on prior validation artifact if present
    stability_w = float(val_cfg.get("stability_prior_weight", 0.15))
    if stability_w > 0:
        try:
            prior = read_dataset("gold", "gold_validation_results_prior")
        except FileNotFoundError:
            prior = None
        if prior is not None:
            prior_all = prior[prior["segment_id"] == "ALL"][["model_a", "model_b", "auc"]].rename(
                columns={"auc": "prior_auc"}
            )
            overall = overall.merge(prior_all, on=["model_a", "model_b"], how="left")
            overall["score"] = overall["score"] + stability_w * _normalize(overall["prior_auc"].fillna(overall["auc"]))

    best = overall.nlargest(1, "score").iloc[0]

    rec = pd.DataFrame(
        [
            {
                "campaign_id": cfg["campaign"]["campaign_id"],
                "prior_campaign_id": cfg["campaign"]["prior_campaign_id"],
                "client_id": cfg["client"]["id"],
                "product_code": cfg["campaign"]["product_code"],
                "recommended_model_a": best["model_a"],
                "recommended_model_b": best["model_b"],
                "rank_mix_pair": json.dumps([best["model_a"], best["model_b"]]),
                "expected_auc": best["auc"],
                "expected_pppm_score": best.get("pppm_score"),
                "expected_pppm_corr": best["pppm_corr"],
                "rationale_json": json.dumps(
                    {
                        "rule": "max_weighted_auc_pppm_with_stability",
                        "weights": {"auc": w_auc, "pppm": w_pppm, "stability": stability_w},
                        "combo_id": best["combo_id"],
                        "validation_mode": best.get("validation_mode"),
                    }
                ),
            }
        ]
    )

    # Archive current validation as prior for next campaign run
    archive = results.copy()
    archive["archived_for_campaign"] = cfg["campaign"]["prior_campaign_id"]
    write_dataset(archive, "gold", "gold_validation_results_prior")

    return write_dataset(rec, "gold", "gold_auto_recommendation")
