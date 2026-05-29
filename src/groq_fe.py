"""
GROQ-assisted feature engineering (optional).

Set GROQ_API_KEY via env, config/groq.local.yaml (gitignored), or src/groq_secrets.local.py (gitignored).
Do not paste keys into this file — it is committed to Git.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.config_loader import load_settings
from src.groq_credentials import get_groq_api_key
from src.id_keys import KEY_COLUMNS


def _univariate_stats(df: pd.DataFrame, target: str, max_cols: int = 80) -> list[dict]:
    cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cols = cols[:max_cols]
    y = df[target].values
    rows = []
    for c in cols:
        x = df[c].fillna(0).values
        try:
            auc = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, x))
        except Exception:
            auc = float("nan")
        rows.append(
            {
                "column_name": c,
                "missing_pct": float(df[c].isna().mean()),
                "cardinality": int(df[c].nunique()),
                "univariate_auc": auc,
            }
        )
    return rows


def rule_based_shortlist(stats: list[dict], min_auc: float = 0.52) -> list[dict]:
    decisions = []
    for s in stats:
        keep = (
            s["missing_pct"] < 0.5
            and s["cardinality"] > 1
            and (np.isnan(s["univariate_auc"]) or s["univariate_auc"] >= min_auc or s["univariate_auc"] <= 1 - min_auc)
        )
        decisions.append(
            {
                "column_name": s["column_name"],
                "keep": bool(keep),
                "binning_strategy": "quantile_10" if keep else "drop",
                "reason": "rule_fallback",
            }
        )
    return decisions


def groq_feature_decisions(
    stats: list[dict],
    product_code: str,
    client_id: str,
) -> list[dict]:
    api_key = get_groq_api_key()
    cfg = load_settings()["groq"]
    if os.environ.get("GROQ_DISABLED", "").lower() in ("1", "true", "yes"):
        return rule_based_shortlist(stats)
    if not api_key or not cfg.get("enabled", True):
        return rule_based_shortlist(stats)

    prompt = {
        "role": "insurance marketing ML",
        "task": "For each column, decide keep/drop and binning for logistic response model (<1% responders).",
        "product": product_code,
        "client": client_id,
        "columns": stats,
        "output_schema": {
            "decisions": [
                {
                    "column_name": "str",
                    "keep": "bool",
                    "binning_strategy": "quantile_10|equal_width|none|drop",
                    "reason": "str",
                }
            ]
        },
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{cfg['api_base']}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": cfg["model"],
                    "messages": [
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": json.dumps(prompt)},
                    ],
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content.strip().strip("`").replace("json\n", ""))
            return parsed.get("decisions", rule_based_shortlist(stats))
    except Exception:
        return rule_based_shortlist(stats)


def apply_feature_engineering(
    df: pd.DataFrame,
    decisions: list[dict],
    target: str = "responder_flag",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_cols = [d["column_name"] for d in decisions if d.get("keep")]
    key_cols = [c for c in KEY_COLUMNS + [target] if c in df.columns]
    use = list(dict.fromkeys(key_cols + keep_cols))

    out = df[use].copy()
    report_rows = []
    for d in decisions:
        col = d["column_name"]
        if col not in df.columns or not d.get("keep"):
            continue
        strat = d.get("binning_strategy", "none")
        if strat == "quantile_10" and pd.api.types.is_numeric_dtype(df[col]):
            try:
                out[f"{col}_bin"] = pd.qcut(df[col].rank(method="first"), 10, labels=False, duplicates="drop")
            except Exception:
                out[f"{col}_bin"] = 0
        report_rows.append(d)

    manifest = pd.DataFrame(report_rows)
    return out, manifest
