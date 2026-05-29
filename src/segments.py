"""Segment dimensions for LOL, rank-mix QC, and Step 8 selection (legacy-aligned)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config_loader import resolve_path

from src.id_keys import KEY_COLUMNS


def load_segment_rules() -> dict[str, Any]:
    path = resolve_path("config", "segment_rules.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_age_band(df: pd.DataFrame, rules: dict | None = None) -> pd.Series:
    if "age" not in df.columns:
        return pd.Series("ALL", index=df.index, dtype=object)
    rules = rules or load_segment_rules()
    bands = rules.get("segments", {}).get("age_bands", [])
    if not bands:
        return pd.cut(df["age"], bins=[0, 35, 51, 120], labels=["under_35", "age_35_50", "over_50"], right=True)
    edges = [b["min"] for b in bands] + [bands[-1]["max"] + 1]
    labels = [b["name"] for b in bands]
    return pd.cut(df["age"], bins=edges, labels=labels, right=True)


def assign_account_tenure_bucket(df: pd.DataFrame) -> pd.Series:
    col = "account_open_days"
    if col not in df.columns:
        return pd.Series("tenure_unknown", index=df.index, dtype=object)
    days = df[col].fillna(-1)
    return pd.cut(
        days,
        bins=[-1, 365, 1095, 10_000],
        labels=["tenure_0_12m", "tenure_13_36m", "tenure_37m_plus"],
    ).astype(str)


def assign_news_segment(df: pd.DataFrame) -> pd.Series:
    if "news_p_flag" not in df.columns and "news_d_flag" not in df.columns:
        return pd.Series("news_none", index=df.index, dtype=object)

    p = df.get("news_p_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)
    d = df.get("news_d_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)
    out = np.where((p == 1) & (d == 1), "news_pd", np.where(p == 1, "news_p", np.where(d == 1, "news_d", "news_none")))
    return pd.Series(out, index=df.index, dtype=object)


def build_segment_id(df: pd.DataFrame, rules: dict | None = None) -> pd.Series:
    """Composite segment key for LOL / selection (client-configurable dimensions)."""
    rules = rules or load_segment_rules()
    dims = rules.get("lol_dimensions", ["age_band", "news_segment", "account_tenure_bucket"])
    work = df.copy()
    if "age_band" in dims:
        work["age_band"] = assign_age_band(work, rules)
    if "news_segment" in dims:
        work["news_segment"] = assign_news_segment(work)
    if "account_tenure_bucket" in dims:
        work["account_tenure_bucket"] = assign_account_tenure_bucket(work)

    parts = []
    for dim in dims:
        if dim in work.columns:
            parts.append(work[dim].astype(str))
    if not parts:
        return pd.Series("ALL", index=df.index, dtype=object)
    seg = parts[0]
    for p in parts[1:]:
        seg = seg + "|" + p
    return seg


def enrich_segment_columns(df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    rules = rules or load_segment_rules()
    out = df.copy()
    out["age_band"] = assign_age_band(out, rules)
    out["news_segment"] = assign_news_segment(out)
    out["account_tenure_bucket"] = assign_account_tenure_bucket(out)
    out["segment_id"] = build_segment_id(out, rules)
    return out


def build_list_key(segment_id: str, decile: int | float) -> str:
    d = int(decile) if pd.notna(decile) else -1
    safe = str(segment_id).replace(" ", "_").replace("|", "_")
    return f"LK_{safe}_D{d}"
