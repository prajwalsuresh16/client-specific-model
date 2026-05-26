"""Step 7 — List-of-lists: composite segment_id × decile grid (legacy list keys)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import resolve_path
from src.dataset_io import read_dataset, write_dataset
from src.segments import build_list_key, enrich_segment_columns


def run() -> Path:
    scored = read_dataset("gold", "gold_scored_records")
    scored = enrich_segment_columns(scored)

    scored["decile"] = pd.qcut(
        scored["rank_mix_score"].rank(method="first"),
        10,
        labels=list(range(10)),
        duplicates="drop",
    )

    scored["list_key"] = scored.apply(
        lambda r: build_list_key(str(r["segment_id"]), r["decile"]),
        axis=1,
    )

    agg: dict = {"name_count": ("bpid", "count"), "mean_rank_mix": ("rank_mix_score", "mean")}
    if "news_p_flag" in scored.columns:
        agg["news_p_count"] = ("news_p_flag", "sum")
    lol = scored.groupby(["segment_id", "decile", "list_key"], observed=True).agg(**agg).reset_index()

    write_dataset(
        scored[["segment_id", "decile", "list_key", "age_band", "news_segment", "account_tenure_bucket"]].drop_duplicates(),
        "gold",
        "gold_lol_segment_catalog",
    )
    return write_dataset(lol, "gold", "gold_list_of_lists")
