"""Step 8+9 — LOL-guided selection, PPPM trim, keep_flag, SO file (100% SD parity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config_loader import load_settings, resolve_path
from src.dataset_io import read_bronze, read_dataset, write_dataset
from src.io_utils import ensure_dir
from src.metrics import expected_pppm_score
from src.segments import build_list_key, enrich_segment_columns
from src.id_keys import KEY_COLUMNS


def _load_selection_rules() -> dict:
    path = resolve_path("config", "selection_rules.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["selection"]


def _max_decile_for_segment(seg: str, rules: dict) -> int:
    defaults = rules.get("default_max_decile_by_segment", {})
    # Match legacy age_band keys when segment_id is composite
    for key in ("under_35", "age_35_50", "over_50"):
        if key in str(seg):
            return int(defaults.get(key, defaults.get("ALL", 4)))
    return int(defaults.get("ALL", 4))


def _apply_manual_overrides(rules: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if rules.get("mode") != "manual_csv":
        return out
    path = resolve_path(rules["manual_overrides_path"])
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        out[str(row["segment_id"])] = {
            "max_decile": int(row["max_decile"]),
            "top_pct_within_segment": float(row.get("top_pct_within_segment", rules["top_pct_within_segment"])),
        }
    return out


def _select_with_row_deciles(grp: pd.DataFrame, max_decile: pd.Series, top_pct: float) -> pd.Series:
    eligible = grp[grp["decile"] <= max_decile]
    if eligible.empty:
        eligible = grp
    n_keep = max(1, int(len(eligible) * top_pct))
    chosen = eligible.nsmallest(n_keep, "rank_mix_score")["bpid"]
    flags = pd.Series("N", index=grp.index)
    flags.loc[grp["bpid"].isin(chosen)] = "Y"
    return flags


def _apply_pppm_trim(scored: pd.DataFrame, rules: dict) -> pd.DataFrame:
    trim = rules.get("pppm_trim", {})
    if not trim.get("enabled", False):
        return scored

    target = float(trim.get("target_pppm", 1.25))
    min_pct = float(trim.get("min_keep_pct", 0.15))
    y = scored.get("responder_flag", pd.Series(0, index=scored.index)).fillna(0).astype(int).values
    prem = scored.get("premium_amount", pd.Series(0.0, index=scored.index)).fillna(0).values
    scores = scored["rank_mix_score"].values

    keep_mask = scored["keep_flag"] == "Y"
    if keep_mask.sum() == 0:
        return scored

    ordered = scored.loc[keep_mask].sort_values("rank_mix_score")
    min_keep = max(1, int(len(scored) * min_pct))
    kept_idx = list(ordered.index)

    while len(kept_idx) > min_keep:
        trial = scored.loc[kept_idx]
        pppm = expected_pppm_score(
            trial.get("responder_flag", pd.Series(0)).fillna(0).astype(int).values,
            trial.get("premium_amount", pd.Series(0.0)).fillna(0).values,
            trial["rank_mix_score"].values,
        )
        if np.isnan(pppm) or pppm >= target:
            break
        kept_idx.pop()

    scored["keep_flag"] = "N"
    scored.loc[kept_idx, "keep_flag"] = "Y"
    return scored


def run() -> Path:
    cfg = load_settings()
    rules = _load_selection_rules()
    overrides = _apply_manual_overrides(rules)
    sd = read_bronze("bronze_sd")
    scored = read_dataset("gold", "gold_scored_records")
    scored = enrich_segment_columns(scored)

    seg_col = "segment_id"
    scored["decile"] = pd.qcut(
        scored["rank_mix_score"].rank(method="first"), 10, labels=False, duplicates="drop"
    ).astype("Int64")

    boost = rules.get("boost_news_names", {})
    boost_on = bool(boost.get("enabled", False))
    extra_p = int(boost.get("news_p_extra_deciles", 1))
    extra_d = int(boost.get("news_d_extra_deciles", 1))

    scored["keep_flag"] = "N"

    for seg, grp in scored.groupby(seg_col, observed=True):
        seg_key = str(seg)
        seg_rules = overrides.get(seg_key, {})
        max_dec_base = int(seg_rules.get("max_decile", _max_decile_for_segment(seg_key, rules)))
        top_pct = float(seg_rules.get("top_pct_within_segment", rules["top_pct_within_segment"]))

        max_dec_row = pd.Series(max_dec_base, index=grp.index, dtype=int)
        if boost_on and "news_p_flag" in grp.columns:
            max_dec_row.loc[grp["news_p_flag"] == 1] = np.minimum(9, max_dec_base + extra_p)
        if boost_on and "news_d_flag" in grp.columns:
            max_dec_row.loc[grp["news_d_flag"] == 1] = np.maximum(
                max_dec_row.loc[grp["news_d_flag"] == 1],
                min(9, max_dec_base + extra_d),
            )

        scored.loc[grp.index, "keep_flag"] = _select_with_row_deciles(grp, max_dec_row, top_pct)

    cap = rules.get("global_mail_quantity_cap")
    if cap is not None and (scored["keep_flag"] == "Y").sum() > int(cap):
        y_rows = scored[scored["keep_flag"] == "Y"].sort_values("rank_mix_score")
        keep_bpids = set(y_rows.head(int(cap))["bpid"])
        scored["keep_flag"] = scored["bpid"].isin(keep_bpids).map({True: "Y", False: "N"})

    scored = _apply_pppm_trim(scored, rules)

    scored["list_key"] = scored.apply(
        lambda r: build_list_key(str(r[seg_col]), r["decile"]),
        axis=1,
    )
    keys = list(KEY_COLUMNS)
    so = sd[keys].merge(
        scored[["bpid", "indiv_id", "rank_mix_score", "keep_flag", "list_key", "decile", "segment_id"]],
        on="bpid",
        how="left",
    )
    so["keep_flag"] = so["keep_flag"].fillna("N")

    if len(so) != len(sd):
        raise ValueError(f"SO/SD row mismatch: SO={len(so)} SD={len(sd)} (must be 100%)")

    write_dataset(
        scored.groupby([seg_col, "decile"], observed=True)["keep_flag"].value_counts().reset_index(name="count"),
        "gold",
        "gold_selection_summary",
    )

    ensure_dir(resolve_path("data", "regional_bank", "bronze", "bronze_so_output"))
    write_dataset(so, "bronze", "bronze_so")
    return write_dataset(so, "gold", "gold_so_output")
