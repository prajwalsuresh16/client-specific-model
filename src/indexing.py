"""
Legacy-style indexing: binning, WOE encoding, IV, exportable index reports.

Fits on training data; applies stored bin edges on scoring data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.id_keys import KEY_COLUMNS


def _woe_iv(y: np.ndarray, x_binned: np.ndarray) -> tuple[float, dict[int, float]]:
    """Information Value and per-bin WOE for binary target."""
    eps = 1e-6
    woe_map: dict[int, float] = {}
    total_good = max((y == 1).sum(), eps)
    total_bad = max((y == 0).sum(), eps)
    iv = 0.0
    for b in np.unique(x_binned):
        mask = x_binned == b
        if not mask.any():
            continue
        good = max((y[mask] == 1).sum(), eps)
        bad = max((y[mask] == 0).sum(), eps)
        dist_good = good / total_good
        dist_bad = bad / total_bad
        woe = float(np.log(dist_good / dist_bad))
        woe_map[int(b)] = woe
        iv += (dist_good - dist_bad) * woe
    return float(iv), woe_map


def _fit_bins(series: pd.Series, n_bins: int, strategy: str) -> np.ndarray:
    s = series.fillna(series.median() if series.notna().any() else 0)
    if strategy == "equal_width":
        return np.linspace(s.min(), s.max() + 1e-9, n_bins + 1)
    ranked = s.rank(method="first")
    try:
        _, edges = pd.qcut(ranked, q=n_bins, retbins=True, duplicates="drop")
        return np.unique(edges)
    except ValueError:
        return np.linspace(s.min(), s.max() + 1e-9, min(n_bins, int(s.nunique())) + 1)


def _apply_bins(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    s = series.fillna(series.median() if series.notna().any() else 0)
    return np.digitize(s, edges[1:-1], right=True).astype(int)


def fit_and_apply_indexing(
    train_df: pd.DataFrame,
    decisions: list[dict],
    target: str = "responder_flag",
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Returns (indexed_train, index_report_df, index_artifacts for scoring).
    """
    if target not in train_df.columns:
        raise ValueError(f"target {target} missing for indexing fit")

    y = train_df[target].fillna(0).astype(int).values
    key_cols = [c for c in KEY_COLUMNS + [target] if c in train_df.columns]
    out = train_df[key_cols].copy()
    report_rows: list[dict] = []
    artifacts: list[dict] = []

    for d in decisions:
        col = d["column_name"]
        if not d.get("keep") or col not in train_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue

        strat = d.get("binning_strategy", "quantile_10")
        if strat == "drop":
            continue
        nb = n_bins if strat in ("quantile_10", "equal_width") else min(5, n_bins)
        bin_strategy = "equal_width" if strat == "equal_width" else "quantile"

        edges = _fit_bins(train_df[col], nb, bin_strategy)
        binned = _apply_bins(train_df[col], edges)
        iv, woe_map = _woe_iv(y, binned)

        woe_col = f"{col}_woe"
        out[woe_col] = pd.Series(binned).map(woe_map).fillna(0.0).astype(np.float32)
        out[f"{col}_bin"] = binned.astype(np.int16)

        report_rows.append(
            {
                "column_name": col,
                "binning_strategy": strat,
                "n_bins": len(np.unique(edges)) - 1,
                "information_value": iv,
                "woe_json": str(woe_map),
                "bin_edges_json": str(edges.tolist()),
            }
        )
        artifacts.append(
            {
                "column_name": col,
                "bin_edges": edges.tolist(),
                "woe_map": {int(k): float(v) for k, v in woe_map.items()},
            }
        )

    report = pd.DataFrame(report_rows)
    return out, report, artifacts


def apply_indexing_artifacts(
    df: pd.DataFrame,
    artifacts: list[dict],
    target: str | None = "responder_flag",
) -> pd.DataFrame:
    key_cols = [c for c in KEY_COLUMNS if c in df.columns]
    if target and target in df.columns:
        key_cols.append(target)
    out = df[key_cols].copy()

    for art in artifacts:
        col = art["column_name"]
        if col not in df.columns:
            continue
        edges = np.array(art["bin_edges"], dtype=float)
        woe_map = {int(k): float(v) for k, v in art["woe_map"].items()}
        binned = _apply_bins(df[col], edges)
        out[f"{col}_woe"] = pd.Series(binned).map(woe_map).fillna(0.0).astype(np.float32)
        out[f"{col}_bin"] = binned.astype(np.int16)
    return out
