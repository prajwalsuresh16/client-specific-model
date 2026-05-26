from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def rank_mix_score(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
    """Legacy rank-mix: average of ranks (lower = better prospect)."""
    r1 = pd.Series(series_a).rank(ascending=False, method="average").values
    r2 = pd.Series(series_b).rank(ascending=False, method="average").values
    return (r1 + r2) / 2.0


def decile_lift(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), n_bins, labels=False, duplicates="drop")
    agg = df.groupby("decile", observed=True).agg(
        n=("y", "count"),
        responders=("y", "sum"),
        rate=("y", "mean"),
    )
    base = df["y"].mean()
    agg["lift"] = agg["rate"] / base if base > 0 else np.nan
    return agg.reset_index()


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    df["cum_good"] = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    df["cum_bad"] = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float((df["cum_good"] - df["cum_bad"]).abs().max())


def pppm_rank_correlation(
    y_respond: np.ndarray, premium: np.ndarray, y_score: np.ndarray
) -> float:
    """Legacy PPPM ranking proxy among responders."""
    mask = y_respond == 1
    if mask.sum() < 10:
        return float("nan")
    return float(np.corrcoef(y_score[mask], premium[mask])[0, 1])


def expected_pppm_score(
    y_respond: np.ndarray,
    premium: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Dollars-per-thousand-mailed style metric: revenue in top score decile / mailed × 1000.
    Higher is better alignment of score with premium dollars.
    """
    n = len(y_score)
    if n < 100:
        return float("nan")
    df = pd.DataFrame({"y": y_respond.astype(float), "prem": premium.astype(float), "s": y_score})
    df["decile"] = pd.qcut(df["s"].rank(method="first"), n_bins, labels=False, duplicates="drop")
    top = df["decile"].max()
    top_rows = df[df["decile"] == top]
    revenue = (top_rows["y"] * top_rows["prem"]).sum()
    pppm = (revenue / n) * 1000.0
    return float(pppm)


def model_correlation(col_a: np.ndarray, col_b: np.ndarray) -> float:
    a = pd.Series(col_a).fillna(0)
    b = pd.Series(col_b).fillna(0)
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return float(a.corr(b))


def passes_orthogonality(col_a: np.ndarray, col_b: np.ndarray, max_abs_corr: float) -> bool:
    return abs(model_correlation(col_a, col_b)) <= max_abs_corr
