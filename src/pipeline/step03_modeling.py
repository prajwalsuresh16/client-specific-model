"""Step 3 — Indexing reports, GROQ shortlist, tier-specific live modeling, MLflow."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config_loader import load_settings, resolve_path
from src.databricks.feature_store import publish_feature_table
from src.databricks.mlflow_setup import configure_mlflow, log_sklearn_model_to_uc
from src.dataset_io import read_dataset, write_dataset
from src.groq_fe import groq_feature_decisions, _univariate_stats
from src.indexing import apply_indexing_artifacts, fit_and_apply_indexing
from src.io_utils import ensure_dir
from src.modeling_accel import log_shap_summary, tune_classifier

from src.id_keys import KEY_COLUMNS, MODELING_EXCLUDE


def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = MODELING_EXCLUDE
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _tier_filter(df: pd.DataFrame, tier: str, cfg: dict) -> pd.DataFrame:
    if tier == "generic":
        return df
    if tier == "product":
        pc = cfg["campaign"]["product_code"]
        return df[df["product_code"].astype(str) == pc]
    if tier == "client_product":
        return df[
            (df["client_id"].astype(str) == cfg["client"]["id"])
            & (df["product_code"].astype(str) == cfg["campaign"]["product_code"])
        ]
    return df


def _build_model(name: str, cfg: dict) -> object:
    cw = cfg["modeling"]["class_weight_responder_multiplier"]
    if name == "logistic_regression":
        return LogisticRegression(max_iter=500, class_weight={0: 1, 1: cw}, random_state=42)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight={0: 1, 1: cw}, random_state=42, n_jobs=-1
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=cw,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=120, class_weight={0: 1, 1: cw}, random_state=42, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=120, depth=6, class_weights=[1, cw], random_seed=42, verbose=False)
    raise ValueError(name)


def run(sample_n: int | None = 50000) -> Path:
    cfg = load_settings()
    gold = ensure_dir(resolve_path("data", "regional_bank", "gold"))
    models_dir = ensure_dir(resolve_path("data", "regional_bank", "models"))
    idx_cfg = cfg.get("indexing", {})
    min_tier_rows = int(cfg["modeling"].get("min_rows_per_tier", 500))

    train = read_dataset("gold", "gold_training_sample", sample_n=sample_n)

    target = "responder_flag"
    stats = _univariate_stats(train, target, max_cols=cfg["groq"].get("max_columns_per_request", 80))
    decisions = groq_feature_decisions(stats, cfg["campaign"]["product_code"], cfg["client"]["id"])
    n_groq = sum(1 for d in decisions if d.get("reason") not in ("rule_fallback", None))
    print(
        f"Feature engineering: GROQ model={cfg['groq']['model']} "
        f"({n_groq}/{len(decisions)} columns via LLM; rest rule fallback). "
        "Audit: gold_groq_fe_audit"
    )

    min_iv = float(idx_cfg.get("min_information_value", 0.02))
    fe_train, index_report, artifacts = fit_and_apply_indexing(
        train, decisions, target=target, n_bins=int(idx_cfg.get("n_bins", 10))
    )
    if not index_report.empty and "information_value" in index_report.columns:
        keep_iv = set(index_report.loc[index_report["information_value"] >= min_iv, "column_name"])
        artifacts = [a for a in artifacts if a["column_name"] in keep_iv]

    write_dataset(fe_train, "gold", "gold_feature_matrix")
    write_dataset(index_report, "gold", "gold_index_report")
    write_dataset(pd.DataFrame(decisions), "gold", "gold_groq_fe_audit")
    publish_feature_table(fe_train, "gold_feature_matrix")
    (models_dir / "index_artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")

    configure_mlflow()
    all_preds: dict[str, pd.Series] = {}

    for tier in ("generic", "product", "client_product"):
        tier_train = _tier_filter(train, tier, cfg)
        if len(tier_train) < min_tier_rows:
            continue

        tier_fe, tier_report, tier_artifacts = fit_and_apply_indexing(
            tier_train, decisions, target=target, n_bins=int(idx_cfg.get("n_bins", 10))
        )
        features = _feature_columns(tier_fe)
        if not features:
            continue

        X = tier_fe[features].fillna(0).values
        y = tier_train[target].astype(int).values
        if len(np.unique(y)) < 2:
            continue

        for algo in cfg["modeling"]["algorithms"]:
            model_name = f"{tier}_{algo}"
            with mlflow.start_run(run_name=model_name):
                model = _build_model(algo, cfg)
                best = tune_classifier(algo, X, y, cfg)
                if best:
                    model.set_params(**best)
                model.fit(X, y)
                log_shap_summary(model, X[: min(500, len(X))], features, models_dir / "shap" / model_name)
                log_sklearn_model_to_uc(
                    model,
                    artifact_path=model_name,
                    tier=tier,
                    algorithm=algo,
                    metrics={"train_pos_rate": float(y.mean()), "train_rows": float(len(tier_train))},
                    params={"tier": tier, "algorithm": algo, "n_features": len(features)},
                )

                bundle = {
                    "model": model,
                    "features": features,
                    "tier": tier,
                    "artifacts": tier_artifacts,
                }
                path = models_dir / f"{model_name}.pkl"
                with path.open("wb") as f:
                    pickle.dump(bundle, f)
                mlflow.log_artifact(str(path))

                if "bpid" in tier_train.columns:
                    proba = model.predict_proba(X)[:, 1]
                    all_preds[f"prob_{model_name}"] = pd.Series(proba, index=tier_train.index)

    scoring = read_dataset("gold", "gold_scoring_population", sample_n=sample_n)

    artifacts_path = models_dir / "index_artifacts.json"
    if artifacts_path.exists():
        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    fe_score = apply_indexing_artifacts(scoring, artifacts, target=None)

    key_cols = [c for c in KEY_COLUMNS if c in scoring.columns]
    score_out = scoring[key_cols].copy()

    for pkl in sorted(models_dir.glob("*.pkl")):
        with pkl.open("rb") as f:
            bundle = pickle.load(f)
        tier = bundle["tier"]
        tier_score = _tier_filter(scoring, tier, cfg)
        if tier_score.empty:
            continue
        tier_fe = apply_indexing_artifacts(tier_score, bundle.get("artifacts", artifacts), target=None)
        features = bundle["features"]
        Xs = tier_fe[features].fillna(0).values
        proba = bundle["model"].predict_proba(Xs)[:, 1]
        col = f"prob_{pkl.stem}"
        if col not in score_out.columns:
            score_out[col] = np.nan
        score_out.loc[tier_score.index, col] = proba

    # Default probability for rows outside tier: use generic tier if present
    prob_cols = [c for c in score_out.columns if c.startswith("prob_")]
    if prob_cols:
        score_out["prob_default"] = score_out[prob_cols].mean(axis=1, skipna=True)

    return write_dataset(score_out, "gold", "gold_model_predictions")
