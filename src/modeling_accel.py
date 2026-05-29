"""Optional Optuna + SHAP accelerators for step03 (env-gated)."""

from __future__ import annotations

import os
from typing import Any


def optuna_enabled() -> bool:
    return os.environ.get("FMG_USE_OPTUNA", "").lower() in ("1", "true", "yes")


def shap_enabled() -> bool:
    return os.environ.get("FMG_USE_SHAP", "").lower() in ("1", "true", "yes")


def tune_classifier(estimator_name: str, x_train, y_train, cfg: dict) -> dict[str, Any]:
    """Hyperparameter search with Optuna when enabled; otherwise defaults."""
    if not optuna_enabled():
        return {}
    import optuna
    from sklearn.model_selection import cross_val_score

    trials = int(os.environ.get("FMG_OPTUNA_TRIALS", cfg.get("modeling", {}).get("optuna_trials", 20)))

    def objective(trial: optuna.Trial) -> float:
        if estimator_name == "random_forest":
            model = __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 80, 200),
                max_depth=trial.suggest_int("max_depth", 6, 16),
                n_jobs=-1,
                random_state=42,
            )
        elif estimator_name == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 80, 200),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2),
                n_jobs=-1,
                random_state=42,
            )
        else:
            return 0.0
        scores = cross_val_score(model, x_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params


def log_shap_summary(model: Any, x_sample, feature_names: list[str], out_dir) -> None:
    if not shap_enabled():
        return
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.Explainer(model, x_sample)
    values = explainer(x_sample)
    shap.summary_plot(values, x_sample, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt

    plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight", dpi=120)
    plt.close()
