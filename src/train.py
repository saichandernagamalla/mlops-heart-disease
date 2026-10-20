"""Train and compare candidates with stratified cross-validation."""
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from .data import FEATURES, TARGET
from .pipeline import candidates

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}

def train(data_path="data/processed/heart.csv", output_dir="artifacts", tracking_uri="./mlruns", quick=False):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease-assignment")
    results = {}
    best_name, best_score = None, -np.inf
    for name, estimator in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            scores = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, return_train_score=False, n_jobs=1)
            summary = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in SCORING}
            spread = {metric: float(np.std(scores[f"test_{metric}"])) for metric in SCORING}
            mlflow.log_params({"candidate": name, "cv": "StratifiedKFold(5, shuffle=True, random_state=42)", "seed": 42})
            for metric, value in summary.items():
                mlflow.log_metric(f"cv_{metric}_mean", value); mlflow.log_metric(f"cv_{metric}_std", spread[metric])
            results[name] = {"mean": summary, "std": spread, "run_id": run.info.run_id}
            if summary["roc_auc"] > best_score: best_name, best_score = name, summary["roc_auc"]
    base = candidates()[best_name]
    grid = ({"model__C": [0.1, 1.0, 10.0]} if best_name == "logistic_regression" else {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 2, 4]})
    search = GridSearchCV(base, grid if not quick else {k: v[:2] for k, v in grid.items()}, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    search.fit(X, y)
    joblib.dump(search.best_estimator_, out / "model.joblib")
    summary = {"selected_model": best_name, "selection_metric": "mean cross-validated ROC-AUC", "candidate_cv": results, "tuned_best_params": search.best_params_, "tuned_cv_roc_auc": float(search.best_score_), "features": FEATURES, "seed": 42}
    summary_path = out / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="final_tuned_model"):
        mlflow.log_params({"selected_candidate": best_name, **{f"tuned_{k}": str(v) for k,v in search.best_params_.items()}})
        mlflow.log_metric("tuned_cv_roc_auc", float(search.best_score_))
        mlflow.log_artifact(str(summary_path)); mlflow.sklearn.log_model(search.best_estimator_, "model")
    return summary
