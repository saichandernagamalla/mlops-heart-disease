"""Train and compare candidates with stratified cross-validation."""

from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from .data import FEATURES, TARGET
from .pipeline import candidates

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def log_diagnostics(target, probabilities):
    for name, display, values in (
        ("confusion_matrix", ConfusionMatrixDisplay, (probabilities >= 0.5).astype(int)),
        ("roc_curve", RocCurveDisplay, probabilities),
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        display.from_predictions(target, values, ax=axis)
        axis.set_title(f"Out-of-fold {name.replace('_', ' ')}")
        figure.tight_layout()
        mlflow.log_figure(figure, f"{name}.png")
        plt.close(figure)


def train(
    data_path="data/processed/heart.csv",
    output_dir="artifacts",
    tracking_uri="./mlruns",
    quick=False,
):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease-assignment")
    results = {}
    best_name, best_score = None, -np.inf
    for name, estimator in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            scores = cross_validate(
                estimator,
                X,
                y,
                cv=cv,
                scoring=SCORING,
                return_train_score=False,
                return_estimator=True,
                n_jobs=1,
            )
            summary = {
                metric: float(np.mean(scores[f"test_{metric}"])) for metric in SCORING
            }
            spread = {
                metric: float(np.std(scores[f"test_{metric}"])) for metric in SCORING
            }
            mlflow.log_params(
                {
                    "candidate": name,
                    "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
                    "seed": 42,
                    **{
                        f"model__{key}": value
                        for key, value in estimator.named_steps["model"].get_params().items()
                    },
                }
            )
            for metric, value in summary.items():
                mlflow.log_metric(f"cv_{metric}_mean", value)
                mlflow.log_metric(f"cv_{metric}_std", spread[metric])
            probabilities = np.empty(len(y))
            for (_, validation), fitted in zip(cv.split(X, y), scores["estimator"]):
                probabilities[validation] = fitted.predict_proba(X.iloc[validation])[:, 1]
            log_diagnostics(y, probabilities)
            mlflow.log_dict(
                {
                    metric: scores[f"test_{metric}"].tolist()
                    for metric in SCORING
                },
                "fold_metrics.json",
            )
            results[name] = {"mean": summary, "std": spread, "run_id": run.info.run_id}
            if summary["roc_auc"] > best_score:
                best_name, best_score = name, summary["roc_auc"]
    base = candidates()[best_name]
    grid = (
        {"model__C": [0.1, 1.0, 10.0]}
        if best_name == "logistic_regression"
        else {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 2, 4]}
    )
    search = GridSearchCV(
        base,
        grid if not quick else {k: v[:2] for k, v in grid.items()},
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    joblib.dump(search.best_estimator_, out / "model.joblib")
    summary = {
        "selected_model": best_name,
        "selection_metric": "mean cross-validated ROC-AUC",
        "candidate_cv": results,
        "tuned_best_params": search.best_params_,
        "tuned_cv_roc_auc": float(search.best_score_),
        "features": FEATURES,
        "seed": 42,
    }
    summary_path = out / "training_summary.json"
    with mlflow.start_run(run_name="final_tuned_model") as run:
        summary["final_run_id"] = run.info.run_id
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_params(
            {
                "selected_candidate": best_name,
                "seed": 42,
                "cv_folds": cv.n_splits,
                **{f"tuned_{k}": str(v) for k, v in search.best_params_.items()},
            }
        )
        mlflow.log_metric("tuned_cv_roc_auc", float(search.best_score_))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_dict(
            pd.DataFrame(search.cv_results_).to_dict(orient="list"),
            "tuning_results.json",
        )
        probabilities = cross_val_predict(
            search.best_estimator_, X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        log_diagnostics(y, probabilities)
        input_example = X.head(3)
        signature = infer_signature(
            input_example, search.best_estimator_.predict(input_example)
        )
        mlflow.sklearn.log_model(
            search.best_estimator_,
            "model",
            signature=signature,
            input_example=input_example,
        )
    return summary
