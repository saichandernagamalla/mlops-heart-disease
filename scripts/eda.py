"""Create reproducible EDA tables and plots and log images to MLflow."""
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data import FEATURES, TARGET

sns.set_theme(style="whitegrid", context="talk")
def main():
    df = pd.read_csv("data/processed/heart.csv")
    out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv("reports/descriptive_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv("reports/missing_values.csv")
    fig, ax = plt.subplots(figsize=(6,4)); sns.countplot(data=df, x=TARGET, ax=ax, hue=TARGET, legend=False, palette="Set2"); ax.set(title="Target balance", xlabel="Heart disease (0 absent, 1 present)", ylabel="Patients"); fig.tight_layout(); fig.savefig(out/"class_balance.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14,14))
    for ax, col in zip(axes.flat, ["age","trestbps","chol","thalach","oldpeak","cp"]): sns.histplot(data=df, x=col, hue=TARGET, multiple="layer", stat="density", common_norm=False, ax=ax)
    fig.suptitle("Selected feature distributions by target", y=1.01); fig.tight_layout(); fig.savefig(out/"feature_distributions.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13,10)); sns.heatmap(df[FEATURES+[TARGET]].corr(numeric_only=True), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax); ax.set_title("Feature correlation (codes shown numerically for exploration)"); fig.tight_layout(); fig.savefig(out/"correlation_heatmap.png", dpi=160); plt.close(fig)
    mlflow.set_tracking_uri("./mlruns"); mlflow.set_experiment("uci-heart-disease-assignment")
    with mlflow.start_run(run_name="eda"):
        mlflow.log_param("rows",len(df)); mlflow.log_param("features",len(FEATURES)); mlflow.log_metric("positive_fraction",float(df[TARGET].mean()))
        for image in out.glob("*.png"): mlflow.log_artifact(str(image), artifact_path="eda")
    print("EDA outputs saved under reports/figures")
if __name__ == "__main__": main()
