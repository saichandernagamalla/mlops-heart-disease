"""Data schema and loading utilities for the processed Cleveland subset."""
from pathlib import Path
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
COLUMNS = FEATURES + ["num"]

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read UCI data; retain '?' values as nulls for fold-local imputation."""
    frame = pd.read_csv(path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[TARGET] = (frame.pop("num") > 0).astype("int8")
    return frame
