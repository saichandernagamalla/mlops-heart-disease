"""Leakage-safe preprocessing and candidate estimators."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import CATEGORICAL, NUMERIC

def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)], remainder="drop")

def candidates(seed: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([("prep", make_preprocessor()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "random_forest": Pipeline([("prep", make_preprocessor()), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1))]),
    }
