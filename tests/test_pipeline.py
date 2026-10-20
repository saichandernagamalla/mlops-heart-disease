import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET
from src.pipeline import candidates, make_preprocessor


def test_preprocessing_imputes_and_accepts_unseen_categories():
    frame = pd.read_csv("data/processed/heart.csv")[FEATURES]
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(frame)
    assert transformed.shape[0] == len(frame)
    assert np.isfinite(transformed).all()
    unseen = frame.iloc[[0]].copy()
    unseen.loc[:, "cp"] = 99
    unseen.loc[:, "ca"] = np.nan
    assert np.isfinite(preprocessor.transform(unseen)).all()


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_model_pipeline_serialization_and_probability(name, tmp_path):
    frame = pd.read_csv("data/processed/heart.csv").iloc[:80]
    model = candidates()[name]
    if name == "random_forest":
        model.set_params(model__n_estimators=5, model__n_jobs=1)
    model.fit(frame[FEATURES], frame[TARGET])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    probabilities = restored.predict_proba(frame[FEATURES])
    assert probabilities.shape == (len(frame), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(
        model.predict(frame[FEATURES]), restored.predict(frame[FEATURES])
    )
