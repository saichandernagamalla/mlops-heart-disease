import numpy as np
import src.api as api


def test_health_without_artifact(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    assert api.health() == {"status": "model_missing", "model_loaded": False}


def test_prediction_response_shape(monkeypatch):
    class Fake:
        def predict(self, frame):
            return np.array([1])

        def predict_proba(self, frame):
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(api, "model", Fake())
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    response = api.predict(api.Patient(**payload))
    assert response == {
        "prediction": 1,
        "label": "positive",
        "confidence_positive": 0.8,
    }
