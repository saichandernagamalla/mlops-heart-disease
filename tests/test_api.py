import numpy as np
import pytest
from fastapi.testclient import TestClient
import src.api as api


def test_health_without_artifact(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "model_missing", "model_loaded": False}


@pytest.fixture
def patient():
    return {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }


def test_prediction_response_shape(monkeypatch, patient):
    class Fake:
        def predict(self, frame):
            return np.array([1])

        def predict_proba(self, frame):
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(api, "model", Fake())
    response = TestClient(api.app).post("/predict", json=patient)
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "label": "positive",
        "confidence_positive": 0.8,
    }


def test_prediction_without_model(monkeypatch, patient):
    monkeypatch.setattr(api, "model", None)
    response = TestClient(api.app).post("/predict", json=patient)
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.parametrize("field,value", [("cp", 0), ("slope", 0), ("ca", 4), ("thal", 1)])
def test_invalid_uci_codes_are_rejected(patient, field, value):
    response = TestClient(api.app).post("/predict", json={**patient, field: value})
    assert response.status_code == 422


def test_metrics_dashboard_and_private_request_log(caplog, patient):
    with caplog.at_level("INFO", logger="uvicorn.error"):
        response = TestClient(api.app).post("/predict", json={**patient, "thal": 1})
    assert response.status_code == 422
    messages = " ".join(record.message for record in caplog.records)
    assert '"event": "api_request"' in messages
    assert '"status": 422' in messages
    assert "chol" not in messages
    client = TestClient(api.app)
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'prediction_requests_total{outcome="invalid"}' in metrics.text
    assert "prediction_latency_seconds" in metrics.text
    assert client.get("/monitor").status_code == 200
