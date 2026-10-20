from fastapi.testclient import TestClient
import src.api as api

client = TestClient(api.app)

def test_health_without_artifact():
    api.model = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False

def test_prediction_response_shape(monkeypatch):
    class Fake:
        def predict(self, frame): return [1]
        def predict_proba(self, frame): return [[0.2, 0.8]]
    monkeypatch.setattr(api, "model", Fake())
    payload = {"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}
    response = client.post("/predict", json=payload)
    assert response.json() == {"prediction":1,"label":"positive","confidence_positive":0.8}
