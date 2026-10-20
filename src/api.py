"""FastAPI inference service. Train before startup to create its model artifact."""

from pathlib import Path
from contextlib import asynccontextmanager
import json
import logging
from time import perf_counter
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import HTMLResponse, JSONResponse, Response
from .data import FEATURES

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    model = None
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Model artifact loaded: %s", MODEL_PATH)
    else:
        logger.error("Model artifact missing: %s; predictions are unavailable", MODEL_PATH)
    yield


app = FastAPI(
    title="Heart Risk Classifier",
    version="1.0.0",
    description="Educational demo; not for clinical use.",
    lifespan=lifespan,
)
REQUESTS = Counter("prediction_requests_total", "Prediction requests", ["outcome"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction request latency")
MODEL_PATH = Path("artifacts/model.joblib")
model = None


@app.middleware("http")
async def log_request(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    logger.info(
        json.dumps(
            {
                "event": "api_request",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((perf_counter() - started) * 1000, 3),
            }
        )
    )
    if request.url.path == "/predict" and response.status_code == 422:
        REQUESTS.labels(outcome="invalid").inc()
    return response


class Patient(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(ge=40, le=300)
    chol: float = Field(ge=50, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=30, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=15)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


@app.get("/health")
def health():
    status = {
        "status": "ok" if model is not None else "model_missing",
        "model_loaded": model is not None,
    }
    return status if model is not None else JSONResponse(status, status_code=503)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    return """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Heart API monitoring</title>
<style>body{font:18px system-ui;margin:3rem;background:#f2f6fa;color:#17365d}
pre{background:white;padding:1.5rem;overflow:auto;font-size:14px}
a{margin-right:2rem}h1{margin-bottom:.5rem}</style>
<h1>Heart API: live monitoring</h1>
<p>Educational demonstration only; not for clinical decisions. Refreshes every 2 seconds.</p>
<p><a href="/docs">API documentation</a><a href="/metrics">Full Prometheus metrics</a></p>
<h2>Model readiness</h2><pre id="health">Loading...</pre>
<h2>Prediction outcomes and latency (seconds)</h2><pre id="metrics">Loading...</pre>
<script>
async function refresh() {
  try {
    const health = await fetch('/health');
    document.getElementById('health').textContent =
      'HTTP ' + health.status + '\\n' + JSON.stringify(await health.json(), null, 2);
    const response = await fetch('/metrics');
    if (!response.ok) throw new Error('Metrics HTTP ' + response.status);
    document.getElementById('metrics').textContent = (await response.text())
      .split('\\n').filter(line => line.startsWith('prediction_')).join('\\n');
  } catch (error) {
    document.getElementById('metrics').textContent = 'Monitoring unavailable: ' + error;
  }
}
refresh(); setInterval(refresh, 2000);
</script></html>"""


@app.post("/predict")
def predict(patient: Patient):
    if model is None:
        REQUESTS.labels(outcome="unavailable").inc()
        raise HTTPException(
            status_code=503, detail="Model artifact unavailable; train the model first."
        )
    with LATENCY.time(), REQUESTS.labels(outcome="error").count_exceptions():
        row = pd.DataFrame(
            [{key: getattr(patient, key) for key in FEATURES}], columns=FEATURES
        )
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1])
    REQUESTS.labels(outcome="success").inc()
    return {
        "prediction": prediction,
        "label": "positive" if prediction else "negative",
        "confidence_positive": round(probability, 6),
    }
