"""FastAPI inference service. Train before startup to create its model artifact."""

from pathlib import Path
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .data import FEATURES


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
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


class Patient(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(ge=40, le=300)
    chol: float = Field(ge=50, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=30, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=15)
    slope: int = Field(ge=0, le=3)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=7)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_missing",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(patient: Patient):
    if model is None:
        REQUESTS.labels(outcome="unavailable").inc()
        raise HTTPException(
            status_code=503, detail="Model artifact unavailable; train the model first."
        )
    with LATENCY.time():
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
