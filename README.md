# Heart Disease MLOps Project

An original, reproducible teaching project for the UCI Cleveland Heart Disease dataset. It demonstrates acquisition, EDA, leakage-safe preprocessing, cross-validated model comparison, MLflow tracking, a FastAPI service, tests, CI, Docker, Kubernetes manifests, and operational metrics.

> Educational assignment only. This model is not clinically validated and must not be used for diagnosis, triage, or care decisions. The data is small, historical, and not representative of current clinical populations.

## Quick start

Requires Python 3.11. From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.fetch_data
python -m scripts.eda
python -m scripts.train
pytest -q
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Training compares Logistic Regression and Random Forest using stratified 5-fold CV, reports accuracy, precision, recall, and ROC-AUC, and tunes the highest mean ROC-AUC candidate. Run `python -m scripts.train --quick` for a reduced search. MLflow uses `./mlruns`; inspect it with `mlflow ui --backend-store-uri ./mlruns`.

The API loads `artifacts/model.joblib`. It exposes `/health`, `/predict`, and Prometheus text metrics at `/metrics`. Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

From PowerShell, run `./scripts/verify_container.ps1` to build the image and check `/health`, `/predict`, and `/metrics` in a local container. See `docs/OPERATIONS.md` for the Minikube steps.

See `docs/` for provenance and operating steps, and `reports/FINAL_REPORT.md` for the report draft. The GitHub Actions workflow runs lint, tests, data acquisition, and a training smoke run.

## Finish your submission

See [the timed recording flow](docs/VIDEO_CHECKLIST.md) for the exact commands and narration.
MLflow is served at `http://127.0.0.1:5000`. The live API dashboard is `/monitor`.

1. Commit and push the local fixes to obtain a new GitHub Actions result. The previous failure was reproduced as `ModuleNotFoundError: No module named 'src'`; `pytest.ini` fixes project-root imports.
2. Enable Docker Desktop Kubernetes and follow [the deployment steps](docs/OPERATIONS.md). The service uses a LoadBalancer, not just a NodePort.
3. Capture actual CI, Docker, Kubernetes, MLflow, and API screenshots in `screenshots/`.
4. Record your own short pipeline walkthrough and insert the final video link in the report.
5. Review the analysis and explain your decisions in your own words, as required by the assignment's academic integrity note.

The API uses the original UCI codes: `cp=1..4`, `slope=1..3`, `ca=0..3`, and `thal` in `{3,6,7}`. Remapped Kaggle codes such as `slope=0` or `thal=1` now return HTTP 422 instead of being silently treated as unknown categories. `/health` returns HTTP 503 when the model is unavailable so deployment probes cannot incorrectly mark it ready. Request logs contain route, method, status, and latency, never patient feature values.

The included Minikube manifest is local learning infrastructure. A public cloud release needs account-specific registry, ingress/TLS, and secrets configuration; the project does not claim a public deployment.
