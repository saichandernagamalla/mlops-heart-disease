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
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

From PowerShell, run `./scripts/verify_container.ps1` to build the image and check `/health`, `/predict`, and `/metrics` in a local container. See `docs/OPERATIONS.md` for the Minikube steps.

See `docs/` for provenance and operating steps, and `reports/FINAL_REPORT.md` for the report draft. The GitHub Actions workflow runs lint, tests, data acquisition, and a training smoke run.

## Finish your submission

1. Run the workflow in your own GitHub repository and add its URL to the report.
2. Capture actual CI, Docker, Kubernetes, MLflow, and API screenshots in `screenshots/`.
3. Record your own short pipeline walkthrough using `docs/VIDEO_CHECKLIST.md`.
4. Fill report results from your real MLflow run; do not invent metrics or deployment evidence.
5. Personalize the analysis and explain your decisions in your own words, as required by the assignment's academic integrity note.

The included Minikube manifest is local learning infrastructure. A public cloud release needs account-specific registry, ingress/TLS, and secrets configuration; the project does not claim a public deployment.
