# End-to-End MLOps for Heart Disease Classification

**Course:** Machine Learning Operations (MLOps) AIMLCZG523  
**Assignment:** 01  
**Author:** [Your name]  
**Date:** [Submission date]  
**Repository:** [Add your repository URL]  
**Deployment:** [Add verified URL or local access instructions]

> Educational demonstration only. This artifact is not a medical device and cannot be used for clinical decisions.

## 1. Problem and scope

This project demonstrates a pipeline that predicts heart disease presence in the processed Cleveland subset. It maps `num=0` to absence and `num=1..4` to presence. The objective is to practice reproducible data handling, model development, experiment tracking, automated validation, packaging, serving, and monitoring. The project does not establish clinical validity.

## 2. Data acquisition and EDA

The dataset comes from the UCI Machine Learning Repository, Heart Disease dataset (DOI: 10.24432/C52P4X). The acquisition script fetches the processed Cleveland file from the official archive. It contains 303 rows, 13 selected predictors, and the binary outcome under the mapping above. Missing `?` tokens remain null for pipeline imputation.

Insert generated figures from `reports/figures/` here and discuss observed class balance, distributions, missingness, and correlations. Coded category correlations are descriptive only because codes need not have interval meaning. Attach your actual EDA outputs.

## 3. Feature processing and model selection

The preprocessing transformer uses median imputation and standardization for numeric features. Coded categories are mode-imputed and one-hot encoded. All steps are in an sklearn Pipeline, and validation folds fit preprocessing independently. Candidates are class-weighted Logistic Regression and Random Forest. Both are compared with shuffled, stratified five-fold cross-validation using accuracy, precision, recall, and ROC-AUC. A small tuning grid is evaluated for the selected candidate using ROC-AUC.

### Cross-validation results

| Candidate | Accuracy mean ± SD | Precision mean ± SD | Recall mean ± SD | ROC-AUC mean ± SD |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8447 ± 0.0554 | 0.8475 ± 0.0570 | 0.8050 ± 0.0857 | 0.9188 ± 0.0213 |
| Random Forest | 0.8416 ± 0.0222 | 0.8447 ± 0.0415 | 0.8056 ± 0.0438 | 0.9154 ± 0.0204 |

Logistic Regression was selected by mean ROC-AUC (0.9188 vs. 0.9154). Its mean accuracy and precision were also slightly higher; Random Forest had marginally higher mean recall and lower fold-to-fold standard deviations for accuracy and recall. The differences are small relative to the fold variation, so this comparison does not establish a meaningful performance advantage. The selected Logistic Regression pipeline was tuned with `C=0.1`; its best cross-validated ROC-AUC was 0.9208. This is a tuning estimate on the same dataset, not a held-out test result. The five-fold estimates are noisy because the dataset is small.

## 4. Experiment tracking and reproducibility

MLflow experiment: `uci-heart-disease-assignment`. Candidate runs record seed, CV procedure, metric means and standard deviations. The EDA run logs figures. The selected pipeline is stored as `artifacts/model.joblib`, with parameters and CV summary in `artifacts/training_summary.json`. Dependencies are listed in `requirements.txt`; stochastic operations use seed 42. The final MLflow model includes an inferred input/output signature and a three-row input example so its expected schema is visible in the run artifacts.

## 5. System architecture

```mermaid
flowchart LR
  U[UCI archive] --> F[fetch_data module]
  F --> D[processed CSV]
  D --> E[EDA figures]
  D --> T[CV training and MLflow]
  T --> M[model.joblib]
  M --> C[Docker image]
  C --> A[FastAPI /predict]
  A --> P[/health and /metrics]
  G[GitHub Actions] --> Q[lint, tests, training smoke]
```

## 6. CI/CD and testing

GitHub Actions installs dependencies, runs Ruff and unit tests, downloads the dataset, performs a reduced training run, and uploads artifacts. Tests cover target conversion/missing values and API health/prediction response shape. The supplied Actions screenshot shows two runs on `master`, both failed: `mlops-ci #2` for commit `44a7ee3` (45 seconds) and `mlops-ci #1` for commit `372cc59` (57 seconds). This confirms the workflow was triggered, but CI is not yet passing. The run-list screenshot does not show job logs, so the failure cause remains to be diagnosed. Inspect the logs, fix the failure, rerun the workflow, and capture a successful run before submission.

![GitHub Actions workflow runs showing both runs failed](../screenshots/github_actions_ci.png)

## 7. Packaging, API, and deployment

FastAPI serves the trained pipeline through `POST /predict`. Its request schema validates the 13 model features, and its response includes a binary prediction, a positive/negative label, and positive-class confidence. `GET /health` reports whether the model is loaded; `GET /metrics` exposes Prometheus request and latency data. Docker packages the API with the trained artifact, and the Kubernetes manifest supports Minikube with a NodePort and health probes.

**Container verification (Linux, 20 October 2026):** The image was built and the API was run on the Linux host with port 8000 published at `http://localhost:8000`. A sample 13-feature JSON request to `POST /predict` returned `{"prediction":0,"label":"negative","confidence_positive":0.355847}`. This confirms the packaged model loaded and served an inference response in the container.

**Deployment evidence:** Local container access was verified at `http://localhost:8000`. Minikube or cloud deployment, its endpoint, and deployment screenshots remain pending.

## 8. Monitoring and limitations

The Linux container's `GET /metrics` output recorded `prediction_requests_total{outcome="success"} 1`, confirming one successful inference request. The latency histogram recorded `prediction_latency_seconds_count 1` and `prediction_latency_seconds_sum 0.02333469199948013` seconds, approximately 23.3 ms for that request. The API therefore exposes and records request outcome and latency in Prometheus format. This is basic instrumentation; a production system would still need Prometheus scraping, dashboards, alerts, structured access-controlled logs, drift monitoring, and privacy controls. Limitations include historical clinical practice, limited sample size, population selection, missingness, and possible demographic and measurement bias. Output probabilities are not calibrated medical risk estimates.

## 9. Conclusion and next steps

The project now includes a trained and tracked model, a containerized FastAPI service, a successful local prediction, and Prometheus request and latency metrics. Next steps include Minikube or cloud deployment, recording CI and deployment evidence, completing author and submission details, and external validation, calibration, subgroup analysis, image scanning, deployment secrets, ingress, and operational monitoring. Do not deploy for patient care.

## 10. References and evidence

- UCI Machine Learning Repository, [Heart Disease dataset](https://doi.org/10.24432/C52P4X).
- Assignment 01 instructions supplied by the course.
- Screenshots: save the Linux container build and API verification screenshots in `screenshots/` before submission.
- Video: [link to your own short recording].
