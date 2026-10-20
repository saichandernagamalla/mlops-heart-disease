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

### Actual results (fill after running)

| Candidate | Accuracy mean ± SD | Precision mean ± SD | Recall mean ± SD | ROC-AUC mean ± SD |
|---|---:|---:|---:|---:|
| Logistic Regression | [from MLflow] | [from MLflow] | [from MLflow] | [from MLflow] |
| Random Forest | [from MLflow] | [from MLflow] | [from MLflow] | [from MLflow] |

Selected model: [from `artifacts/training_summary.json`]. Explain the tradeoff and why ROC-AUC was chosen. Small sample size makes fold estimates noisy; avoid claims of clinical performance.

## 4. Experiment tracking and reproducibility

MLflow experiment: `uci-heart-disease-assignment`. Candidate runs record seed, CV procedure, metric means and standard deviations. The EDA run logs figures. The selected pipeline is stored as `artifacts/model.joblib`, with parameters and CV summary in `artifacts/training_summary.json`. Dependencies are listed in `requirements.txt`; stochastic operations use seed 42.

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

GitHub Actions installs dependencies, runs Ruff, unit tests, downloads the dataset, performs a reduced training run, and uploads artifacts. Tests cover target conversion/missing values and API health/prediction response shape. Add your screenshot of a successful workflow run and report failures and fixes.

## 7. Packaging, API, and deployment

FastAPI validates input fields and returns a binary prediction with positive-class probability. `GET /health` reports model availability; `GET /metrics` exports request count and latency. Docker packages the trained artifact and API. The included Kubernetes manifest supports Minikube with a NodePort and health probes.

**Deployment evidence:** [insert actual platform, steps, endpoint/access details, date, and screenshots]. The repository does not claim a public deployment until you perform and verify one.

## 8. Monitoring and limitations

Request totals by outcome and request latency are exposed in Prometheus format. This is basic instrumentation; a production system would need scraping, dashboards, alerts, structured access-controlled logs, drift monitoring, and privacy controls. Limitations include historical clinical practice, limited sample size, population selection, missingness, and possible demographic and measurement bias. Output probabilities are not calibrated medical risk estimates.

## 9. Conclusion and next steps

Summarize verified outcomes in your own words. Next steps include external validation on a current representative cohort, calibration and subgroup analysis, stronger CI gating, image scanning, deployment secrets and ingress, and operational monitoring. Do not deploy for patient care.

## 10. References and evidence

- UCI Machine Learning Repository, [Heart Disease dataset](https://doi.org/10.24432/C52P4X).
- Assignment 01 instructions supplied by the course.
- Screenshots: [list actual files in `screenshots/`].
- Video: [link to your own short recording].
