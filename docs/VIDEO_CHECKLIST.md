# Recording flow: 5-6 minutes

Author: SAI CHANDER NAGAMALLA, 2025AE05029.

## Before pressing Record

- Open this repository in VS Code and a PowerShell terminal at the project root.
- Open MLflow: <http://127.0.0.1:5000/#/experiments/144320529290097270>.
- In MLflow, filter runs with `tags.recording_demo = 'ready'`. Four finished runs are tagged: EDA, Logistic Regression, Random Forest, and the final tuned model.
- Open the API dashboard at <http://localhost:8001/monitor> and keep Docker Desktop visible.
- If Kubernetes is ready, keep <http://localhost:8000/monitor> in a second tab.
- Close unrelated windows and notifications. Do not show credentials.
- Do not reinstall packages, rebuild the image, or retrain during the recording; show the completed outputs and reproducible commands instead.

## 0:00-0:30 -- Introduce the problem and architecture

Show `README.md` and the architecture in `reports/FINAL_REPORT.md`.

Say in your own words: "This is my reproducible MLOps pipeline for binary heart-disease classification. It covers UCI acquisition, EDA, leakage-safe training, MLflow tracking, tests, container serving, local Kubernetes, and monitoring. It is an educational model, not a clinical tool."

## 0:30-1:10 -- Data and EDA

Show `scripts/fetch_data.py`, the cleaned CSV, and the three figures in `reports/figures/`.

Explain: 303 rows, 13 predictors plus target; 164 negative and 139 positive records; four missing `ca` values and two missing `thal` values. Missing values are imputed inside the model pipeline, not before CV.

Mention that positive records have lower average maximum heart rate and higher average ST depression; these are associations, not clinical or causal conclusions.

## 1:10-2:30 -- MLflow experiments and model selection

1. Open the experiment above and apply `tags.recording_demo = 'ready'`.
2. Select the Logistic Regression and Random Forest runs and choose **Compare**.
3. Show `cv_accuracy_mean`, `cv_precision_mean`, `cv_recall_mean`, and `cv_roc_auc_mean`.
4. Explain five-fold stratified CV and seed 42. Logistic Regression ROC-AUC: **0.9188**; Random Forest: **0.9154**. Differences are small.
5. Open `final_tuned_model`: <http://127.0.0.1:5000/#/experiments/144320529290097270/runs/a7f870e2af164e1ea71d50eff5ee422f>.
6. Show `tuned_model__C=0.1`, tuned CV ROC-AUC **0.9208**, and **Artifacts**: `model`, `training_summary.json`, `tuning_results.json`, `roc_curve.png`, and `confusion_matrix.png`.
7. Open the EDA run's `eda` artifact folder to show the tracked figures.

Say: "Preprocessing is fitted within each fold. Model selection and tuning share this small development dataset, so these are not independent held-out test results."

## 2:30-3:00 -- Tests and CI

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

Expected current result: **12 passed**. Show `.github/workflows/ci.yml`: lint, unit tests, data acquisition, EDA, training, and artifact upload.

Be accurate: the old remote run failed on imports. The local fix is verified; a new remote result requires committing and pushing. Do not describe a failed Actions screenshot as passing.

## 3:00-3:45 -- Docker and live inference

Show the running `heart-api-demo` container in Docker Desktop, then:

```powershell
docker ps --filter name=heart-api-demo
.\scripts\demo_api.ps1
```

Explain the JSON prediction, label, and `confidence_positive`. Show `scripts/sample_patient.json`. The confidence is an uncalibrated model output, not a medical risk estimate.

## 3:45-4:30 -- Kubernetes deployment

Only claim this step completed if the commands really show a ready deployment:

```powershell
kubectl --context docker-desktop get deployment,pods,svc -l app=heart-api
kubectl --context docker-desktop rollout status deployment/heart-api
.\scripts\demo_api.ps1 -BaseUrl http://localhost:8000
```

Show `deploy/k8s/deployment.yaml`: Deployment, resource limits, readiness/liveness probes, and LoadBalancer service.

If Docker Desktop Kubernetes is not enabled or not ready, state that deployment remains pending. Showing a manifest is not deployment proof.

## 4:30-5:15 -- Monitoring and request logs

Open <http://localhost:8001/monitor>, run `.\scripts\demo_api.ps1` again, and show the success counter and latency count increasing. Show logs:

```powershell
docker logs --tail 12 heart-api-demo
```

Explain that structured request logs record method, route, status and latency without logging patient feature values.

## 5:15-5:40 -- Close

Show the report and repository URL. Mention historical data, small sample size, no calibration or external validation, and the educational-use limitation. State any genuinely unfinished submission items.

After recording: save the video or upload it to your chosen sharing service, verify access, and insert its real link in the report. Capture genuine screenshots with Win+Shift+S if needed.
