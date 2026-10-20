# Recording-ready local operations

Run commands from the repository root in PowerShell.

## MLflow

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000/#/experiments/144320529290097270>.
Filter with `tags.recording_demo = 'ready'` to show the four verified recording runs.
The final run contains the model signature, input example, tuning results, ROC curve and confusion matrix. Candidate runs contain parameters, fold metrics and out-of-fold plots.
Leave an already-running MLflow server alone; do not start a second process on the same port.

## Docker

Build and automatically test the isolated image:

```powershell
.\scripts\verify_container.ps1
```

If the Docker network cannot complete TLS connections to PyPI, use a trusted HTTPS index that your network permits. The current environment supports:

```powershell
.\scripts\verify_container.ps1 -PackageIndex https://packagefeedproxy.microsoft.io/pypi/simple/
```

This changes only the package source; pinned versions remain unchanged. Never disable TLS verification or put credentials into a build argument.

After a successful build, leave a recording container on port 8001:

```powershell
docker run --detach --name heart-api-demo -p 127.0.0.1:8001:8000 heart-mlops:local
.\scripts\demo_api.ps1
```

Open <http://localhost:8001/monitor>. `/health` returns 503 if the model is missing. `/predict` accepts the original UCI category codes in `scripts/sample_patient.json`. `/metrics` exposes outcomes and latency. Request logs omit patient feature values.

## Docker Desktop Kubernetes

1. In Docker Desktop, enable Kubernetes and wait until its status is running. This is a required manual step if no Kubernetes context exists.
2. Verify the context before applying anything:

```powershell
kubectl config get-contexts
kubectl --context docker-desktop get nodes
kubectl --context docker-desktop apply -f .\deploy\k8s\deployment.yaml
kubectl --context docker-desktop rollout status deployment/heart-api --timeout=180s
kubectl --context docker-desktop get deployment,pods,svc -l app=heart-api
.\scripts\demo_api.ps1 -BaseUrl http://localhost:8000
```

The manifest uses a LoadBalancer service on port 8000, with pod port 8000. Docker Desktop normally exposes it at localhost. This is separate from the recording Docker container on port 8001. If the LoadBalancer address remains pending, inspect `kubectl --context docker-desktop describe svc heart-api`; do not claim ingress access from a pod-only test.

If the cluster cannot find `heart-mlops:local`, inspect the pod events. Docker Desktop image-store configuration can affect local-image visibility. Load the image into the cluster's supported image store or use an approved registry; do not silently substitute a different image.

For Minikube instead, load the image with `minikube image load heart-mlops:local`, apply the same manifest, and use `minikube tunnel` to expose its LoadBalancer. A public release needs registry, authentication, TLS, and account-specific configuration.

## Evidence and cleanup

Capture the real MLflow comparison, model artifacts, Docker container/API, dashboard, successful Actions run, and Kubernetes ready status. The previous failed CI screenshot is historical evidence, not a success.

After recording, stop only these project resources:

```powershell
docker stop heart-api-demo
docker rm heart-api-demo
kubectl --context docker-desktop delete -f .\deploy\k8s\deployment.yaml
```

Stop the MLflow process separately when it is no longer needed. Never publish patient information or credentials.
