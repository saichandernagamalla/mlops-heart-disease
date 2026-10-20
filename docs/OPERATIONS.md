# Local API, Docker, and Minikube

## Local API

After `python -m scripts.fetch_data && python -m scripts.train`, run `uvicorn src.api:app --host 0.0.0.0 --port 8000`. Check `GET /health`, send a sample JSON body to `POST /predict`, and inspect Prometheus counters at `GET /metrics`.

## Docker proof

From project root, run `./scripts/verify_container.ps1` in PowerShell. It builds the image, starts the API, checks `/health`, sends the sample patient to `/predict`, checks `/metrics`, prints a JSON summary, and stops the container. Save the successful output and a screenshot for the report. The trained artifact is copied into the image; rebuild after retraining. Port 8000 must be available.

If the build fails while `pip` downloads packages from `pypi.org` or `files.pythonhosted.org`, the build container cannot complete its HTTPS connection. The Dockerfile retries package downloads with a 60-second timeout. If the error persists, check that Docker Desktop can reach both hosts, and configure Docker Desktop's proxy when using a VPN or managed network. Do not disable TLS certificate checks. Then rerun `./scripts/verify_container.ps1`.

## Minikube

1. Train and build: `docker build -t heart-mlops:local .`.
2. Load it: `minikube image load heart-mlops:local`.
3. Apply: `kubectl apply -f deploy/k8s/deployment.yaml`.
4. Wait: `kubectl rollout status deployment/heart-api`; check `kubectl get pods,svc`.
5. Access with `minikube service heart-api --url`, then query `/health` and `/predict`.
6. Save actual screenshots. Remove resources after documenting with `kubectl delete -f deploy/k8s/deployment.yaml`.

The manifest is for a local learning cluster. Public cloud deployment needs a registry, ingress/load-balancer, TLS, and secrets. This demo has request counters and latency but no durable log storage, alerting, or dashboard provisioning. Never publish patient information.
