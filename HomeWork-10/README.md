# ML Zoomcamp 2026 - Module 10: Kubernetes (Homework 10)

Deploys the 2026 lead-scoring API (FastAPI + scikit-learn) to a local
Kubernetes cluster created with `kind`. Files are copied unchanged from the
official 2026 cohort folders:

- `cohorts/2026/homework/05-deployment` -> `05-deployment/`
- `cohorts/2026/homework/10-kubernetes` -> `10-kubernetes/`

The layout is kept the same as the official repo, so the homework commands
(`cd ../10-kubernetes`, `python ../05-deployment/q6_test.py`) work as written.

## Structure

    mlzoomcamp-module10/
    |-- 05-deployment/            # the API image (Homework 5 artifact)
    |   |-- Dockerfile            # python 3.11.15-slim + uv, port 9696
    |   |-- predict.py            # FastAPI: GET /health, POST /predict
    |   |-- model.py              # missing-value handling
    |   |-- pipeline.bin          # frozen model (DictVectorizer + LogisticRegression)
    |   |-- feature_defaults.json
    |   |-- model_metadata.json
    |   |-- pyproject.toml, uv.lock, .python-version
    |   |-- q6_test.py            # client used in the homework
    |   `-- smoke_test.py
    |-- 10-kubernetes/
    |   |-- deployment.yaml       # name: subscription, port 9696, /health probe
    |   |-- service.yaml          # ClusterIP, 80 -> 9696
    |   `-- hpa.yaml              # 1 to 3 replicas
    |-- .gitignore
    `-- README.md

## Requirements

Docker, `kubectl`, `kind`, Python 3 (to run `q6_test.py`, standard library only).

## 1. Build and test the image locally

    cd 05-deployment
    docker build -t zoomcamp-model:2026-hw10 .
    docker run --rm -p 9696:9696 zoomcamp-model:2026-hw10

Second terminal:

    python q6_test.py
    curl -s http://localhost:9696/health

Expected: `{'conversion_probability': 0.769799, 'conversion': True}` and a
health response with `"status":"ok"` and model checksum
`1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`.

Stop the container (Ctrl+C) before the Kubernetes part.

## 2. Create the cluster and deploy

    cd ../10-kubernetes
    kind --version
    kubectl version --client
    kind create cluster --name mlzoomcamp-2026
    kubectl cluster-info --context kind-mlzoomcamp-2026
    kubectl get services --context kind-mlzoomcamp-2026

    kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026

    kubectl apply -f deployment.yaml --context kind-mlzoomcamp-2026
    kubectl rollout status deployment/subscription --context kind-mlzoomcamp-2026
    kubectl get pods --context kind-mlzoomcamp-2026

    kubectl apply -f service.yaml --context kind-mlzoomcamp-2026
    kubectl get service subscription --context kind-mlzoomcamp-2026

## 3. Test through Kubernetes

    kubectl port-forward service/subscription 9696:80 --context kind-mlzoomcamp-2026

Second terminal:

    python ../05-deployment/q6_test.py

The probability must match step 1 (within 0.005).

## 4. Autoscaling

    kubectl apply -f hpa.yaml --context kind-mlzoomcamp-2026
    kubectl get hpa subscription-hpa --context kind-mlzoomcamp-2026

`TARGETS` may show `<unknown>` without metrics-server. This is expected and
is not part of the graded answers.

## 5. Clean up

    kind delete cluster --name mlzoomcamp-2026

## Homework 10 answers

| Question | Answer |
|---|---|
| 1. conversion_probability (3 decimals) | **0.770** (API returns 0.769799) |
| 2. Environment check | record the output of `kind --version` and `kubectl version --client` (not graded) |
| 3. Smallest deployable unit | **Pod** |
| 4. TYPE of the `kubernetes` service | **ClusterIP** |
| 5. Command that loads the image | **kind load docker-image** |
| 6. Container port | **9696** |
| 7. Selector | **app: subscription** |
| 8. maxReplicas in hpa.yaml | **3** |

Submit at: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw10

## Module 10 lessons and where they appear

| Lesson | Covered by |
|---|---|
| 10.1 Overview | Containers + Kubernetes serving a model (this project) |
| 10.2 TensorFlow Serving | Not used in the 2026 homework (module marks it outdated) |
| 10.3 Pre-processing service | `model.py` + `predict.py` (input cleaning, then model call) |
| 10.4 Docker Compose | Replaced by plain `docker run` in step 1 |
| 10.5 Kubernetes intro | Pods, Deployment, Service, HPA (steps 2 to 4) |
| 10.6 Simple service on Kubernetes | `deployment.yaml` + `service.yaml` |
| 10.7 Models on Kubernetes | Same API, loaded into kind and port-forwarded |
| 10.8 EKS | Not required by the 2026 homework (it uses local kind) |
