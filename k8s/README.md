# Kubernetes deployment

Kustomize manifests for a single-node local cluster (Docker Desktop Kubernetes or kind).
They deploy the gateway, Redis 8, PostgreSQL 16 and Jaeger. Ollama stays on the host, so it
can use the GPU directly, and pods reach it at `host.docker.internal:11434`.

| Object | Notes |
|---|---|
| `Deployment/gateway` | 2 replicas; rolling update with `maxUnavailable: 0`; readiness `/readyz` (Redis reachable), liveness `/livez`; non-root UID 10001; read-only root filesystem; all capabilities dropped; CPU and memory requests and limits |
| `HorizontalPodAutoscaler/gateway` | 2–6 replicas at 70% CPU (needs metrics-server) |
| `PodDisruptionBudget/gateway` | keeps at least 1 pod up during node drains |
| `StatefulSet/redis`, `StatefulSet/postgres` | persistent volumes; readiness via `redis-cli ping` / `pg_isready` |
| `Deployment/jaeger` | OTLP/HTTP on 4318, UI on 16686 |
| `ConfigMap/llm-gateway-config` | all non-secret settings |
| `Secret/llm-gateway-secrets` | **not in git**; created from your `.env` by `scripts/k8s_secret.py` |

## Deploy

Prerequisites: a local cluster, either Docker Desktop's (Settings → Kubernetes → Enable
Kubernetes, Kubeadm) or kind (`kind create cluster --name llmgw`, then
`kind load docker-image llm-gateway:local --name llmgw` after the build below), and Ollama
running on the host.

```bash
docker build -t llm-gateway:local .                 # the cluster uses the local image
kubectl apply -f k8s/namespace.yaml
python scripts/k8s_secret.py                        # Secret from .env; values never printed
kubectl apply -k k8s/
kubectl -n llm-gateway rollout status deploy/gateway
kubectl -n llm-gateway port-forward svc/gateway 8000:8000   # then use http://localhost:8000
```

On Windows, if `kubectl` isn't on your PATH, use
`"C:\Program Files\Docker\Docker\resources\bin\kubectl.exe"`.

To tear it down: `kubectl delete -k k8s/`. This keeps the PersistentVolumeClaims; to remove
the data too, delete them with `kubectl -n llm-gateway delete pvc --all`.

## Verified (kind v0.24.0 on Docker Desktop, 2026-10-19)

- Both gateway pods became ready; `/livez` and `/readyz` returned `ok` through the Service.
- A chat completion through `svc/gateway` was served by Ollama on the host
  (`x-gateway-backend: ollama`), with `x-ratelimit-*` headers from the shared Redis.
- During `kubectl rollout restart deploy/gateway`, a pod in the cluster called
  `http://gateway:8000/readyz` every 0.1 s for 75 s: 677 of 677 calls succeeded.

The first deploy crash-looped. Kubernetes adds Docker-link style variables for every Service
in the namespace, so the `gateway` Service produced `GATEWAY_PORT=tcp://10.96.x.x:8000`, which
the gateway's `GATEWAY_PORT` setting could not parse. The pod spec now sets
`enableServiceLinks: false`. Schema validation passes either way, so CI also deploys the
manifests to a kind cluster (`k8s-deploy` job).
