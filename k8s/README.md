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

Prerequisites: Docker Desktop with Kubernetes enabled (Settings → Kubernetes → Enable
Kubernetes, Kubeadm), and Ollama running on the host.

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
