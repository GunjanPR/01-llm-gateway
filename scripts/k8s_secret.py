"""Create/update the `llm-gateway-secrets` Kubernetes Secret from your local .env.

The Secret is piped straight to `kubectl apply -f -`, so secret values are never written
to disk or printed; only the key names are shown.

    python scripts/k8s_secret.py            # apply to the current kubectl context
    python scripts/k8s_secret.py --kubectl "<path to kubectl.exe>"
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

KEYS = ["GATEWAY_API_KEYS", "POSTGRES_PASSWORD", "HOSTED_API_KEY", "VLLM_API_KEY"]


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--namespace", default="llm-gateway")
    parser.add_argument("--kubectl", default="kubectl")
    args = parser.parse_args()

    env = read_env(Path(args.env_file))
    data = {k: env[k] for k in KEYS if env.get(k)}
    for required in ("GATEWAY_API_KEYS", "POSTGRES_PASSWORD"):
        if required not in data:
            raise SystemExit(f"{required} is missing from {args.env_file}")
    # Inside the cluster the database is the `postgres` Service, not localhost.
    data["LEDGER_DATABASE_URL"] = (
        f"postgresql://llmgw:{data['POSTGRES_PASSWORD']}@postgres:5432/llmgw"
    )
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "llm-gateway-secrets", "namespace": args.namespace},
        "type": "Opaque",
        "data": {k: base64.b64encode(v.encode()).decode() for k, v in data.items()},
    }
    subprocess.run(
        [args.kubectl, "apply", "-f", "-"], input=json.dumps(secret).encode(), check=True
    )
    print("secret keys:", ", ".join(sorted(data)))


if __name__ == "__main__":
    main()
