"""Prometheus metrics, token accounting and cost."""

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from llm_gateway.config import BackendConfig

OVERHEAD_BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60)


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "llmgw_requests_total",
            "Requests by route, model and outcome",
            ["route", "model", "outcome"],
            registry=r,
        )
        self.latency = Histogram(
            "llmgw_request_latency_seconds",
            "End-to-end request latency seen by the client",
            ["route", "model", "backend", "cache"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.upstream_latency = Histogram(
            "llmgw_upstream_latency_seconds",
            "Time spent waiting on the backend (successful calls)",
            ["route", "backend"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.overhead = Histogram(
            "llmgw_gateway_overhead_seconds",
            "Time the gateway adds per request: total handling time minus backend wait",
            ["route"],
            buckets=OVERHEAD_BUCKETS,
            registry=r,
        )
        self.guardrail_events = Counter(
            "llmgw_guardrail_events_total",
            "PII entities and prompt-injection rules triggered",
            ["direction", "kind", "action"],
            registry=r,
        )
        self.guardrail_seconds = Histogram(
            "llmgw_guardrail_seconds",
            "Time spent in guardrail checks",
            ["direction"],
            buckets=OVERHEAD_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "llmgw_ttft_seconds",
            "Time to first token for streaming requests",
            ["model", "backend"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "llmgw_tokens_total",
            "Tokens processed by backends",
            ["model", "backend", "kind"],
            registry=r,
        )
        self.tokens_per_second = Histogram(
            "llmgw_completion_tokens_per_second",
            "Completion tokens per second of generation",
            ["model", "backend"],
            buckets=(1, 5, 10, 20, 40, 60, 80, 100, 150, 200, 400),
            registry=r,
        )
        self.cost = Counter(
            "llmgw_cost_usd_total", "Estimated spend in USD", ["model", "backend"], registry=r
        )
        self.saved_cost = Counter(
            "llmgw_cache_saved_cost_usd_total",
            "Estimated spend avoided by cache hits",
            ["model"],
            registry=r,
        )
        self.saved_tokens = Counter(
            "llmgw_cache_saved_tokens_total",
            "Tokens not regenerated because of cache hits",
            ["model"],
            registry=r,
        )
        self.semantic_cache = Counter(
            "llmgw_semantic_cache_total",
            "Semantic cache lookups by outcome (hit, no_candidate, rejected, error)",
            ["outcome"],
            registry=r,
        )
        self.semantic_similarity = Histogram(
            "llmgw_semantic_similarity",
            "Cosine similarity of the nearest cached prompt, when above threshold",
            buckets=(0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0),
            registry=r,
        )
        self.semantic_seconds = Histogram(
            "llmgw_semantic_cache_seconds",
            "Time per semantic cache stage",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ledger_rows = Counter(
            "llmgw_ledger_rows_total", "Usage rows written to PostgreSQL", registry=r
        )
        self.ledger_errors = Counter(
            "llmgw_ledger_write_errors_total", "Failed ledger batch writes", registry=r
        )
        self.ledger_dropped = Counter(
            "llmgw_ledger_dropped_total",
            "Usage rows dropped because the queue was full",
            registry=r,
        )
        self.cache = Counter("llmgw_cache_requests_total", "Cache lookups", ["result"], registry=r)
        self.backend_errors = Counter(
            "llmgw_backend_errors_total", "Failed backend calls", ["backend", "kind"], registry=r
        )
        self.rate_limited = Counter(
            "llmgw_rate_limited_total", "Requests rejected by the rate limiter", registry=r
        )
        self.backend_up = Gauge(
            "llmgw_backend_available",
            "1 if the backend's circuit breaker is closed",
            ["backend"],
            registry=r,
        )


def usage_from(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) when a backend does not report usage."""
    return max(1, len(text) // 4) if text else 0


def prompt_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(parts)


def cost_usd(config: BackendConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1000 * config.cost_per_1k_input
        + completion_tokens / 1000 * config.cost_per_1k_output
    )
