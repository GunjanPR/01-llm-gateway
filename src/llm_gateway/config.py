"""Gateway settings, loaded from environment variables and `.env`."""

import logging
from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# URL settings that may point at a service on this machine.
LOOPBACK_URL_FIELDS = (
    "redis_url",
    "ollama_base_url",
    "vllm_base_url",
    "ledger_database_url",
    "otel_exporter_otlp_endpoint",
)


TLS_SCHEMES = {"https", "rediss"}


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ipv4_loopback(url: str) -> str:
    """Replace a `localhost` host with 127.0.0.1, keeping scheme, credentials, port and path.

    On Windows, "localhost" resolves to ::1 first. Ollama and many dev servers listen on
    127.0.0.1 only, and Windows retries a refused TCP connect for about 2 seconds before
    the client moves on to IPv4, so every new connection to "localhost" stalls.
    """
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != "localhost":
        return url
    # TLS certificates name the host: rewriting would break hostname verification
    if parts.scheme in TLS_SCHEMES or "sslmode=verify" in parts.query:
        return url
    userinfo, _, hostport = parts.netloc.rpartition("@")
    hostport = "127.0.0.1" + hostport[len("localhost") :]
    netloc = f"{userinfo}@{hostport}" if userinfo else hostport
    return urlunsplit(parts._replace(netloc=netloc))


def parse_models(value: str) -> dict[str, str]:
    """Parse "llama3.2,fast=llama3.2:1b" into {public name: upstream name}.

    A bare name maps to itself; "*" means the backend accepts any model name as-is.
    """
    models: dict[str, str] = {}
    for item in _split(value):
        public, _, upstream = item.partition("=")
        models[public.strip()] = (upstream or public).strip()
    return models or {"*": "*"}


class BackendConfig(BaseModel):
    """One upstream server that speaks the OpenAI-compatible API (Ollama, vLLM, hosted)."""

    name: str
    base_url: str  # must end in /v1
    api_key: str = ""
    models: dict[str, str] = Field(default_factory=lambda: {"*": "*"})  # public -> upstream
    cost_per_1k_input: float = 0.0  # USD
    cost_per_1k_output: float = 0.0  # USD
    quality: float = 0.5  # 0-1, used by the "quality" routing policy
    timeout_seconds: float = 60.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_seconds: float = 60.0
    http_transport: Literal["httpx", "aiohttp"] = "httpx"

    def serves(self, model: str) -> bool:
        return model in self.models or "*" in self.models

    def upstream_model(self, model: str) -> str:
        """Name to send upstream for a public model name."""
        return self.models.get(model, model)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    gateway_api_keys: str = ""
    log_level: str = "info"
    # text = human-readable lines | json = one JSON object per line (for log shippers)
    log_format: Literal["text", "json"] = "text"
    # Reject request bodies larger than this with 413 (0 = no limit)
    max_request_bytes: int = 4 * 1024 * 1024

    # Rewrite "localhost" to 127.0.0.1 in the URLs below (see ipv4_loopback)
    prefer_ipv4_loopback: bool = True

    # Redis: cache + rate limiting
    redis_url: str = "redis://127.0.0.1:6379/0"
    rate_limit_rpm: int = 60
    cache_mode: Literal["all", "deterministic", "off"] = "all"
    cache_ttl_seconds: int = 3600
    # tenant = cached answers are only reused for the same API key; global = shared
    cache_scope: Literal["tenant", "global"] = "tenant"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_models: str = "*"
    ollama_quality: float = 0.6

    # vLLM
    vllm_base_url: str = ""
    vllm_api_key: str = ""
    vllm_models: str = "*"
    vllm_quality: float = 0.6

    # Optional hosted OpenAI-compatible provider (OpenAI, Groq, Together, ...)
    hosted_name: str = "hosted"
    hosted_base_url: str = ""
    hosted_api_key: str = ""
    hosted_models: str = ""
    hosted_cost_per_1k_input: float = 0.0
    hosted_cost_per_1k_output: float = 0.0
    hosted_quality: float = 0.8

    # Usage ledger (PostgreSQL); blank disables
    ledger_database_url: str = ""
    ledger_batch_size: int = 200
    ledger_flush_interval_seconds: float = 1.0
    ledger_max_queue: int = 10_000

    # Semantic cache (needs Redis 8+ / Redis Stack for vector search)
    semantic_cache_enabled: bool = True
    semantic_cache_embed_model: str = "nomic-embed-text"
    semantic_cache_verify_model: str = "llama3.2"  # blank = no verifier (not recommended)
    semantic_cache_threshold: float = 0.86

    # Guardrails
    guardrail_pii: Literal["off", "redact", "block"] = "redact"
    guardrail_output_pii: Literal["off", "redact"] = "redact"
    guardrail_injection: Literal["off", "flag", "block"] = "flag"

    # Tracing (OTLP/HTTP, e.g. Jaeger at http://127.0.0.1:4318); blank disables
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "llm-gateway"

    # Routing
    default_model: str = "llama3.2"
    routing_policy: Literal["cost", "latency", "quality"] = "cost"
    backend_timeout_seconds: float = 60.0
    # Connection pool per backend. httpx closes idle connections after 5 s by default, so a
    # client pausing for a few seconds paid a fresh TCP connect on its next request.
    backend_max_connections: int = 100
    backend_max_keepalive_connections: int = 20
    backend_keepalive_seconds: float = 60.0
    # aiohttp's connection pool and C parser cost about half the CPU per backend call of
    # httpx's default transport (measured with scripts/mock_upstream.py); the httpx API
    # on top stays the same either way.
    backend_http_transport: Literal["httpx", "aiohttp"] = "aiohttp"
    # Connect to Redis and every backend at startup instead of on the first request
    warm_up_connections: bool = True
    backend_max_retries: int = 1
    retry_backoff_seconds: float = 0.2
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_seconds: float = 30.0
    # Load shedding: at most this many concurrent calls per backend (0 = unlimited). When
    # every backend for a model is full, requests wait up to BACKEND_QUEUE_TIMEOUT_SECONDS
    # for a slot, then get 503 with Retry-After instead of piling up behind the GPU.
    backend_max_concurrency: int = 64
    backend_queue_timeout_seconds: float = 30.0
    # Cancel the backend call when the client disconnects, freeing the GPU and the slot
    cancel_on_disconnect: bool = True
    # Probe every backend and Redis in the background (0 disables); feeds the
    # llmgw_backend_reachable / llmgw_redis_up gauges so outages show without traffic
    health_check_interval_seconds: float = 15.0

    @model_validator(mode="after")
    def _ipv4_loopback(self) -> "Settings":
        if not self.prefer_ipv4_loopback:
            return self
        for field in LOOPBACK_URL_FIELDS:
            urls = _split(getattr(self, field))
            fixed = [ipv4_loopback(u) for u in urls]
            if fixed != urls:
                logger.info("%s: using 127.0.0.1 instead of localhost", field.upper())
                setattr(self, field, ",".join(fixed))
        return self

    @property
    def api_keys(self) -> set[str]:
        return set(_split(self.gateway_api_keys))

    def backend_configs(self) -> list[BackendConfig]:
        """Backends in priority order; the order breaks ties between equal policy scores."""
        backends: list[BackendConfig] = []
        pool = {
            "timeout_seconds": self.backend_timeout_seconds,
            "max_connections": self.backend_max_connections,
            "max_keepalive_connections": self.backend_max_keepalive_connections,
            "keepalive_seconds": self.backend_keepalive_seconds,
            "http_transport": self.backend_http_transport,
        }
        if self.vllm_base_url:
            backends.append(
                BackendConfig(
                    name="vllm",
                    base_url=f"{self.vllm_base_url.rstrip('/').removesuffix('/v1')}/v1",
                    api_key=self.vllm_api_key,
                    models=parse_models(self.vllm_models),
                    quality=self.vllm_quality,
                    **pool,
                )
            )
        # Several Ollama nodes may be listed, comma-separated: ollama, ollama-2, ...
        for i, url in enumerate(_split(self.ollama_base_url), start=1):
            backends.append(
                BackendConfig(
                    name="ollama" if i == 1 else f"ollama-{i}",
                    base_url=f"{url.rstrip('/').removesuffix('/v1')}/v1",
                    models=parse_models(self.ollama_models),
                    quality=self.ollama_quality,
                    **pool,
                )
            )
        if self.hosted_base_url and self.hosted_api_key:
            backends.append(
                BackendConfig(
                    name=self.hosted_name,
                    base_url=self.hosted_base_url.rstrip("/"),
                    api_key=self.hosted_api_key,
                    models=parse_models(self.hosted_models),
                    cost_per_1k_input=self.hosted_cost_per_1k_input,
                    cost_per_1k_output=self.hosted_cost_per_1k_output,
                    quality=self.hosted_quality,
                    **pool,
                )
            )
        return backends


@lru_cache
def get_settings() -> Settings:
    return Settings()
