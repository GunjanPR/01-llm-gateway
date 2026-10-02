"""Gateway settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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

    # Redis: cache + rate limiting
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_rpm: int = 60
    cache_mode: Literal["all", "deterministic", "off"] = "all"
    cache_ttl_seconds: int = 3600
    # tenant = cached answers are only reused for the same API key; global = shared
    cache_scope: Literal["tenant", "global"] = "tenant"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
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

    # Routing
    default_model: str = "llama3.2"
    routing_policy: Literal["cost", "latency", "quality"] = "cost"
    backend_timeout_seconds: float = 60.0
    backend_max_retries: int = 1
    retry_backoff_seconds: float = 0.2
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_seconds: float = 30.0

    @property
    def api_keys(self) -> set[str]:
        return set(_split(self.gateway_api_keys))

    def backend_configs(self) -> list[BackendConfig]:
        """Backends in priority order; the order breaks ties between equal policy scores."""
        backends: list[BackendConfig] = []
        if self.vllm_base_url:
            backends.append(
                BackendConfig(
                    name="vllm",
                    base_url=f"{self.vllm_base_url.rstrip('/').removesuffix('/v1')}/v1",
                    api_key=self.vllm_api_key,
                    models=parse_models(self.vllm_models),
                    quality=self.vllm_quality,
                    timeout_seconds=self.backend_timeout_seconds,
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
                    timeout_seconds=self.backend_timeout_seconds,
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
                    timeout_seconds=self.backend_timeout_seconds,
                )
            )
        return backends


@lru_cache
def get_settings() -> Settings:
    return Settings()
