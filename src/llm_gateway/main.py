"""FastAPI app: the OpenAI-compatible gateway API."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from llm_gateway import __version__
from llm_gateway.backends import BackendError, OpenAICompatibleBackend
from llm_gateway.cache import ResponseCache, cache_key, is_cacheable
from llm_gateway.config import Settings, get_settings
from llm_gateway.guardrails import GuardrailReport, redact_response, scan_messages
from llm_gateway.metrics import (
    GatewayMetrics,
    cost_usd,
    estimate_tokens,
    prompt_text,
    usage_from,
)
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.router import Backend, NoBackendAvailable, Router

logger = logging.getLogger("llm_gateway")


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.headers = headers or {}


def _error_body(message: str, error_type: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": None}}


def create_app(
    settings: Settings | None = None,
    backends: list[Backend] | None = None,
    redis: Redis | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    if backends is None:
        backends = [OpenAICompatibleBackend(cfg) for cfg in settings.backend_configs()]
    if not backends:
        logger.warning("no backends configured: set OLLAMA_BASE_URL, VLLM_BASE_URL or HOSTED_*")
    redis = redis or Redis.from_url(
        settings.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
    )
    metrics = GatewayMetrics()
    router = Router(
        backends,
        policy=settings.routing_policy,
        max_retries=settings.backend_max_retries,
        retry_backoff_seconds=settings.retry_backoff_seconds,
        breaker_failures=settings.circuit_breaker_failures,
        breaker_cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        on_backend_error=lambda e: metrics.backend_errors.labels(e.backend, e.kind).inc(),
    )
    cache = ResponseCache(redis, settings.cache_ttl_seconds)
    limiter = RateLimiter(redis, settings.rate_limit_rpm)
    api_keys = settings.api_keys
    if not api_keys:
        logger.warning("GATEWAY_API_KEYS is empty: every /v1 request will be rejected")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        for backend in backends:
            await backend.aclose()
        await redis.aclose()

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.router = router
    app.state.metrics = metrics

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            _error_body(exc.message, exc.error_type), exc.status_code, headers=exc.headers
        )

    async def tenant(request: Request) -> str:
        """Authenticate the bearer key and enforce the tenant's rate limit."""
        request.state.received = time.perf_counter()
        header = request.headers.get("authorization", "")
        scheme, _, key = header.partition(" ")
        if scheme.lower() != "bearer" or not any(
            hmac.compare_digest(key.encode(), k.encode()) for k in api_keys
        ):
            raise GatewayError(401, "invalid or missing API key", "authentication_error")
        tenant_id = hashlib.sha256(key.encode()).hexdigest()[:12]
        result = await limiter.hit(tenant_id)
        if not result.allowed:
            metrics.rate_limited.inc()
            raise GatewayError(
                429,
                f"rate limit of {result.limit} requests/minute exceeded",
                "rate_limit_error",
                headers={"Retry-After": str(result.reset_seconds)},
            )
        return tenant_id

    async def read_payload(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise GatewayError(400, "request body must be JSON", "invalid_request_error") from exc
        if not isinstance(payload, dict):
            raise GatewayError(400, "request body must be a JSON object", "invalid_request_error")
        payload.setdefault("model", settings.default_model)
        return payload

    def upstream_failure(route: str, model: str, exc: Exception) -> GatewayError:
        metrics.requests.labels(route, model, "error").inc()
        if isinstance(exc, BackendError):  # client error passed through from the backend
            return GatewayError(exc.status_code or 400, exc.message, "invalid_request_error")
        status = 404 if isinstance(exc, NoBackendAvailable) and not exc.errors else 502
        return GatewayError(status, str(exc), "upstream_error")

    def guard_input(payload: dict[str, Any]) -> GuardrailReport:
        """Redact PII and check for prompt injection; raise if policy says block."""
        t0 = time.perf_counter()
        messages, report = scan_messages(
            payload["messages"],
            redact=settings.guardrail_pii != "off",
            check_injection=settings.guardrail_injection != "off",
        )
        metrics.guardrail_seconds.labels("input").observe(time.perf_counter() - t0)
        for kind, n in report.pii.items():
            metrics.guardrail_events.labels(
                "input", f"pii_{kind.lower()}", settings.guardrail_pii
            ).inc(n)
        for rule in report.injection:
            metrics.guardrail_events.labels(
                "input", f"injection_{rule}", settings.guardrail_injection
            ).inc()
        if report.pii and settings.guardrail_pii == "block":
            kinds = ", ".join(sorted(report.pii))
            raise GatewayError(400, f"request contains PII ({kinds})", "guardrail_violation")
        if report.injection and settings.guardrail_injection == "block":
            rules = ", ".join(report.injection)
            raise GatewayError(400, f"possible prompt injection ({rules})", "guardrail_violation")
        payload["messages"] = messages
        return report

    def guard_output(response: dict[str, Any], report: GuardrailReport) -> dict[str, Any]:
        if settings.guardrail_output_pii == "off":
            return response
        t0 = time.perf_counter()
        response, found = redact_response(response)
        metrics.guardrail_seconds.labels("output").observe(time.perf_counter() - t0)
        for kind, n in found.items():
            metrics.guardrail_events.labels("output", f"pii_{kind.lower()}", "redact").inc(n)
            report.pii[f"OUT_{kind}"] += n
        return response

    def record_usage(model: str, backend: Backend, prompt: int, completion: int) -> float:
        metrics.tokens.labels(model, backend.name, "prompt").inc(prompt)
        metrics.tokens.labels(model, backend.name, "completion").inc(completion)
        cost = cost_usd(backend.config, prompt, completion)
        metrics.cost.labels(model, backend.name).inc(cost)
        return cost

    def refresh_backend_gauges() -> None:
        for backend in backends:
            metrics.backend_up.labels(backend.name).set(1 if router.is_available(backend) else 0)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness plus a quick, concurrent probe of Redis and every backend."""
        refresh_backend_gauges()

        async def redis_ok() -> bool:
            try:
                return bool(await redis.ping())
            except Exception:  # report Redis as down instead of failing the health check
                return False

        redis_up, *reachable = await asyncio.gather(redis_ok(), *(b.health() for b in backends))
        return {
            "status": "ok",
            "version": __version__,
            "redis": redis_up,
            "backends": [
                {
                    "name": b.name,
                    "reachable": up,
                    "circuit_open": not router.is_available(b),
                    "models": list(b.config.models),
                }
                for b, up in zip(backends, reachable, strict=True)
            ],
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        refresh_backend_gauges()
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models", dependencies=[Depends(tenant)])
    async def list_models() -> dict[str, Any]:
        names = sorted({m for b in backends for m in b.config.models if m != "*"})
        if settings.default_model not in names:
            names.insert(0, settings.default_model)
        return {
            "object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "llm-gateway"} for n in names],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, _: str = Depends(tenant)) -> Response:
        payload = await read_payload(request)
        model = payload["model"]
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError(400, "'messages' must be a non-empty list", "invalid_request_error")
        report = guard_input(payload)
        if payload.get("stream"):
            return await stream_chat(payload, model, report)

        start = time.perf_counter()
        bypass = "no-cache" in request.headers.get("cache-control", "").lower()
        cacheable = is_cacheable(payload, settings.cache_mode) and not bypass
        key = cache_key(payload) if cacheable else None
        if key and (hit := await cache.get(key)):
            metrics.cache.labels("hit").inc()
            metrics.requests.labels("chat", model, "cache_hit").inc()
            metrics.saved_tokens.labels(model).inc(hit["prompt_tokens"] + hit["completion_tokens"])
            metrics.saved_cost.labels(model).inc(hit["cost_usd"])
            elapsed = time.perf_counter() - start
            metrics.latency.labels("chat", model, hit["backend"], "hit").observe(elapsed)
            return JSONResponse(
                hit["response"],
                headers=_headers(hit["backend"], "HIT", 0.0, elapsed, report),
            )
        if key:
            metrics.cache.labels("miss").inc()

        routed = time.perf_counter()
        try:
            response, backend = await router.run(model, lambda b: b.chat(payload))
        except (BackendError, NoBackendAvailable) as exc:
            raise upstream_failure("chat", model, exc) from exc
        upstream = time.perf_counter() - routed
        metrics.upstream_latency.labels("chat", backend.name).observe(upstream)

        response = guard_output(response, report)
        prompt, completion = usage_from(response)
        if not response.get("usage"):
            prompt = estimate_tokens(prompt_text(payload))
            completion = estimate_tokens(_completion_text(response))
        cost = record_usage(model, backend, prompt, completion)
        if completion and upstream > 0:
            metrics.tokens_per_second.labels(model, backend.name).observe(completion / upstream)
        if key:
            await cache.set(
                key,
                {
                    "response": response,
                    "backend": backend.name,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "cost_usd": cost,
                },
            )
        end = time.perf_counter()
        elapsed = end - start
        metrics.overhead.labels("chat").observe(end - request.state.received - upstream)
        cache_state = "MISS" if key else "BYPASS"
        metrics.requests.labels("chat", model, "ok").inc()
        metrics.latency.labels("chat", model, backend.name, cache_state.lower()).observe(elapsed)
        return JSONResponse(
            response, headers=_headers(backend.name, cache_state, cost, elapsed, report)
        )

    async def stream_chat(
        payload: dict[str, Any], model: str, report: GuardrailReport
    ) -> StreamingResponse:
        start = time.perf_counter()
        # Ask OpenAI-compatible servers to append a usage chunk at the end of the stream.
        payload = {**payload, "stream_options": {"include_usage": True}}
        try:
            backend, lines, ttft = await router.open_stream(model, payload)
        except (BackendError, NoBackendAvailable) as exc:
            raise upstream_failure("chat_stream", model, exc) from exc
        metrics.ttft.labels(model, backend.name).observe(ttft)

        async def body() -> AsyncIterator[bytes]:
            text: list[str] = []
            usage: dict[str, Any] = {}
            outcome = "ok"
            try:
                async for line in lines:
                    data = line.removeprefix("data:").strip()
                    if data and data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            chunk = {}
                        usage = chunk.get("usage") or usage
                        for choice in chunk.get("choices") or []:
                            text.append((choice.get("delta") or {}).get("content") or "")
                    yield f"{line}\n\n".encode()
            except BackendError as exc:
                outcome = "error"
                metrics.backend_errors.labels(exc.backend, exc.kind).inc()
                error = json.dumps(_error_body(exc.message, "upstream_error"))
                yield f"data: {error}\n\n".encode()
            finally:
                elapsed = time.perf_counter() - start
                prompt = int(usage.get("prompt_tokens") or estimate_tokens(prompt_text(payload)))
                completion = int(usage.get("completion_tokens") or estimate_tokens("".join(text)))
                record_usage(model, backend, prompt, completion)
                if completion and elapsed > ttft:
                    metrics.tokens_per_second.labels(model, backend.name).observe(
                        completion / (elapsed - ttft)
                    )
                metrics.requests.labels("chat_stream", model, outcome).inc()
                metrics.latency.labels("chat_stream", model, backend.name, "bypass").observe(
                    elapsed
                )

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "x-gateway-backend": backend.name,
                "x-gateway-cache": "BYPASS",
                **({"x-gateway-guardrails": report.header} if report.header else {}),
            },
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, _: str = Depends(tenant)) -> Response:
        payload = await read_payload(request)
        model = payload["model"]
        if "input" not in payload:
            raise GatewayError(400, "'input' is required", "invalid_request_error")
        start = time.perf_counter()
        try:
            response, backend = await router.run(model, lambda b: b.embeddings(payload))
        except (BackendError, NoBackendAvailable) as exc:
            raise upstream_failure("embeddings", model, exc) from exc
        prompt, _ = usage_from(response)
        cost = record_usage(model, backend, prompt, 0)
        elapsed = time.perf_counter() - start
        metrics.upstream_latency.labels("embeddings", backend.name).observe(elapsed)
        metrics.requests.labels("embeddings", model, "ok").inc()
        metrics.latency.labels("embeddings", model, backend.name, "bypass").observe(elapsed)
        return JSONResponse(response, headers=_headers(backend.name, "BYPASS", cost, elapsed))

    return app


def _completion_text(response: dict[str, Any]) -> str:
    return "".join(
        (choice.get("message") or {}).get("content") or "" for choice in response.get("choices", [])
    )


def _headers(
    backend: str, cache: str, cost: float, elapsed: float, report: GuardrailReport | None = None
) -> dict[str, str]:
    headers = {
        "x-gateway-backend": backend,
        "x-gateway-cache": cache,
        "x-gateway-cost-usd": f"{cost:.8f}",
        "x-gateway-latency-ms": f"{elapsed * 1000:.1f}",
    }
    if report and report.header:
        headers["x-gateway-guardrails"] = report.header
    return headers
