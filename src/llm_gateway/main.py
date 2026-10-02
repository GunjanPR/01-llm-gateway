"""FastAPI app: the OpenAI-compatible gateway API."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from llm_gateway import __version__
from llm_gateway.backends import BackendError, OpenAICompatibleBackend
from llm_gateway.cache import ResponseCache, cache_key, is_cacheable
from llm_gateway.config import Settings, get_settings
from llm_gateway.guardrails import GuardrailReport, redact_response, scan_messages
from llm_gateway.ledger import PostgresStore, UsageEvent, UsageLedger, Writer
from llm_gateway.metrics import (
    GatewayMetrics,
    cost_usd,
    estimate_tokens,
    prompt_text,
    usage_from,
)
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.router import Backend, NoBackendAvailable, Router
from llm_gateway.semantic_cache import RedisVectorIndex, SemanticCache, SemanticLookup, VectorIndex
from llm_gateway.tracing import current_trace_id, get_tracer, setup_tracing

logger = logging.getLogger("llm_gateway")

# Routes whose requests are written to the usage ledger, and their ledger names.
LEDGER_ROUTES = {"/v1/chat/completions": "chat", "/v1/embeddings": "embeddings"}


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
    vector_index: VectorIndex | None = None,
    ledger_writer: Writer | None = None,
    span_exporter: Any = None,
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
    semantic = (
        SemanticCache(
            vector_index or RedisVectorIndex(redis),
            router,
            embed_model=settings.semantic_cache_embed_model,
            verify_model=settings.semantic_cache_verify_model,
            threshold=settings.semantic_cache_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
            metrics=metrics,
        )
        if settings.semantic_cache_enabled
        else None
    )
    store = (
        PostgresStore(settings.ledger_database_url)
        if settings.ledger_database_url and ledger_writer is None
        else None
    )
    writer = ledger_writer or (store.write if store else None)
    ledger = (
        UsageLedger(
            writer,
            metrics,
            batch_size=settings.ledger_batch_size,
            flush_interval_seconds=settings.ledger_flush_interval_seconds,
            max_queue=settings.ledger_max_queue,
        )
        if writer
        else None
    )
    api_keys = settings.api_keys
    if not api_keys:
        logger.warning("GATEWAY_API_KEYS is empty: every /v1 request will be rejected")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if store:
            try:
                await store.connect()
            except Exception as exc:  # keep serving; rows queue until the DB is reachable
                logger.warning("usage ledger database unavailable at startup: %s", exc)
        if ledger:
            ledger.start()
        yield
        if ledger:
            await ledger.stop()
        if store:
            await store.close()
        for backend in backends:
            await backend.aclose()
        await redis.aclose()

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.router = router
    app.state.metrics = metrics
    app.state.ledger = ledger

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        """Tag every response with a request id and write its usage row."""
        request.state.request_id = uuid.uuid4()
        trace_id = current_trace_id()
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = str(request.state.request_id)
        if trace_id:
            response.headers["x-trace-id"] = trace_id
        usage = getattr(request.state, "usage", None)
        if ledger and usage is not None and not usage.pop("deferred", False):
            ledger.record(
                UsageEvent(
                    id=request.state.request_id,
                    status=response.status_code,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    trace_id=trace_id,
                    **usage,
                )
            )
        return response

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
        route = LEDGER_ROUTES.get(request.url.path)
        if route:
            request.state.usage = {"tenant": tenant_id, "route": route, "model": "-"}
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

    def guard_input(payload: dict[str, Any], usage: dict[str, Any]) -> GuardrailReport:
        """Redact PII and check for prompt injection; raise if policy says block."""
        t0 = time.perf_counter()
        with get_tracer().start_as_current_span("guardrails.input") as span:
            messages, report = scan_messages(
                payload["messages"],
                redact=settings.guardrail_pii != "off",
                check_injection=settings.guardrail_injection != "off",
            )
            if report.header:
                span.set_attribute("llmgw.guardrails", report.header)
        metrics.guardrail_seconds.labels("input").observe(time.perf_counter() - t0)
        usage["guardrails"] = report.header or None
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
    async def chat_completions(request: Request, tenant_id: str = Depends(tenant)) -> Response:
        payload = await read_payload(request)
        model = payload["model"]
        usage = request.state.usage
        usage["model"] = model
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError(400, "'messages' must be a non-empty list", "invalid_request_error")
        report = guard_input(payload, usage)
        if payload.get("stream"):
            return await stream_chat(request, payload, model, report)

        start = time.perf_counter()
        bypass = "no-cache" in request.headers.get("cache-control", "").lower()
        cacheable = is_cacheable(payload, settings.cache_mode) and not bypass
        namespace = tenant_id if settings.cache_scope == "tenant" else ""
        key = cache_key(payload, namespace) if cacheable else None
        hit = None
        if key:
            with get_tracer().start_as_current_span("cache.exact.lookup") as span:
                hit = await cache.get(key)
                span.set_attribute("llmgw.cache.hit", hit is not None)
        if hit:
            metrics.cache.labels("hit").inc()
            metrics.requests.labels("chat", model, "cache_hit").inc()
            metrics.saved_tokens.labels(model).inc(hit["prompt_tokens"] + hit["completion_tokens"])
            metrics.saved_cost.labels(model).inc(hit["cost_usd"])
            elapsed = time.perf_counter() - start
            metrics.latency.labels("chat", model, hit["backend"], "hit").observe(elapsed)
            usage.update(backend=hit["backend"], cache="HIT")
            _annotate_span(usage)
            return JSONResponse(
                hit["response"],
                headers=_headers(hit["backend"], "HIT", 0.0, elapsed, report),
            )
        if key:
            metrics.cache.labels("miss").inc()
        lookup = SemanticLookup()
        if key and semantic:
            lookup = await semantic.lookup(payload, namespace)
            if lookup.entry:
                hit = lookup.entry
                metrics.requests.labels("chat", model, "semantic_hit").inc()
                metrics.saved_tokens.labels(model).inc(
                    hit["prompt_tokens"] + hit["completion_tokens"]
                )
                metrics.saved_cost.labels(model).inc(hit["cost_usd"])
                elapsed = time.perf_counter() - start
                metrics.latency.labels("chat", model, hit["backend"], "semantic_hit").observe(
                    elapsed
                )
                headers = _headers(hit["backend"], "SEMANTIC_HIT", 0.0, elapsed, report)
                headers["x-gateway-similarity"] = f"{lookup.similarity:.4f}"
                usage.update(backend=hit["backend"], cache="SEMANTIC_HIT")
                _annotate_span(usage)
                return JSONResponse(hit["response"], headers=headers)

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
            entry = {
                "response": response,
                "backend": backend.name,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost_usd": cost,
            }
            await cache.set(key, entry)
            if semantic:
                await semantic.store(lookup, entry)
        end = time.perf_counter()
        elapsed = end - start
        metrics.overhead.labels("chat").observe(end - request.state.received - upstream)
        cache_state = "MISS" if key else "BYPASS"
        metrics.requests.labels("chat", model, "ok").inc()
        metrics.latency.labels("chat", model, backend.name, cache_state.lower()).observe(elapsed)
        usage.update(
            backend=backend.name,
            cache=cache_state,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
        )
        _annotate_span(usage)
        return JSONResponse(
            response, headers=_headers(backend.name, cache_state, cost, elapsed, report)
        )

    async def stream_chat(
        request: Request, payload: dict[str, Any], model: str, report: GuardrailReport
    ) -> StreamingResponse:
        start = time.perf_counter()
        # Ask OpenAI-compatible servers to append a usage chunk at the end of the stream.
        payload = {**payload, "stream_options": {"include_usage": True}}
        try:
            backend, lines, ttft = await router.open_stream(model, payload)
        except (BackendError, NoBackendAvailable) as exc:
            raise upstream_failure("chat_stream", model, exc) from exc
        metrics.ttft.labels(model, backend.name).observe(ttft)
        # The row is written when the stream ends, once token counts are known.
        row = request.state.usage
        row["deferred"] = True
        row["trace_id"] = current_trace_id()

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
                cost = record_usage(model, backend, prompt, completion)
                if ledger:
                    ledger.record(
                        UsageEvent(
                            id=request.state.request_id,
                            tenant=row["tenant"],
                            route="chat_stream",
                            model=model,
                            backend=backend.name,
                            cache="BYPASS",
                            status=200 if outcome == "ok" else 502,
                            prompt_tokens=prompt,
                            completion_tokens=completion,
                            cost_usd=cost,
                            latency_ms=(time.perf_counter() - request.state.received) * 1000,
                            guardrails=row.get("guardrails"),
                            trace_id=row.get("trace_id"),
                        )
                    )
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
        request.state.usage["model"] = model
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
        request.state.usage.update(
            backend=backend.name, cache="BYPASS", prompt_tokens=prompt, cost_usd=cost
        )
        return JSONResponse(response, headers=_headers(backend.name, "BYPASS", cost, elapsed))

    setup_tracing(
        app,
        clients=[b._client for b in backends if hasattr(b, "_client")],
        endpoint=settings.otel_exporter_otlp_endpoint,
        service_name=settings.otel_service_name,
        exporter=span_exporter,
    )

    @app.get("/v1/usage")
    async def usage_summary(tenant_id: str = Depends(tenant), days: int = 30) -> dict[str, Any]:
        """The caller's own usage over the last `days` days, per model."""
        if store is None:
            raise GatewayError(503, "usage ledger is not configured", "ledger_unavailable")
        since = datetime.now(UTC) - timedelta(days=max(1, min(days, 366)))
        if ledger:
            await ledger.flush()  # include requests made a moment ago
        try:
            rows = await store.summary(tenant_id, since)
        except Exception as exc:
            raise GatewayError(
                503, f"usage ledger unavailable: {exc}", "ledger_unavailable"
            ) from exc
        totals = {
            k: sum(r[k] for r in rows)
            for k in (
                "requests",
                "errors",
                "cache_hits",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
            )
        }
        return {"tenant": tenant_id, "since": since.isoformat(), "totals": totals, "models": rows}

    return app


def _annotate_span(usage: dict[str, Any]) -> None:
    """Put the request outcome on the server span (GenAI semantic conventions)."""
    span = trace.get_current_span()
    span.set_attribute("gen_ai.request.model", usage.get("model", "-"))
    span.set_attribute("gen_ai.usage.input_tokens", usage.get("prompt_tokens", 0))
    span.set_attribute("gen_ai.usage.output_tokens", usage.get("completion_tokens", 0))
    span.set_attribute("llmgw.cache", usage.get("cache") or "-")
    span.set_attribute("llmgw.backend", usage.get("backend") or "-")
    span.set_attribute("llmgw.tenant", usage.get("tenant", "-"))
    span.set_attribute("llmgw.cost_usd", usage.get("cost_usd", 0.0))


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
