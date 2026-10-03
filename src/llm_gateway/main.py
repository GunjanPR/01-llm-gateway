"""FastAPI app: the OpenAI-compatible gateway API."""

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway import __version__
from llm_gateway.backends import BackendError, OpenAICompatibleBackend
from llm_gateway.cache import ResponseCache, cache_key, is_cacheable
from llm_gateway.config import Settings, get_settings
from llm_gateway.guardrails import GuardrailReport, redact_response, scan_messages
from llm_gateway.ledger import PostgresStore, UsageEvent, UsageLedger, Writer
from llm_gateway.logs import configure_logging
from llm_gateway.metrics import (
    GatewayMetrics,
    cost_usd,
    estimate_tokens,
    prompt_text,
    usage_from,
)
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.router import Backend, NoBackendAvailable, Overloaded, Router
from llm_gateway.semantic_cache import RedisVectorIndex, SemanticCache, SemanticLookup, VectorIndex
from llm_gateway.tracing import current_trace_id, get_tracer, setup_tracing

logger = logging.getLogger("llm_gateway")
T = TypeVar("T")
# Not sent to anyone (the client is gone); used in logs, metrics and the usage ledger.
CLIENT_CLOSED_STATUS = 499

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


class RequestLogMiddleware:
    """Tag every response with a request id (and trace id) and write its usage row.

    Plain ASGI rather than `@app.middleware("http")`: Starlette's BaseHTTPMiddleware runs
    each request through an extra task group and memory streams, which cost more CPU
    per request than the rest of the gateway's own processing.
    """

    def __init__(self, app: ASGIApp, ledger: UsageLedger | None, metrics: GatewayMetrics) -> None:
        self.app = app
        self.ledger = ledger
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _client_request_id(scope) or uuid.uuid4()
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        trace_id = current_trace_id()
        start = time.perf_counter()
        status = 500

        async def send_with_ids(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["x-request-id"] = str(request_id)
                if trace_id:
                    headers["x-trace-id"] = trace_id
            await send(message)

        self.metrics.inflight.inc()
        try:
            await self.app(scope, receive, send_with_ids)
        finally:
            self.metrics.inflight.dec()
            elapsed_ms = (time.perf_counter() - start) * 1000
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            self.metrics.http_responses.labels(path, f"{status // 100}xx").inc()
            usage = state.get("usage")
            _log_access(scope, path, status, elapsed_ms, request_id, usage)
            if self.ledger and usage is not None and not usage.pop("deferred", False):
                self.ledger.record(
                    UsageEvent(
                        id=request_id,
                        status=status,
                        latency_ms=elapsed_ms,
                        trace_id=trace_id,
                        **usage,
                    )
                )


QUIET_PATHS = {"/metrics", "/livez", "/readyz", "/health"}
access_logger = logging.getLogger("llm_gateway.access")


def _log_access(
    scope: Scope,
    path: str,
    status: int,
    elapsed_ms: float,
    request_id: uuid.UUID,
    usage: dict[str, Any] | None,
) -> None:
    """One line per request; probes and scrapes only at DEBUG so they don't drown the log."""
    level = logging.DEBUG if path in QUIET_PATHS else logging.INFO
    if status >= 500:
        level = logging.WARNING
    if not access_logger.isEnabledFor(level):
        return
    fields = {
        "request_id": str(request_id),
        "method": scope["method"],
        "path": scope["path"],
        "status": status,
        "latency_ms": round(elapsed_ms, 1),
    }
    if usage:
        for key in ("tenant", "model", "backend", "cache", "prompt_tokens", "completion_tokens"):
            if usage.get(key) is not None:
                fields[key] = usage[key]
    access_logger.log(
        level,
        "%s %s %s %.1fms",
        scope["method"],
        scope["path"],
        status,
        elapsed_ms,
        extra={"fields": fields},
    )


def _client_request_id(scope: Scope) -> uuid.UUID | None:
    """Reuse the caller's X-Request-ID when it is a UUID, so logs line up across services."""
    for name, value in scope["headers"]:
        if name == b"x-request-id":
            try:
                return uuid.UUID(value.decode("latin-1"))
            except ValueError:
                return None
    return None


class ClientDisconnected(Exception):
    """The client closed the connection before the backend answered."""


async def _disconnected(request: Request) -> None:
    """Return once the client has gone away (only call after the body has been read)."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def until_disconnect(request: Request, work: Awaitable[T]) -> T:
    """Await `work`, cancelling it if the client disconnects first.

    Without this, an abandoned request keeps its backend slot and keeps the GPU generating
    an answer nobody will read; under overload those orphans pile up and every later
    request queues behind them.
    """
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_disconnected(request))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
    if task.done():
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    raise ClientDisconnected


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
    configure_logging(settings.log_level, settings.log_format)
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
        max_concurrency=settings.backend_max_concurrency,
        queue_timeout_seconds=settings.backend_queue_timeout_seconds,
        on_queue_wait=metrics.queue_wait.observe,
    )
    metrics.build_info.labels(__version__).set(1)

    def redis_error(op: str) -> None:
        metrics.redis_errors.labels(op).inc()

    cache = ResponseCache(redis, settings.cache_ttl_seconds, on_error=redis_error)
    limiter = RateLimiter(redis, settings.rate_limit_rpm, on_error=redis_error)
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
    # Cache keys being answered right now -> the leader's result, for request coalescing
    in_flight: dict[str, asyncio.Future[dict[str, Any] | None]] = {}
    api_keys = settings.api_keys
    if not api_keys:
        logger.warning("GATEWAY_API_KEYS is empty: every /v1 request will be rejected")

    probe_state: dict[str, Any] = {"redis": None, "backends": {}, "at": None}

    async def probe() -> tuple[bool, list[bool]]:
        """Ping Redis and every backend concurrently; update the health gauges."""

        async def redis_ok() -> bool:
            try:
                return bool(await redis.ping())
            except Exception:  # Redis down: cache and rate limiter fail open
                return False

        redis_up, *reachable = await asyncio.gather(redis_ok(), *(b.health() for b in backends))
        metrics.redis_up.set(1 if redis_up else 0)
        for backend, up in zip(backends, reachable, strict=True):
            metrics.backend_reachable.labels(backend.name).set(1 if up else 0)
            if probe_state["backends"].get(backend.name, True) and not up:
                logger.warning("backend %s is unreachable", backend.name)
            elif probe_state["backends"].get(backend.name) is False and up:
                logger.info("backend %s is reachable again", backend.name)
            probe_state["backends"][backend.name] = up
        if probe_state["redis"] is not False and not redis_up:
            logger.warning("redis is unreachable: cache and rate limiting are bypassed")
        elif probe_state["redis"] is False and redis_up:
            logger.info("redis is reachable again")
        probe_state["redis"] = redis_up
        probe_state["at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return redis_up, reachable

    async def warm_up() -> None:
        """Open the Redis and backend connections now, so the first request doesn't pay
        for TCP setup. Failures are logged, not fatal: the gateway still starts."""
        t0 = time.perf_counter()
        redis_up, reachable = await probe()
        down = [b.name for b, up in zip(backends, reachable, strict=True) if not up]
        logger.info(
            "warm-up in %.0f ms: redis %s, backends down: %s",
            (time.perf_counter() - t0) * 1000,
            "up" if redis_up else "DOWN",
            ", ".join(down) or "none",
        )

    async def monitor() -> None:
        """Background probes, so the gauges (and alerts) catch outages with no traffic."""
        while True:
            await asyncio.sleep(settings.health_check_interval_seconds)
            try:
                await probe()
            except Exception:  # never let the monitor die
                logger.exception("health probe failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if store:
            try:
                await store.connect()
            except Exception as exc:  # keep serving; rows queue until the DB is reachable
                logger.warning("usage ledger database unavailable at startup: %s", exc)
        if ledger:
            ledger.start()
        if settings.warm_up_connections:
            await warm_up()
        monitor_task = (
            asyncio.create_task(monitor()) if settings.health_check_interval_seconds > 0 else None
        )
        yield
        if monitor_task:
            monitor_task.cancel()
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

    app.add_middleware(RequestLogMiddleware, ledger=ledger, metrics=metrics)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            _error_body(exc.message, exc.error_type), exc.status_code, headers=exc.headers
        )

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """A bug, not an upstream failure: log it with the request id, answer in OpenAI form."""
        request_id = str(request.scope.get("state", {}).get("request_id", "-"))
        logger.exception("unhandled error in %s (request %s)", request.url.path, request_id)
        return JSONResponse(
            _error_body(f"internal error (request {request_id})", "internal_error"),
            500,
            headers={"x-request-id": request_id},
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
        limit = settings.max_request_bytes
        declared = request.headers.get("content-length", "")
        too_big = GatewayError(413, f"request body exceeds {limit} bytes", "invalid_request_error")
        if limit and declared.isdigit() and int(declared) > limit:
            raise too_big
        body = await request.body()
        if limit and len(body) > limit:
            raise too_big
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GatewayError(400, "request body must be JSON", "invalid_request_error") from exc
        if not isinstance(payload, dict):
            raise GatewayError(400, "request body must be a JSON object", "invalid_request_error")
        payload.setdefault("model", settings.default_model)
        return payload

    def upstream_call(request: Request, work: Awaitable[T]) -> Awaitable[T]:
        return until_disconnect(request, work) if settings.cancel_on_disconnect else work

    def upstream_failure(route: str, model: str, exc: Exception) -> GatewayError:
        if isinstance(exc, ClientDisconnected):
            metrics.requests.labels(route, model, "cancelled").inc()
            return GatewayError(CLIENT_CLOSED_STATUS, "client disconnected", "client_closed")
        if isinstance(exc, Overloaded):
            metrics.requests.labels(route, model, "shed").inc()
            metrics.load_shed.labels(route).inc()
            return GatewayError(503, str(exc), "overloaded_error", headers={"Retry-After": "1"})
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
            metrics.backend_inflight.labels(backend.name).set(router.state[backend.name].inflight)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """A live probe of Redis and every backend, plus circuit-breaker and load state.

        status: ok = everything reachable; degraded = Redis or some backend is down, or a
        circuit is open, but requests can still be served; down = no backend reachable.
        Always HTTP 200: use /readyz for load-balancer decisions.
        """
        refresh_backend_gauges()
        redis_up, reachable = await probe()
        rows = [
            {
                "name": b.name,
                "reachable": up,
                "circuit_open": not router.is_available(b),
                "inflight": router.state[b.name].inflight,
                "saturated": router.is_saturated(b),
                "ewma_latency_ms": (
                    round(router.state[b.name].ewma_latency * 1000, 1)
                    if router.state[b.name].ewma_latency is not None
                    else None
                ),
                "models": list(b.config.models),
            }
            for b, up in zip(backends, reachable, strict=True)
        ]
        if backends and not any(r["reachable"] and not r["circuit_open"] for r in rows):
            status = "down"
        elif not redis_up or any(not r["reachable"] or r["circuit_open"] for r in rows):
            status = "degraded"
        else:
            status = "ok"
        return {
            "status": status,
            "version": __version__,
            "redis": redis_up,
            "backends": rows,
            "max_concurrency_per_backend": settings.backend_max_concurrency or None,
        }

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        """Liveness: the process is serving. No dependency checks (Kubernetes liveness)."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        """Readiness: Redis answers. Backends are not required (the router handles them)."""
        try:
            await redis.ping()
        except Exception as exc:  # not ready: take this pod out of the Service
            return JSONResponse({"status": "unavailable", "redis": str(exc)}, 503)
        return JSONResponse({"status": "ok"})

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
            return serve_stored(hit, "HIT", model, start, usage, report)
        if key:
            metrics.cache.labels("miss").inc()
            leader = in_flight.get(key)
            if leader is not None:
                # An identical request is already being answered: share its result instead
                # of generating the same answer twice (a cold-cache stampede).
                await asyncio.wait({leader})
                if not leader.cancelled() and leader.exception() is None and leader.result():
                    metrics.cache.labels("coalesced").inc()
                    return serve_stored(leader.result(), "COALESCED", model, start, usage, report)
            # The leader failed or its client left: answer this one ourselves.
        flight: asyncio.Future[dict[str, Any] | None] | None = None
        if key and key not in in_flight:
            flight = asyncio.get_running_loop().create_future()
            in_flight[key] = flight
        try:
            response, entry = await answer(request, payload, model, key, namespace, start, report)
        except BaseException as exc:
            if flight and not flight.done():
                if isinstance(exc, Exception):
                    flight.set_exception(exc)
                    flight.exception()  # mark retrieved; followers inspect it themselves
                else:
                    flight.cancel()
            raise
        else:
            if flight and not flight.done():
                flight.set_result(entry)
        finally:
            if flight and in_flight.get(key) is flight:
                del in_flight[key]
        return response

    def serve_stored(
        hit: dict[str, Any],
        state: str,
        model: str,
        start: float,
        usage: dict[str, Any],
        report: GuardrailReport,
        similarity: float | None = None,
    ) -> JSONResponse:
        """Answer from a stored entry (exact hit, semantic hit, or a coalesced request)."""
        outcome = {"HIT": "cache_hit", "SEMANTIC_HIT": "semantic_hit"}.get(state, "coalesced")
        metrics.requests.labels("chat", model, outcome).inc()
        metrics.saved_tokens.labels(model).inc(hit["prompt_tokens"] + hit["completion_tokens"])
        metrics.saved_cost.labels(model).inc(hit["cost_usd"])
        elapsed = time.perf_counter() - start
        label = {"HIT": "hit", "SEMANTIC_HIT": "semantic_hit"}.get(state, "coalesced")
        metrics.latency.labels("chat", model, hit["backend"], label).observe(elapsed)
        headers = _headers(hit["backend"], state, 0.0, elapsed, report)
        if similarity is not None:
            headers["x-gateway-similarity"] = f"{similarity:.4f}"
        usage.update(backend=hit["backend"], cache=state)
        _annotate_span(usage)
        return JSONResponse(hit["response"], headers=headers)

    async def answer(
        request: Request,
        payload: dict[str, Any],
        model: str,
        key: str | None,
        namespace: str,
        start: float,
        report: GuardrailReport,
    ) -> tuple[JSONResponse, dict[str, Any] | None]:
        """A cache miss: try the semantic cache, then the backends. Returns the response
        and the stored entry (shared with coalesced identical requests)."""
        usage = request.state.usage
        lookup = SemanticLookup()
        if key and semantic:
            lookup = await semantic.lookup(payload, namespace)
            if lookup.entry:
                hit = lookup.entry
                response = serve_stored(
                    hit, "SEMANTIC_HIT", model, start, usage, report, lookup.similarity
                )
                return response, hit

        routed = time.perf_counter()
        try:
            response, backend = await upstream_call(
                request, router.run(model, lambda b: b.chat(payload))
            )
        except (BackendError, NoBackendAvailable, Overloaded, ClientDisconnected) as exc:
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
        entry = None
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
        json_response = JSONResponse(
            response, headers=_headers(backend.name, cache_state, cost, elapsed, report)
        )
        return json_response, entry

    async def stream_chat(
        request: Request, payload: dict[str, Any], model: str, report: GuardrailReport
    ) -> StreamingResponse:
        start = time.perf_counter()
        # Ask OpenAI-compatible servers to append a usage chunk at the end of the stream.
        payload = {**payload, "stream_options": {"include_usage": True}}
        try:
            backend, lines, ttft = await upstream_call(request, router.open_stream(model, payload))
        except (BackendError, NoBackendAvailable, Overloaded, ClientDisconnected) as exc:
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
            except (asyncio.CancelledError, GeneratorExit):
                outcome = "cancelled"  # the client went away mid-stream
                raise
            finally:
                await lines.aclose()  # stop the upstream generation and free the slot
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
            response, backend = await upstream_call(
                request, router.run(model, lambda b: b.embeddings(payload))
            )
        except (BackendError, NoBackendAvailable, Overloaded, ClientDisconnected) as exc:
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
