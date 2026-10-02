"""OpenTelemetry tracing: one trace per request, exported over OTLP/HTTP (e.g. to Jaeger).

Auto-instrumented: the FastAPI server span, every httpx call to a backend, and every
Redis command. The gateway adds its own spans for guardrails, the exact and semantic
cache (embed / search / verify), and each router attempt, with GenAI semantic-convention
attributes (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...).

With `OTEL_EXPORTER_OTLP_ENDPOINT` unset, nothing is exported and the tracer is a no-op.
"""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from llm_gateway import __version__

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("llm_gateway", __version__)

_provider: TracerProvider | None = None


def setup_tracing(
    app: Any,
    clients: list[Any],
    endpoint: str,
    service_name: str,
    exporter: SpanExporter | None = None,
) -> TracerProvider | None:
    """Instrument the app, backend httpx clients and Redis.

    `exporter` is for tests (an in-memory exporter, exported synchronously).
    """
    global _provider
    if not endpoint and exporter is None:
        return None
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": __version__})
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        url = endpoint.rstrip("/")
        if not url.endswith("/v1/traces"):
            url += "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
    if _provider is None:
        trace.set_tracer_provider(provider)
    _provider = provider

    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls="health,livez,readyz,metrics"
    )
    for client in clients:
        HTTPXClientInstrumentor.instrument_client(client, tracer_provider=provider)
    instrumentor = RedisInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider)
    logger.info("tracing enabled for %s", service_name)
    return provider


def get_tracer() -> trace.Tracer:
    """The active tracer (the test provider when one is installed)."""
    if _provider is not None:
        return _provider.get_tracer("llm_gateway", __version__)
    return tracer


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
