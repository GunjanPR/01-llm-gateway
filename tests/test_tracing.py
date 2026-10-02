from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conftest import AUTH, Upstream, fail, make_backend, ok
from llm_gateway.tracing import setup_tracing

CHAT = {"model": "llama3.2", "messages": [{"role": "user", "content": "mail me at a@b.co"}]}


def traced_client(client_factory, backends, **overrides):
    exporter = InMemorySpanExporter()
    client = client_factory(backends, span_exporter=exporter, **overrides)
    return client, exporter


def test_request_produces_a_trace_with_gateway_spans(client_factory):
    client, exporter = traced_client(
        client_factory,
        [make_backend("primary", Upstream(fail(503))), make_backend("secondary", Upstream(ok()))],
    )
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 200
    spans = exporter.get_finished_spans()
    by_name: dict[str, list] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)

    server = by_name["POST /v1/chat/completions"][0]
    assert format(server.context.trace_id, "032x") == resp.headers["x-trace-id"]
    assert server.attributes["gen_ai.request.model"] == "llama3.2"
    assert server.attributes["gen_ai.usage.output_tokens"] == 7
    assert server.attributes["llmgw.backend"] == "secondary"
    assert server.attributes["llmgw.cache"] == "MISS"

    attempts = by_name["router.attempt"]
    assert [a.attributes["llmgw.backend"] for a in attempts] == ["primary", "primary", "secondary"]
    assert [a.status.is_ok for a in attempts] == [False, False, True]
    assert by_name["guardrails.input"][0].attributes["llmgw.guardrails"] == "pii_email=1"
    assert "cache.exact.lookup" in by_name

    # Each backend HTTP call is a child of its router attempt.
    http_spans = [
        s
        for s in spans
        if s.kind.name == "CLIENT"
        and "POST" in (s.attributes.get("http.method"), s.attributes.get("http.request.method"))
    ]
    attempt_ids = {a.context.span_id for a in attempts}
    assert http_spans and all(s.parent.span_id in attempt_ids for s in http_spans)
    # All spans belong to the same trace.
    assert {s.context.trace_id for s in spans if s.name != "connect"} == {server.context.trace_id}


def test_tracing_is_off_without_an_endpoint():
    # OpenTelemetry providers and httpx patches are process-global, so "off" is checked at
    # the setup function rather than by inspecting a second app in the same process.
    assert setup_tracing(app=None, clients=[], endpoint="", service_name="x") is None
