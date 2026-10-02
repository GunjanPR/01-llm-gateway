import pytest

from conftest import AUTH, Upstream, fail, make_backend, ok
from llm_gateway.ledger import UsageEvent, UsageLedger
from llm_gateway.metrics import GatewayMetrics

CHAT = {"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]}


class Sink:
    """Collects batches instead of writing to PostgreSQL."""

    def __init__(self, fail_times: int = 0) -> None:
        self.rows: list[UsageEvent] = []
        self.fail_times = fail_times

    async def __call__(self, batch: list[UsageEvent]) -> None:
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("database down")
        self.rows.extend(batch)


def ledger_client(client_factory, backends, **overrides):
    sink = Sink()
    client = client_factory(backends, ledger_writer=sink, **overrides)

    def rows() -> list[UsageEvent]:
        client.portal.call(client.app.state.ledger.flush)
        return sink.rows

    return client, rows


def test_successful_chat_is_logged(client_factory):
    client, rows = ledger_client(
        client_factory, [make_backend("ollama", Upstream(ok()), cost_out=1.0)]
    )
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    [row] = rows()
    assert str(row.id) == resp.headers["x-request-id"]
    assert (row.route, row.model, row.backend, row.cache, row.status) == (
        "chat",
        "llama3.2",
        "ollama",
        "MISS",
        200,
    )
    assert (row.prompt_tokens, row.completion_tokens) == (5, 7)
    assert row.cost_usd == pytest.approx(0.007)
    assert row.latency_ms > 0
    assert len(row.tenant) == 12  # hashed key, never the key itself


def test_cache_hits_errors_and_rejections_are_logged(client_factory):
    client, rows = ledger_client(
        client_factory, [make_backend("a", Upstream(ok()))], rate_limit_rpm=3
    )
    client.post("/v1/chat/completions", json=CHAT, headers=AUTH)  # miss
    client.post("/v1/chat/completions", json=CHAT, headers=AUTH)  # hit
    pii = {**CHAT, "messages": [{"role": "user", "content": "SSN 123-45-6789"}]}
    client.post("/v1/chat/completions", json=pii, headers=AUTH)  # redacted, still OK
    client.post("/v1/chat/completions", json=CHAT, headers=AUTH)  # 4th: rate limited
    logged = rows()
    assert [(r.cache, r.status) for r in logged] == [
        ("MISS", 200),
        ("HIT", 200),
        ("MISS", 200),
        (None, 429),
    ]
    assert logged[1].prompt_tokens == 0 and logged[1].cost_usd == 0
    assert logged[2].guardrails == "pii_ssn=1"
    assert logged[3].model == "-"


def test_blocked_and_failed_requests_are_logged(client_factory):
    client, rows = ledger_client(
        client_factory, [make_backend("a", Upstream(fail(500)))], guardrail_pii="block"
    )
    pii = {**CHAT, "messages": [{"role": "user", "content": "mail a@b.co"}]}
    assert client.post("/v1/chat/completions", json=pii, headers=AUTH).status_code == 400
    assert client.post("/v1/chat/completions", json=CHAT, headers=AUTH).status_code == 502
    blocked, failed = rows()
    assert (blocked.status, blocked.guardrails) == (400, "pii_email=1")
    assert (failed.status, failed.backend) == (502, None)


def test_streams_are_logged_with_token_counts(client_factory):
    client, rows = ledger_client(client_factory, [make_backend("a", Upstream(ok()))])
    with client.stream(
        "POST", "/v1/chat/completions", json={**CHAT, "stream": True}, headers=AUTH
    ) as resp:
        list(resp.iter_lines())
    [row] = rows()
    assert (row.route, row.status, row.prompt_tokens, row.completion_tokens) == (
        "chat_stream",
        200,
        4,
        2,
    )


def test_embeddings_logged_and_unauthenticated_requests_not(client_factory):
    client, rows = ledger_client(client_factory, [make_backend("a", Upstream(ok()))])
    client.post("/v1/embeddings", json={"model": "e", "input": "x"}, headers=AUTH)
    client.post("/v1/embeddings", json={"model": "e", "input": "x"})  # 401, no tenant
    [row] = rows()
    assert (row.route, row.prompt_tokens) == ("embeddings", 3)


def test_usage_endpoint_needs_a_database(client_factory):
    client, _ = ledger_client(client_factory, [make_backend("a", Upstream(ok()))])
    assert client.get("/v1/usage", headers=AUTH).status_code == 503


def event(i: int = 0) -> UsageEvent:
    return UsageEvent(tenant="t", route="chat", model="m", status=200, latency_ms=float(i))


@pytest.mark.anyio
async def test_failed_writes_keep_rows_for_the_next_flush():
    sink = Sink(fail_times=1)
    ledger = UsageLedger(sink, GatewayMetrics(), batch_size=2)
    for i in range(3):
        ledger.record(event(i))
    assert await ledger.flush() == 0  # database down: nothing lost
    assert len(ledger.queue) == 3
    assert await ledger.flush() == 3
    assert [e.latency_ms for e in sink.rows] == [0.0, 1.0, 2.0]  # order preserved


def test_full_queue_drops_oldest_and_counts_it():
    metrics = GatewayMetrics()
    ledger = UsageLedger(Sink(), metrics, max_queue=2)
    for i in range(3):
        ledger.record(event(i))
    assert [e.latency_ms for e in ledger.queue] == [1.0, 2.0]
    assert metrics.ledger_dropped._value.get() == 1
