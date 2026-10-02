import json

from conftest import AUTH, Upstream, fail, make_backend, ok

CHAT = {"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]}


def test_rejects_missing_or_wrong_key(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 401
    wrong = client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer x"})
    assert wrong.status_code == 401
    assert wrong.json()["error"]["type"] == "authentication_error"


def test_rejects_everything_when_no_keys_configured(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))], gateway_api_keys="")
    assert client.post("/v1/chat/completions", json=CHAT, headers=AUTH).status_code == 401


def test_chat_completion_passthrough(client_factory):
    upstream = Upstream(ok("hello there"))
    client = client_factory([make_backend("ollama", upstream, models={"llama3.2": "llama3.2:3b"})])
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello there"
    assert resp.headers["x-gateway-backend"] == "ollama"
    assert resp.headers["x-gateway-cache"] == "MISS"
    # The public model name is rewritten to the backend's upstream name.
    assert upstream.requests[0]["body"]["model"] == "llama3.2:3b"
    assert upstream.requests[0]["path"] == "/v1/chat/completions"


def test_default_model_applied(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)])
    client.post("/v1/chat/completions", json={"messages": CHAT["messages"]}, headers=AUTH)
    assert upstream.requests[0]["body"]["model"] == "llama3.2"


def test_validation_errors(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    assert client.post("/v1/chat/completions", json={"model": "x"}, headers=AUTH).status_code == 400
    assert client.post("/v1/chat/completions", content=b"nope", headers=AUTH).status_code == 400


def test_cache_hit_skips_backend(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)])
    first = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    second = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert second.headers["x-gateway-cache"] == "HIT"
    assert second.json() == first.json()
    assert len(upstream.requests) == 1


def test_cache_bypass_header_and_deterministic_mode(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)], cache_mode="deterministic")

    def post(body, headers=AUTH):
        return client.post("/v1/chat/completions", json=body, headers=headers)

    post(CHAT)
    assert post(CHAT).headers["x-gateway-cache"] == "BYPASS"  # temperature is not 0
    det = {**CHAT, "temperature": 0}
    post(det)
    assert post(det).headers["x-gateway-cache"] == "HIT"
    assert post(det, {**AUTH, "Cache-Control": "no-cache"}).headers["x-gateway-cache"] == "BYPASS"
    assert len(upstream.requests) == 4


def test_falls_back_when_primary_fails(client_factory):
    primary, secondary = Upstream(fail(503)), Upstream(ok("from secondary"))
    client = client_factory(
        [make_backend("primary", primary), make_backend("secondary", secondary)]
    )
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["x-gateway-backend"] == "secondary"
    assert len(primary.requests) == 2  # one try + one retry


def test_client_errors_are_not_retried(client_factory):
    primary, secondary = Upstream(fail(400)), Upstream(ok())
    client = client_factory(
        [make_backend("primary", primary), make_backend("secondary", secondary)]
    )
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 400
    assert len(primary.requests) == 1
    assert secondary.requests == []


def test_502_when_all_backends_fail(client_factory):
    client = client_factory([make_backend("a", Upstream(fail(500)))])
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


def test_404_when_no_backend_serves_model(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()), models={"other": "other"})])
    assert client.post("/v1/chat/completions", json=CHAT, headers=AUTH).status_code == 404


def test_rate_limit(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))], rate_limit_rpm=2)
    responses = [client.post("/v1/chat/completions", json=CHAT, headers=AUTH) for _ in range(3)]
    assert [r.status_code for r in responses] == [200, 200, 429]
    assert "Retry-After" in responses[-1].headers


def test_streaming(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)])
    body = {**CHAT, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body, headers=AUTH) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [line for line in resp.iter_lines() if line.startswith("data:")]
    assert events[-1] == "data: [DONE]"
    content = "".join(
        c["delta"].get("content", "") for e in events[:-1] for c in json.loads(e[5:])["choices"]
    )
    assert content == "hello"
    assert upstream.requests[0]["body"]["stream"] is True
    assert upstream.requests[0]["body"]["stream_options"] == {"include_usage": True}
    metrics = client.get("/metrics").text
    assert 'llmgw_tokens_total{backend="a",kind="completion",model="llama3.2"} 2.0' in metrics


def test_streaming_falls_back_before_first_token(client_factory):
    client = client_factory(
        [make_backend("bad", Upstream(fail(500))), make_backend("good", Upstream(ok()))]
    )
    body = {**CHAT, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body, headers=AUTH) as resp:
        assert resp.headers["x-gateway-backend"] == "good"


def test_embeddings(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)])
    body = {"model": "nomic-embed-text", "input": "text"}
    resp = client.post("/v1/embeddings", json=body, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["data"][0]["embedding"] == [0.1, 0.2]
    assert upstream.requests[0]["path"] == "/v1/embeddings"


def test_models_lists_configured_names(client_factory):
    backend = make_backend("a", Upstream(ok()), models={"llama3.2": "llama3.2", "fast": "x:1b"})
    client = client_factory([backend])
    ids = [m["id"] for m in client.get("/v1/models", headers=AUTH).json()["data"]]
    assert ids == ["fast", "llama3.2"]


def test_cost_is_reported_and_metrics_exported(client_factory):
    backend = make_backend("hosted", Upstream(ok()), cost_in=1.0, cost_out=2.0)
    client = client_factory([backend])
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    # Upstream reports 5 prompt + 7 completion tokens: 5/1000*1 + 7/1000*2 = 0.019
    assert float(resp.headers["x-gateway-cost-usd"]) == 0.019
    client.post("/v1/chat/completions", json=CHAT, headers=AUTH)  # cache hit
    text = client.get("/metrics").text
    assert 'llmgw_tokens_total{backend="hosted",kind="completion",model="llama3.2"} 7.0' in text
    assert 'llmgw_cache_requests_total{result="hit"} 1.0' in text
    assert 'llmgw_cache_saved_cost_usd_total{model="llama3.2"} 0.019' in text
    assert 'llmgw_backend_available{backend="hosted"} 1.0' in text


def test_health_reports_backends_and_redis(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    body = client.get("/health").json()
    assert body["redis"] is True
    assert body["backends"][0]["name"] == "a"


def test_gateway_overhead_is_measured(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    text = client.get("/metrics").text
    assert 'llmgw_gateway_overhead_seconds_count{route="chat"} 1.0' in text


def test_cache_is_per_tenant_by_default(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)], gateway_api_keys="k1,k2")
    for key in ("k1", "k2"):
        resp = client.post(
            "/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {key}"}
        )
        assert resp.headers["x-gateway-cache"] == "MISS"
    assert len(upstream.requests) == 2


def test_global_cache_scope_shares_entries(client_factory):
    upstream = Upstream(ok())
    client = client_factory(
        [make_backend("a", upstream)], gateway_api_keys="k1,k2", cache_scope="global"
    )
    client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer k1"})
    resp = client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer k2"})
    assert resp.headers["x-gateway-cache"] == "HIT"


def test_liveness_and_readiness(client_factory):
    client = client_factory([make_backend("a", Upstream(fail(500)))])
    assert client.get("/livez").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200  # backends down doesn't matter
