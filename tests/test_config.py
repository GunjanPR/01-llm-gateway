import pytest

from llm_gateway.config import Settings, ipv4_loopback, parse_models


def test_parse_models():
    assert parse_models("") == {"*": "*"}
    assert parse_models("llama3.2, fast=llama3.2:1b") == {
        "llama3.2": "llama3.2",
        "fast": "llama3.2:1b",
    }


def test_backend_configs_from_settings():
    settings = Settings(
        _env_file=None,
        ollama_base_url="http://localhost:11434",
        vllm_base_url="http://gpu:8001/v1",
        hosted_base_url="https://api.groq.com/openai/v1",
        hosted_api_key="k",
        hosted_name="groq",
        hosted_models="llama3.2=llama-3.1-8b-instant",
        hosted_cost_per_1k_input=0.05,
    )
    backends = {b.name: b for b in settings.backend_configs()}
    assert list(backends) == ["vllm", "ollama", "groq"]
    assert backends["ollama"].base_url == "http://127.0.0.1:11434/v1"
    assert backends["vllm"].base_url == "http://gpu:8001/v1"
    assert backends["groq"].upstream_model("llama3.2") == "llama-3.1-8b-instant"
    assert backends["groq"].cost_per_1k_input == 0.05


def test_hosted_backend_needs_a_key():
    settings = Settings(_env_file=None, hosted_base_url="https://x/v1", ollama_base_url="")
    assert settings.backend_configs() == []


def test_api_keys_parsing():
    assert Settings(_env_file=None, gateway_api_keys=" a, b ,").api_keys == {"a", "b"}


def test_multiple_ollama_nodes():
    settings = Settings(_env_file=None, ollama_base_url="http://a:11434, http://b:11435/v1")
    backends = settings.backend_configs()
    assert [b.name for b in backends] == ["ollama", "ollama-2"]
    assert [b.base_url for b in backends] == ["http://a:11434/v1", "http://b:11435/v1"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434", "http://127.0.0.1:11434"),
        ("http://LOCALHOST:11434/v1", "http://127.0.0.1:11434/v1"),
        ("redis://localhost:6379/0", "redis://127.0.0.1:6379/0"),
        ("postgresql://u:p@localhost:5432/db", "postgresql://u:p@127.0.0.1:5432/db"),
        ("http://localhost", "http://127.0.0.1"),
        ("http://localhost.example.com:80", "http://localhost.example.com:80"),
        ("http://gpu-box:8001/v1", "http://gpu-box:8001/v1"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_ipv4_loopback(url, expected):
    assert ipv4_loopback(url) == expected


def test_settings_rewrite_localhost_urls():
    settings = Settings(
        _env_file=None,
        redis_url="redis://localhost:6379/0",
        ollama_base_url="http://localhost:11434, http://gpu:11434",
        ledger_database_url="postgresql://u:p@localhost:5432/db",
        otel_exporter_otlp_endpoint="http://localhost:4318",
    )
    assert settings.redis_url == "redis://127.0.0.1:6379/0"
    assert [b.base_url for b in settings.backend_configs()] == [
        "http://127.0.0.1:11434/v1",
        "http://gpu:11434/v1",
    ]
    assert settings.ledger_database_url == "postgresql://u:p@127.0.0.1:5432/db"
    assert settings.otel_exporter_otlp_endpoint == "http://127.0.0.1:4318"


def test_localhost_rewrite_can_be_disabled():
    settings = Settings(
        _env_file=None, prefer_ipv4_loopback=False, ollama_base_url="http://localhost:11434"
    )
    assert settings.ollama_base_url == "http://localhost:11434"


def test_backend_pool_settings_reach_the_http_client():
    from llm_gateway.backends import OpenAICompatibleBackend

    settings = Settings(_env_file=None, backend_keepalive_seconds=90, backend_max_connections=7)
    config = settings.backend_configs()[0]
    pool = OpenAICompatibleBackend(config)._client._transport._pool
    assert pool._keepalive_expiry == 90
    assert pool._max_connections == 7
