from llm_gateway.config import Settings, parse_models


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
    assert backends["ollama"].base_url == "http://localhost:11434/v1"
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
