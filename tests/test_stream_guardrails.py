"""Output PII redaction on streamed answers, where an entity can span several chunks."""

import json
import random

import httpx
import pytest

from conftest import AUTH, Upstream, make_backend, sse
from llm_gateway.guardrails import StreamRedactor, redact_pii


def stream_through(parts: list[str]) -> tuple[str, StreamRedactor]:
    redactor = StreamRedactor()
    return "".join(redactor.feed(p) for p in parts) + redactor.flush(), redactor


@pytest.mark.parametrize(
    "parts",
    [
        ["Mail jane.d", "oe@exam", "ple.com today"],
        ["Call (302) 5", "55-01", "42 now"],
        ["Card 4111 ", "1111 1111", " 1111 ok"],
        ["Server 192.168.", "10.24 down"],
        ["SSN 123-4", "5-6789."],
    ],
)
def test_entities_split_across_chunks_are_redacted(parts):
    out, _ = stream_through(parts)
    assert out == redact_pii("".join(parts))[0]
    assert "[REDACTED_" in out


def test_text_is_released_once_the_next_word_starts():
    redactor = StreamRedactor()
    assert redactor.feed("The answer is") == "The answer "  # "is" may still grow
    assert redactor.feed(" 42") == "is"  # " 42" may still grow into a phone number
    assert redactor.feed(" apples.") == " 42 "
    assert redactor.flush() == "apples."


def test_unbroken_runs_are_not_held_forever():
    redactor = StreamRedactor()
    emitted = redactor.feed("x" * 2000)
    assert len(emitted) == 2000 - 254  # only an email's worth stays held
    assert emitted + redactor.flush() == "x" * 2000


PIECES = [
    "Contact ",
    "jane.doe@example.com",
    " or call ",
    "(555) 123-4567",
    " card ",
    "4111 1111 1111 1111",
    " ip 192.168.1.20",
    " ssn 123-45-6789",
    " order 12345 shipped",
    " +1 415-555-0100.",
    " id 1700000000 ",
    "\n",
    "ok",
]


def test_any_chunking_matches_redacting_the_whole_text():
    """However the text is cut into chunks, the stream gets exactly the one-shot result."""
    rng = random.Random(0)
    for _ in range(2000):
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 12)))
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(0, 15))))
        parts = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)], strict=True)]
        out, redactor = stream_through(parts)
        expected, found = redact_pii(text)
        assert out == expected, parts
        assert redactor.found == found


def streaming(chunks: list[str], finish: bool = True):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        text = sse(chunks, {"prompt_tokens": 4, "completion_tokens": len(chunks)})
        if finish:  # a final chunk with no content and a finish reason, as Ollama sends
            done = {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            text = text.replace("data: [DONE]", f"data: {json.dumps(done)}\n\ndata: [DONE]")
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    return handler


def read_stream(client, **body) -> tuple[str, list[dict]]:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with client.stream(
        "POST", "/v1/chat/completions", json={**payload, **body}, headers=AUTH
    ) as resp:
        assert resp.status_code == 200
        events = [line[5:].strip() for line in resp.iter_lines() if line.startswith("data:")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["delta"].get("content") or "" for e in chunks for c in e["choices"])
    return text, chunks


@pytest.mark.parametrize("finish", [True, False])
def test_api_redacts_pii_split_across_stream_chunks(client_factory, finish):
    chunks = ["Email ", "admin@si", "te.org or ", "call 302-", "555-0142"]
    client = client_factory([make_backend("a", Upstream(streaming(chunks, finish)))])
    text, _ = read_stream(client)
    assert text == "Email [REDACTED_EMAIL] or call [REDACTED_PHONE]"
    metrics = client.get("/metrics").text
    assert 'kind="pii_email"' in metrics and 'direction="output"' in metrics


def test_api_stream_redaction_can_be_turned_off(client_factory):
    chunks = ["Email ", "admin@si", "te.org"]
    client = client_factory(
        [make_backend("a", Upstream(streaming(chunks)))], guardrail_output_pii="off"
    )
    text, _ = read_stream(client)
    assert text == "Email admin@site.org"
