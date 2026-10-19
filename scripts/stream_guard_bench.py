"""Streaming output redaction: leaks, delay and CPU cost on model-like token streams.

    python scripts/stream_guard_bench.py

Builds answers that contain PII, cuts them into token-sized chunks the way an LLM streams
(words split into 1-4 character pieces, spaces attached to the next piece), and compares:

* no redaction (what streaming did before): every entity reaches the client;
* naive per-chunk redaction: each chunk redacted on its own;
* `StreamGuard`: the chunk-safe redactor the gateway now uses.

Reports how many entities leak, how far the visible text lags behind the upstream, and
the CPU time per chunk. Writes a JSON summary to results/stream_guard.json.
"""

import json
import platform
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from llm_gateway.guardrails import StreamGuard, redact_pii

ENTITIES = [
    "lab.lead@university.edu",
    "jane.doe+work@example.co.uk",
    "(302) 555-0142",
    "+1 415-555-0100",
    "302.555.0199",
    "4111 1111 1111 1111",
    "5500-0000-0000-0004",
    "123-45-6789",
    "192.168.10.24",
]
FILLER_TEXT = (
    "BRCA1 variants reduce homologous recombination repair in patient derived cell lines and "
    "the corresponding author shared the supplementary tables with the review board after "
    "the trial closed in 2019 with 412 participants across 3 sites"
)
FILLER = FILLER_TEXT.split(" ")


def answer(rng: random.Random) -> tuple[str, list[str]]:
    words, entities = [], []
    for _ in range(rng.randint(40, 120)):
        if rng.random() < 0.05:
            entity = rng.choice(ENTITIES)
            entities.append(entity)
            words.append(entity)
        else:
            words.append(rng.choice(FILLER))
    return " ".join(words) + ".", entities


def tokenize(text: str, rng: random.Random) -> list[str]:
    """Cut text into LLM-like pieces: 1-4 characters, a space starts a new piece."""
    pieces, i = [], 0
    while i < len(text):
        size = rng.randint(1, 4)
        j = i + 1
        while j < len(text) and j - i < size and text[j] != " ":
            j += 1
        pieces.append(text[i:j])
        i = j
    return pieces


def event(content: str, finish: bool = False) -> dict:
    choice = {"index": 0, "delta": {"content": content}}
    if finish:
        choice["finish_reason"] = "stop"
    return {"id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [choice]}


def leaked(output: str, entities: list[str]) -> int:
    return sum(output.count(e) for e in set(entities))


def main() -> None:
    rng = random.Random(42)
    totals = {"entities": 0, "none": 0, "naive": 0, "guard": 0}
    lags, chunk_seconds, n_chunks = [], [], 0
    for _ in range(500):
        text, entities = answer(rng)
        pieces = tokenize(text, rng)
        totals["entities"] += len(entities)
        totals["none"] += leaked(text, entities)
        totals["naive"] += leaked("".join(redact_pii(p)[0] for p in pieces), entities)

        guard, out = StreamGuard(), []
        for k, piece in enumerate(pieces):
            chunk = event(piece, finish=k == len(pieces) - 1)
            t0 = time.perf_counter()
            line = guard.rewrite(chunk)
            chunk_seconds.append(time.perf_counter() - t0)
            content = json.loads(line[5:])["choices"][0]["delta"]["content"]
            out.append(content)
            if k < len(pieces) - 1:  # characters received upstream but not yet sent on
                lags.append(len(redact_pii("".join(pieces[: k + 1]))[0]) - len("".join(out)))
        n_chunks += len(pieces)
        totals["guard"] += leaked("".join(out), entities)

    chunk_us = sorted(s * 1e6 for s in chunk_seconds)
    summary = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.machine()} / Python {platform.python_version()}",
        "answers": 500,
        "chunks": n_chunks,
        "entities": totals["entities"],
        "leaked_no_redaction": totals["none"],
        "leaked_naive_per_chunk": totals["naive"],
        "leaked_stream_guard": totals["guard"],
        "lag_chars_median": statistics.median(lags),
        "lag_chars_p95": sorted(lags)[int(0.95 * (len(lags) - 1))],
        "cpu_us_per_chunk_p50": round(chunk_us[len(chunk_us) // 2], 1),
        "cpu_us_per_chunk_p99": round(chunk_us[int(0.99 * (len(chunk_us) - 1))], 1),
    }
    print(json.dumps(summary, indent=2))
    out = Path("results/stream_guard.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
