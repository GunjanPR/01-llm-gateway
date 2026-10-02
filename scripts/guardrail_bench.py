"""Time the input guardrails (PII redaction + injection checks) on realistic prompt sizes.

    python scripts/guardrail_bench.py

Prints p50/p95/p99 per prompt size and appends a JSON line to results/guardrails.jsonl.
"""

import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from llm_gateway.guardrails import scan_messages

PARAGRAPH = (
    "BRCA1 variants such as c.68_69delAG reduce homologous recombination repair in "
    "patient-derived cell lines (PMC1234567). Contact the corresponding author at "
    "lab.lead@university.edu or (302) 555-0142 for the supplementary tables. "
)
SIZES = {"short_200ch": 1, "medium_2kch": 10, "long_20kch": 100}


def time_scan(text: str, iterations: int) -> list[float]:
    messages = [
        {"role": "system", "content": "You answer biomedical questions with citations."},
        {"role": "user", "content": text},
    ]
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        scan_messages(messages, redact=True, check_injection=True)
        samples.append(time.perf_counter() - start)
    return samples


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    rows = {}
    for label, repeats in SIZES.items():
        text = (PARAGRAPH * repeats).strip()
        time_scan(text, 50)  # warm up the regex engine
        samples = time_scan(text, 2000 if repeats < 100 else 500)
        rows[label] = {
            "chars": len(text),
            "p50_us": round(pct(samples, 0.50) * 1e6, 1),
            "p95_us": round(pct(samples, 0.95) * 1e6, 1),
            "p99_us": round(pct(samples, 0.99) * 1e6, 1),
            "mean_us": round(statistics.fmean(samples) * 1e6, 1),
        }
        print(f"{label:>12}: {rows[label]}")
    out = Path("results/guardrails.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "results": rows,
    }
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
