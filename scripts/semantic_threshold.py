"""Pick a safe semantic-cache threshold from labelled prompt pairs.

For every pair in eval/semantic_pairs.jsonl, embed both prompts and compute cosine
similarity. For each candidate threshold, report the paraphrase hit rate (pairs that
should share an answer) and the number of false hits (pairs that must not). A false hit
returns a wrong cached answer, so the threshold is chosen as the lowest value with zero
false hits on this set.

    python scripts/semantic_threshold.py --base-url http://127.0.0.1:8000/v1 --api-key $KEY
"""

import argparse
import json
import math
from pathlib import Path

import httpx

PREFIXES = {"none": "", "clustering": "clustering: ", "search_query": "search_query: "}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--pairs", default="eval/semantic_pairs.jsonl")
    parser.add_argument("--out", default="results/semantic_threshold.json")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line]
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    report: dict = {"pairs": len(pairs), "same": sum(p["same"] for p in pairs), "prefixes": {}}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=60) as client:
        for name, prefix in PREFIXES.items():
            texts = sorted({t for p in pairs for t in (p["a"], p["b"])})
            resp = client.post(
                "/embeddings", json={"model": args.model, "input": [prefix + t for t in texts]}
            )
            resp.raise_for_status()
            vectors = {t: d["embedding"] for t, d in zip(texts, resp.json()["data"], strict=True)}
            scored = [(cosine(vectors[p["a"]], vectors[p["b"]]), p["same"]) for p in pairs]
            same = sorted(s for s, ok in scored if ok)
            different = sorted(s for s, ok in scored if not ok)
            sweep = []
            for t in [x / 100 for x in range(80, 100)]:
                hits = sum(s >= t for s in same)
                false_hits = sum(s >= t for s in different)
                sweep.append(
                    {"threshold": t, "hit_rate": hits / len(same), "false_hits": false_hits}
                )
            safe = next((row for row in sweep if row["false_hits"] == 0), None)
            report["prefixes"][name] = {
                "paraphrase_similarity_min_median": [
                    round(same[0], 4),
                    round(same[len(same) // 2], 4),
                ],
                "near_miss_similarity_median_max": [
                    round(different[len(different) // 2], 4),
                    round(different[-1], 4),
                ],
                "lowest_safe_threshold": safe,
                "hardest_near_miss": max(
                    ((s, p) for (s, ok), p in zip(scored, pairs, strict=True) if not ok),
                    key=lambda x: x[0],
                )[1],
                "sweep": sweep,
            }
            summary = {k: v for k, v in report["prefixes"][name].items() if k != "sweep"}
            print(name, json.dumps(summary, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
