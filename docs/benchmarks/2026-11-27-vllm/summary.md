# vLLM vs Ollama on the RTX 5070 Ti (2026-11-27)

Same model family on both backends, `Qwen2.5-3B-Instruct`, measured straight to each backend and
once through this gateway pointed at vLLM.

## Setup

- Desktop "XTRRAGE": Ryzen 7 9800X3D, 31 GB RAM, RTX 5070 Ti 16 GB (driver 621.40), Windows 11 Home 10.0.26300, mains power. Nothing else running on the CPU or GPU.
- **vLLM 0.34.0** (`vllm/vllm-openai:latest` in Docker Desktop with the NVIDIA runtime): `--model Qwen/Qwen2.5-3B-Instruct --gpu-memory-utilization 0.8 --max-model-len 4096`. BF16 weights, FlashAttention 2, CUDA graphs; 6.06 GiB KV cache (163,840 tokens, 40 concurrent 4k-token requests).
- **Ollama 0.40.18** with `qwen2.5:3b` (GGUF Q4_K_M, 2.2 GB loaded, 100% GPU, context 4096, default parallelism settings; `ollama_ps.txt`).
- **The two backends are not the same weights:** vLLM serves the BF16 checkpoint, Ollama a 4-bit quantization. That is how each is normally run, and it is the main reason Ollama is faster for a single stream.
- The backends never shared the GPU: all vLLM runs first (Ollama had no model loaded), then the vLLM container was stopped and the Ollama runs followed. Runs are not alternated between backends for that reason.
- This gateway (commit `6f4ff89`) on the host at :8100 with `VLLM_BASE_URL=http://127.0.0.1:8001`, `VLLM_MODELS=Qwen/Qwen2.5-3B-Instruct`, `RATE_LIMIT_RPM=0`, semantic cache off, `Cache-Control: no-cache`.
- Load generator `scripts/bench.py`: `--requests 200 --max-tokens 128`, one run per cell, 20-request warm-up per target first (`warmup.jsonl`). Answers average 58-60 completion tokens on both backends (most stop before 128).
- **0 errors across 1,800 measured requests.** Per-run summaries: `raw.jsonl`.

## Results

| Concurrency | Target | p50 | p95 | p99 | req/s | tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Ollama direct | 229 ms | 315 ms | 343 ms | 4.34 | 260 |
| 1 | vLLM direct | 496 ms | 704 ms | 759 ms | 1.97 | 115 |
| 1 | gateway → vLLM | 496 ms | 691 ms | 760 ms | 1.96 | 115 |
| 4 | Ollama direct | 896 ms | 1054 ms | 1132 ms | 4.50 | 268 |
| 4 | vLLM direct | 535 ms | 802 ms | 805 ms | 7.35 | 432 |
| 4 | gateway → vLLM | 528 ms | 759 ms | 998 ms | 7.29 | 430 |
| 16 | Ollama direct | 3505 ms | 3938 ms | 3994 ms | 4.54 | 268 |
| 16 | vLLM direct | 554 ms | 759 ms | 994 ms | 26.13 | 1534 |
| 16 | gateway → vLLM | 550 ms | 766 ms | 847 ms | 26.72 | 1557 |

## Reading it

- **One request at a time, Ollama wins:** 260 vs 115 tok/s. Decoding a 3B model is memory-bandwidth bound, and the Q4 weights are about a quarter of the BF16 bytes.
- **Under concurrency, vLLM wins by a wide margin.** Ollama's throughput stays at ~4.5 req/s (~268 tok/s) from 1 to 16 clients, so extra clients only queue (p50 3.5 s at 16). vLLM's continuous batching keeps p50 near 0.5 s and reaches 26 req/s / ~1,540 tok/s at 16 clients: **5.8× Ollama's throughput** at that load.
- The crossover is between 1 and 4 concurrent clients on this card.
- **The gateway adds no measurable cost in front of vLLM:** p50 within ±7 ms of direct at every concurrency and throughput within 2%, which is run-to-run noise at ~0.5 s per request.
