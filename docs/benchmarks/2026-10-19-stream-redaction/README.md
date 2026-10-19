# PII redaction on streamed answers (2026-10-19)

**Problem.** `GUARDRAIL_OUTPUT_PII=redact` only covered non-streaming answers. Streams
went out unredacted, because a model emits an email or phone number in several chunks
(`"jane.d"`, `"oe@exam"`, `"ple.com"`), and no single chunk contains the whole entity.

**Fix.** `StreamRedactor` (in `guardrails.py`) holds back only text that a later chunk
could still turn into a match: the trailing word (it may grow into an email, IP or SSN)
and a trailing run of digits and separators (it may grow into a phone or card number),
and it never cuts through a match. Everything before that is redacted and sent at once.
Held text is released when the next word starts, and flushed at the choice's
`finish_reason` or at the end of the stream.

While testing it, the benchmark also found a false negative in the one-shot redactor
used for prompts and non-streaming answers: a digit run that looked like a card but failed
the Luhn check was consumed whole, so `"in 2019 4111 1111 1111 1111"` hid the card behind
the failed candidate `"2019 4111 1111 1111"`, and `"412 123-45-6789"` hid the SSN. The
scan now tries the longest valid card at that position, and otherwise resumes one
character later.

## Leaks ([stream_guard.json](stream_guard.json), `python scripts/stream_guard_bench.py`)

500 synthetic answers with 2,010 PII entities (emails, phones, cards, SSNs, IPs, placed
next to ordinary numbers like years and counts), cut into 131,379 LLM-like chunks of 1-4
characters:

| | Entities that reached the client |
|---|---|
| Streams before this change (no redaction) | 2,010 / 2,010 |
| Redacting each chunk on its own | 2,010 / 2,010 |
| One-shot redactor on the full text, before the Luhn fix | 68 / 2,010 |
| **`StreamRedactor`, after** | **0 / 2,010** |

A property test (`tests/test_stream_guardrails.py`) checks that 2,000 random chunkings
give exactly the same output as redacting the whole text at once.

## Delay

Text reaches the client a median of **4 characters** (p95: 13) behind the upstream,
about one token. No chunk is held for a whole sentence, and the first token is not
delayed by a fixed buffer.

## CPU ([cpu_ab.txt](cpu_ab.txt))

Same method as [2026-10-03-cpu-profile](../2026-10-03-cpu-profile/): mock upstream with
0 ms latency streaming 200 chunks per answer, 1,000 streams at concurrency 16, gateway
process CPU divided by requests, 3 alternating runs each on a 4-vCPU Linux cloud VM:

| `GUARDRAIL_OUTPUT_PII` | Gateway CPU per 200-chunk stream |
|---|---|
| off | 5.95 ms (5.64, 5.55, 6.67) |
| redact | 8.21 ms (8.00, 8.56, 8.08) |

That is about **11 µs per chunk** (7 µs of it in the redactor and re-encoding the chunk,
measured in-process). The laptop load test in the main README produced 26.5 tokens/s,
one token about every 38 ms, so this is about 0.03% of a stream's duration.
`GUARDRAIL_OUTPUT_PII=off` turns it off.
