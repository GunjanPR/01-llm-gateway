"""Request/response guardrails: PII redaction and prompt-injection detection.

Both are deterministic, rule-based checks that run in-process before any model call, so
they add microseconds, not a second model round trip. They are a first line of defence,
not a guarantee: regexes miss novel PII formats, and the injection rules catch common
phrasings ("ignore previous instructions", "reveal your system prompt", ...), not every
adversarial prompt.
"""

import json
import re
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# Order matters: more specific patterns run first so a card number is not also a phone.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "SSN": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "PHONE": re.compile(
        # Needs a separator or parentheses after the area code, and "+" or a separator after a
        # country code, so bare digit runs (IDs, Unix timestamps) are not treated as phones.
        r"(?<![\w-])(?:\+\d{1,3}[ .-]?|\d{1,3}[ .-])?"
        r"(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]?\d{4}(?![\w-])"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
}

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "ignore_instructions": r"\b(ignore|disregard|forget|override)\b.{0,40}\b"
        r"(previous|prior|above|earlier|all|any|system|your)\b.{0,20}\b"
        r"(instructions?|prompts?|rules|directions|guidelines)\b",
        "reveal_system_prompt": r"\b(reveal|show|print|repeat|leak|output|tell me)\b.{0,40}\b"
        r"(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
        "role_override": r"\byou are (now|no longer)\b.{0,60}\b(unrestricted|unfiltered|"
        r"jailbroken|dan|evil|without (any )?(rules|restrictions|limits))\b",
        "jailbreak_keyword": r"\b(jailbreak|do anything now|developer mode enabled|dan mode)\b",
        "delimiter_injection": r"(<\|?(im_start|im_end|system)\|?>|\[/?INST\]|###\s*system\s*:)",
    }.items()
}


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# All PII kinds in one alternation: a single scan of the text instead of one per kind.
# Alternation order keeps the precedence above (first kind that matches at a position wins).
_PII_COMBINED = re.compile("|".join(f"(?P<{k}>{p.pattern})" for k, p in PII_PATTERNS.items()))

# Cheap substring pre-checks (on the lowercased text): an injection rule's regex only runs
# if one of its trigger words is present, which skips almost all benign prompts.
_INJECTION_TRIGGERS: dict[str, tuple[str, ...]] = {
    "ignore_instructions": ("ignore", "disregard", "forget", "override"),
    "reveal_system_prompt": ("system", "hidden", "initial", "original"),
    "role_override": ("you are now", "you are no longer"),
    "jailbreak_keyword": ("jailbreak", "do anything now", "developer mode", "dan mode"),
    "delimiter_injection": ("<|", "<system", "[inst", "[/inst", "###"),
}


class _Hit(NamedTuple):
    kind: str
    start: int
    end: int


def _card_prefix_end(text: str, start: int, end: int) -> int | None:
    """End of the longest Luhn-valid card number at `start` inside a failed candidate.

    The card pattern is greedy, so "5500-0000-0000-0004 3" is one 17-digit candidate that
    fails the Luhn check although its first 16 digits are a valid card.
    """
    digit_ends = [i + 1 for i in range(start, end) if text[i].isdigit()]
    for n in range(len(digit_ends) - 1, 12, -1):
        stop = digit_ends[n - 1]
        if not text[stop : stop + 1].isdigit() and _luhn_ok(text[start:stop]):
            return stop
    return None


def _pii_matches(text: str, pos: int = 0) -> Iterator[_Hit]:
    """PII matches in `text` from `pos`, left to right, without overlaps.

    A digit run that looks like a card but fails the Luhn check is not consumed whole: a
    valid card at its start (followed by an unrelated number) is still found, and
    otherwise the scan resumes one character later, so a real entity inside it is found
    too. Without this, "in 2019 4111 1111 1111 1111" hid the card behind the failed
    candidate "2019 4111 1111 1111", and "412 123-45-6789" hid the SSN.
    """
    while match := _PII_COMBINED.search(text, pos):
        kind, start, end = match.lastgroup or "", match.start(), match.end()
        if kind == "CREDIT_CARD" and not _luhn_ok(match.group()):
            shorter = _card_prefix_end(text, start, end)
            if shorter is None:
                pos = start + 1
                continue
            end = shorter
        yield _Hit(kind, start, end)
        pos = end


def redact_pii(text: str) -> tuple[str, Counter[str]]:
    """Replace PII with `[REDACTED_<KIND>]` and count what was found."""
    found: Counter[str] = Counter()
    out, pos = [], 0
    for hit in _pii_matches(text):
        found[hit.kind] += 1
        out += (text[pos : hit.start], f"[REDACTED_{hit.kind}]")
        pos = hit.end
    if not found:
        return text, found
    out.append(text[pos:])
    return "".join(out), found


def detect_injection(text: str) -> list[str]:
    lowered = text.lower()
    return [
        name
        for name, pattern in INJECTION_PATTERNS.items()
        if any(t in lowered for t in _INJECTION_TRIGGERS[name]) and pattern.search(text)
    ]


@dataclass
class GuardrailReport:
    pii: Counter[str] = field(default_factory=Counter)
    injection: list[str] = field(default_factory=list)

    @property
    def header(self) -> str:
        parts = [f"pii_{k.lower()}={v}" for k, v in sorted(self.pii.items())]
        parts += [f"injection={name}" for name in self.injection]
        return ",".join(parts)


def _map_text(content: Any, fn) -> Any:
    """Apply `fn` to the text of a message's content (string or list of parts)."""
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        return [
            {**part, "text": fn(part["text"])}
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            else part
            for part in content
        ]
    return content


def scan_messages(
    messages: list[dict[str, Any]], redact: bool, check_injection: bool
) -> tuple[list[dict[str, Any]], GuardrailReport]:
    """Return (possibly redacted) messages and what was found.

    Injection checks run on user and tool messages only: system prompts written by the
    application legitimately contain instructions.
    """
    report = GuardrailReport()
    out: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if check_injection and message.get("role") in ("user", "tool"):

            def collect(text: str) -> str:
                report.injection.extend(
                    n for n in detect_injection(text) if n not in report.injection
                )
                return text

            _map_text(content, collect)
        if redact:

            def scrub(text: str) -> str:
                clean, found = redact_pii(text)
                report.pii.update(found)
                return clean

            message = {**message, "content": _map_text(content, scrub)}
        out.append(message)
    return out, report


# Characters a phone or card number can contain. A run of them at the end of the streamed
# text may still grow into a match, so it is held back along with the last word.
_NUMBERISH = frozenset("0123456789 ()+.-")
# The longest PII match that contains whitespace (a card with separators is 19 digits plus
# 3 separators; a phone with country code is shorter), so a held-back numeric run is
# bounded and text cannot be held indefinitely.
_MAX_SPACED_MATCH = 40
# An email address is at most 254 characters, so a longer unbroken run only needs its last
# 254 characters held back.
_MAX_EMAIL = 254
_MAX_HELD_WORD = 512
# Every PII pattern needs a digit or an "@", so text without either skips the regex scan.
_PII_HINT = re.compile(r"[\d@]")


class StreamRedactor:
    """Redacts PII in streamed text that arrives in arbitrary pieces.

    A model can emit an email or phone number split across several chunks
    (`"jane.d"`, `"oe@exam"`, `"ple.com"`), so redacting each chunk on its own misses it.
    `feed()` returns only text that no future chunk can turn into (or extend) a match: it
    holds back the trailing word, which may still grow into an email, IP or SSN, plus any
    trailing run of digits and separators that may grow into a phone or card number, and
    never cuts through a match. The held-back tail is usually one word, so the stream is
    delayed by about one token. `flush()` returns the rest when the stream ends.
    """

    # Already-emitted text kept in front of the pending text, so word boundaries and
    # lookbehinds at the start of the pending text see the real preceding characters.
    _CONTEXT = 64

    def __init__(self) -> None:
        self.found: Counter[str] = Counter()
        self._text = ""  # recent emitted text + the pending text
        self._done = 0  # offset in _text where the pending (not yet emitted) text starts

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._text += text
        return self._emit(self._safe_cut())

    def flush(self) -> str:
        return self._emit(len(self._text))

    def _safe_cut(self) -> int:
        text = self._text
        cut = len(text)
        while cut > self._done and not text[cut - 1].isspace():  # the last word may grow
            cut -= 1
        if len(text) - cut > _MAX_HELD_WORD:  # a huge "word" (base64, a URL): don't hold it all
            cut = len(text) - _MAX_EMAIL
        if all(c in _NUMBERISH for c in text[cut:]):  # a number may still be growing
            run_start = cut
            while run_start > self._done and text[run_start - 1] in _NUMBERISH:
                run_start -= 1
            if any(c.isdigit() for c in text[run_start:]):
                cut = min(cut, max(run_start, len(text) - _MAX_SPACED_MATCH))
        if not _PII_HINT.search(text, self._done):
            return cut
        for hit in _pii_matches(text, self._done):  # never cut through a match
            if hit.start >= cut:
                break
            if cut < hit.end:
                cut = hit.start
                break
        return cut

    def _emit(self, cut: int) -> str:
        if cut <= self._done:
            return ""
        text = self._text
        out, pos = [], self._done
        hits = _pii_matches(text, pos) if _PII_HINT.search(text, pos, cut) else ()
        # Scan the whole text from the pending offset (not a slice), so the lookarounds
        # see the characters on both sides of the pending region.
        for hit in hits:
            if hit.end > cut:
                break
            out.append(text[pos : hit.start])
            out.append(f"[REDACTED_{hit.kind}]")
            self.found[hit.kind] += 1
            pos = hit.end
        out.append(text[pos:cut])
        keep = max(0, cut - self._CONTEXT)
        self._text, self._done = text[keep:], cut - keep
        return "".join(out)


_dumps = json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).encode


class StreamGuard:
    """Output PII redaction for an OpenAI-style SSE stream: one `StreamRedactor` per choice.

    `rewrite()` takes a parsed chunk and returns its `data:` line with each choice's
    `delta.content` replaced by the text that is safe to send so far; a choice's held-back
    text is released when its `finish_reason` arrives. `flush_chunk()` returns one extra
    event with anything still held (for a stream that ends without a finish reason).
    """

    def __init__(self) -> None:
        self._choices: dict[int, StreamRedactor] = {}
        self._last: dict[str, Any] = {}  # the last chunk: flush_chunk copies its id, model, ...
        self.seconds = 0.0

    @property
    def found(self) -> Counter[str]:
        total: Counter[str] = Counter()
        for redactor in self._choices.values():
            total.update(redactor.found)
        return total

    def rewrite(self, chunk: dict[str, Any]) -> str:
        t0 = time.perf_counter()
        self._last = chunk
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            redactor = self._choices.get(index)
            if redactor is None:
                redactor = self._choices[index] = StreamRedactor()
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            safe = redactor.feed(content) if isinstance(content, str) else ""
            if choice.get("finish_reason"):
                safe += redactor.flush()
            if isinstance(content, str) or safe:
                choice["delta"] = {**(delta if isinstance(delta, dict) else {}), "content": safe}
        line = "data: " + _dumps(chunk)
        self.seconds += time.perf_counter() - t0
        return line

    def flush_chunk(self) -> bytes | None:
        choices = [
            {"index": index, "delta": {"content": tail}}
            for index, redactor in self._choices.items()
            if (tail := redactor.flush())
        ]
        if not choices:
            return None
        envelope = ("id", "object", "created", "model")
        event = {**{k: self._last[k] for k in envelope if k in self._last}, "choices": choices}
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def redact_response(response: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    """Redact PII in a chat completion's assistant messages."""
    found: Counter[str] = Counter()
    choices = []
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            clean, hits = redact_pii(content)
            found.update(hits)
            choice = {**choice, "message": {**message, "content": clean}}
        choices.append(choice)
    return ({**response, "choices": choices} if found else response), found
