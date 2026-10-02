"""Request/response guardrails: PII redaction and prompt-injection detection.

Both are deterministic, rule-based checks that run in-process before any model call, so
they add microseconds, not a second model round trip. They are a first line of defence,
not a guarantee: regexes miss novel PII formats, and the injection rules catch common
phrasings ("ignore previous instructions", "reveal your system prompt", ...), not every
adversarial prompt.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

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


def redact_pii(text: str) -> tuple[str, Counter[str]]:
    """Replace PII with `[REDACTED_<KIND>]` and count what was found."""
    found: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        kind = match.lastgroup or ""
        if kind == "CREDIT_CARD" and not _luhn_ok(match.group()):
            return match.group()
        found[kind] += 1
        return f"[REDACTED_{kind}]"

    return _PII_COMBINED.sub(replace, text), found


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
