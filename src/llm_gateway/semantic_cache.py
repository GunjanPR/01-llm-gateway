"""Semantic response cache: reuse an answer when a new prompt *means* the same thing.

Two stages, because embedding similarity alone is not safe: on our labelled pairs
"Convert 100 Fahrenheit to Celsius" vs "...Celsius to Fahrenheit" scores 0.996 cosine,
higher than most true paraphrases (see eval/semantic_pairs.jsonl and the README).

1. **Candidate:** embed the last user message (`nomic-embed-text`, `search_query:` prefix)
   and run a KNN search in Redis (RediSearch HNSW, cosine), restricted to entries with the
   same *scope*: model, sampling parameters and the whole conversation before that
   message. A candidate needs similarity >= `SEMANTIC_CACHE_THRESHOLD`.
2. **Verify:** a small LLM judges whether the two requests need the same answer (few-shot,
   YES/NO). Only a YES is served from cache.

Any failure (Redis without the search module, embedding or verifier errors) means a
cache miss, never a failed request.
"""

import hashlib
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import RedisError, ResponseError

from llm_gateway.backends import BackendError
from llm_gateway.cache import IGNORED_FIELDS
from llm_gateway.router import NoBackendAvailable, Router
from llm_gateway.tracing import get_tracer

logger = logging.getLogger(__name__)

INDEX_NAME = "llmgw:sem:idx"
KEY_PREFIX = "llmgw:sem:"
QUERY_PREFIX = "search_query: "

VERIFY_SYSTEM = (
    "Decide if two questions are paraphrases: they ask for the same thing, so one answer "
    "fits both. Different wording is fine. Any change in entity, number, unit direction, "
    "or task means NO. Answer YES or NO only."
)
# Few-shot examples; deliberately not taken from the evaluation pairs.
VERIFY_SHOTS = [
    ("How tall is Mount Everest?", "What is the height of Mount Everest?", "YES"),
    ("How tall is Mount Everest?", "How tall is K2?", "NO"),
    ("Convert 5 miles to km.", "How many kilometers is 5 miles?", "YES"),
    ("Convert 5 miles to km.", "Convert 5 km to miles.", "NO"),
    ("Give tips for better sleep.", "How can I sleep better?", "YES"),
    ("Give tips for better sleep.", "Give tips for better focus.", "NO"),
]


class VectorIndex(Protocol):
    async def search(
        self, scope: str, vector: list[float]
    ) -> tuple[float, dict[str, Any]] | None: ...
    async def add(
        self, scope: str, vector: list[float], entry: dict[str, Any], ttl_seconds: int
    ) -> None: ...


class RedisVectorIndex:
    """HNSW cosine index over Redis hashes (needs Redis 8+ or Redis Stack)."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        self._dim: int | None = None

    async def _ensure(self, dim: int) -> None:
        if self._dim == dim:
            return
        try:
            await self.redis.ft(INDEX_NAME).create_index(
                [
                    TagField("scope"),
                    VectorField(
                        "vec",
                        "HNSW",
                        {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH),
            )
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._dim = dim

    async def search(self, scope: str, vector: list[float]) -> tuple[float, dict[str, Any]] | None:
        await self._ensure(len(vector))
        query = (
            Query(f"(@scope:{{{scope}}})=>[KNN 1 @vec $v AS dist]")
            .sort_by("dist")
            .return_fields("dist", "entry")
            .dialect(2)
        )
        result = await self.redis.ft(INDEX_NAME).search(query, query_params={"v": _pack(vector)})
        if not result.docs:
            return None
        doc = result.docs[0]
        return 1.0 - float(doc.dist), json.loads(doc.entry)

    async def add(
        self, scope: str, vector: list[float], entry: dict[str, Any], ttl_seconds: int
    ) -> None:
        await self._ensure(len(vector))
        key = f"{KEY_PREFIX}{uuid.uuid4().hex}"
        mapping = {"scope": scope, "vec": _pack(vector), "entry": json.dumps(entry)}
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, ttl_seconds)
            await pipe.execute()


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def split_request(payload: dict[str, Any], namespace: str = "") -> tuple[str, str] | None:
    """(scope, query text) for a request, or None if it can't be semantically cached."""
    messages = payload.get("messages") or []
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        return None
    query = _text(messages[-1].get("content")).strip()
    if not query:
        return None
    context = {k: v for k, v in payload.items() if k not in IGNORED_FIELDS | {"messages"}}
    context["history"] = messages[:-1]
    context["__namespace__"] = namespace
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32], query


@dataclass
class SemanticLookup:
    scope: str | None = None
    query: str | None = None
    vector: list[float] | None = None
    entry: dict[str, Any] | None = None  # set on a verified hit
    similarity: float | None = None
    outcome: str = "skipped"  # hit | no_candidate | rejected | error | skipped


class SemanticCache:
    def __init__(
        self,
        index: VectorIndex,
        router: Router,
        embed_model: str,
        verify_model: str,
        threshold: float,
        ttl_seconds: int,
        metrics: Any,
    ) -> None:
        self.index = index
        self.router = router
        self.embed_model = embed_model
        self.verify_model = verify_model
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.metrics = metrics

    async def _embed(self, text: str) -> list[float]:
        body = {"model": self.embed_model, "input": QUERY_PREFIX + text}
        start = time.perf_counter()
        with get_tracer().start_as_current_span("semantic_cache.embed"):
            response, _ = await self.router.run(self.embed_model, lambda b: b.embeddings(body))
        self.metrics.semantic_seconds.labels("embed").observe(time.perf_counter() - start)
        return response["data"][0]["embedding"]

    async def _verify(self, cached_query: str, query: str) -> bool:
        messages: list[dict[str, str]] = [{"role": "system", "content": VERIFY_SYSTEM}]
        for a, b, answer in VERIFY_SHOTS:
            messages.append({"role": "user", "content": f"Q1: {a}\nQ2: {b}"})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"Q1: {cached_query}\nQ2: {query}"})
        body = {"model": self.verify_model, "messages": messages, "temperature": 0, "max_tokens": 2}
        start = time.perf_counter()
        with get_tracer().start_as_current_span("semantic_cache.verify") as span:
            response, _ = await self.router.run(self.verify_model, lambda b: b.chat(body))
        self.metrics.semantic_seconds.labels("verify").observe(time.perf_counter() - start)
        choices = response.get("choices") or [{}]
        answer = (choices[0].get("message") or {}).get("content") or ""
        verdict = answer.strip().upper().startswith("YES")
        span.set_attribute("llmgw.semantic.verdict", verdict)
        return verdict

    async def lookup(self, payload: dict[str, Any], namespace: str = "") -> SemanticLookup:
        split = split_request(payload, namespace)
        if split is None:
            return SemanticLookup()
        result = SemanticLookup(scope=split[0], query=split[1])
        try:
            result.vector = await self._embed(result.query)
            start = time.perf_counter()
            with get_tracer().start_as_current_span("semantic_cache.search") as span:
                found = await self.index.search(result.scope, result.vector)
                if found:
                    span.set_attribute("llmgw.semantic.similarity", found[0])
            self.metrics.semantic_seconds.labels("search").observe(time.perf_counter() - start)
            if found is None or found[0] < self.threshold:
                result.outcome = "no_candidate"
                return result
            result.similarity, entry = found
            self.metrics.semantic_similarity.observe(result.similarity)
            if self.verify_model and not await self._verify(entry["query"], result.query):
                result.outcome = "rejected"
                return result
            result.entry, result.outcome = entry, "hit"
        except (RedisError, BackendError, NoBackendAvailable, KeyError, ValueError) as exc:
            logger.warning("semantic cache lookup failed, treating as miss: %s", exc)
            result.outcome = "error"
        finally:
            self.metrics.semantic_cache.labels(result.outcome).inc()
        return result

    async def store(self, lookup: SemanticLookup, entry: dict[str, Any]) -> None:
        if lookup.vector is None or lookup.scope is None:
            return
        try:
            await self.index.add(
                lookup.scope, lookup.vector, {**entry, "query": lookup.query}, self.ttl_seconds
            )
        except RedisError as exc:
            logger.warning("semantic cache store failed: %s", exc)
