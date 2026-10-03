"""Backend selection with retries, fallback and a per-backend circuit breaker."""

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from opentelemetry.trace import Status, StatusCode, use_span

from llm_gateway.backends import BackendError
from llm_gateway.config import BackendConfig
from llm_gateway.tracing import get_tracer

T = TypeVar("T")

EWMA_ALPHA = 0.3


class Backend(Protocol):
    name: str
    config: BackendConfig

    def serves(self, model: str) -> bool: ...
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def chat_stream(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]: ...
    async def health(self) -> bool: ...
    async def aclose(self) -> None: ...


class NoBackendAvailable(Exception):
    def __init__(self, model: str, errors: list[BackendError]) -> None:
        detail = "; ".join(str(e) for e in errors) or "no configured backend serves this model"
        super().__init__(f"all backends failed for model '{model}': {detail}")
        self.model = model
        self.errors = errors


class Overloaded(Exception):
    """Every backend for the model was at its concurrency limit for the whole queue wait."""

    def __init__(self, model: str, waited: float) -> None:
        super().__init__(
            f"all backends for model '{model}' are at capacity (waited {waited:.1f} s)"
        )
        self.model = model
        self.waited = waited


@dataclass
class BackendState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    ewma_latency: float | None = None
    inflight: int = 0


class Router:
    def __init__(
        self,
        backends: list[Backend],
        policy: str = "cost",
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
        breaker_failures: int = 3,
        breaker_cooldown_seconds: float = 30.0,
        on_backend_error: Callable[[BackendError], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_concurrency: int = 0,
        queue_timeout_seconds: float = 30.0,
        on_queue_wait: Callable[[float], None] | None = None,
    ) -> None:
        self.backends = backends
        self.policy = policy
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.breaker_failures = breaker_failures
        self.breaker_cooldown_seconds = breaker_cooldown_seconds
        self.on_backend_error = on_backend_error
        self.clock = clock
        self.state: dict[str, BackendState] = {b.name: BackendState() for b in backends}
        # Per-backend cap on concurrent upstream calls (0 = no cap). A saturated backend is
        # skipped in favour of the next candidate; when all are saturated, requests queue
        # for up to queue_timeout_seconds and then fail fast with Overloaded (HTTP 503).
        self.max_concurrency = max_concurrency
        self.queue_timeout_seconds = queue_timeout_seconds
        self.on_queue_wait = on_queue_wait
        self._slot_freed: asyncio.Condition | None = None

    def is_available(self, backend: Backend) -> bool:
        return self.state[backend.name].open_until <= self.clock()

    def is_saturated(self, backend: Backend) -> bool:
        return 0 < self.max_concurrency <= self.state[backend.name].inflight

    @property
    def _cond(self) -> asyncio.Condition:
        if self._slot_freed is None:  # created lazily, inside the running event loop
            self._slot_freed = asyncio.Condition()
        return self._slot_freed

    def _acquire(self, backend: Backend) -> None:
        self.state[backend.name].inflight += 1

    async def _release(self, backend: Backend) -> None:
        self.state[backend.name].inflight -= 1
        if self.max_concurrency:
            async with self._cond:
                self._cond.notify_all()

    async def _wait_for_slot(self, model: str, pending: list[Backend], since: float) -> None:
        remaining = self.queue_timeout_seconds - (time.perf_counter() - since)
        try:
            if remaining <= 0:
                raise TimeoutError
            async with self._cond:
                await asyncio.wait_for(
                    self._cond.wait_for(lambda: any(not self.is_saturated(b) for b in pending)),
                    remaining,
                )
        except TimeoutError:
            waited = time.perf_counter() - since
            if self.on_queue_wait:
                self.on_queue_wait(waited)
            raise Overloaded(model, waited) from None

    def _score(self, backend: Backend) -> float:
        cfg = backend.config
        if self.policy == "latency":
            # Untried backends score 0 so they get measured.
            return self.state[backend.name].ewma_latency or 0.0
        if self.policy == "quality":
            return -cfg.quality
        return cfg.cost_per_1k_input + cfg.cost_per_1k_output

    def candidates(self, model: str) -> list[Backend]:
        """Backends serving `model`, best first. Open breakers go last, not away:
        if everything is down we still try rather than fail without a request."""
        serving = [b for b in self.backends if b.serves(model)]
        # sorted() is stable, so config order breaks ties.
        ranked = sorted(serving, key=self._score)
        return [b for b in ranked if self.is_available(b)] + [
            b for b in ranked if not self.is_available(b)
        ]

    def record_success(self, backend: Backend, latency: float) -> None:
        state = self.state[backend.name]
        state.consecutive_failures = 0
        state.open_until = 0.0
        state.ewma_latency = (
            latency
            if state.ewma_latency is None
            else EWMA_ALPHA * latency + (1 - EWMA_ALPHA) * state.ewma_latency
        )

    def record_failure(self, backend: Backend, error: BackendError) -> None:
        state = self.state[backend.name]
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.breaker_failures:
            state.open_until = self.clock() + self.breaker_cooldown_seconds
        if self.on_backend_error:
            self.on_backend_error(error)

    async def _attempts(self, model: str) -> AsyncIterator[tuple[Backend, int]]:
        pending = self.candidates(model)
        queued_since: float | None = None
        while pending:
            free = [b for b in pending if not self.is_saturated(b)]
            if not free:
                queued_since = queued_since or time.perf_counter()
                await self._wait_for_slot(model, pending, queued_since)
                continue
            if queued_since is not None and self.on_queue_wait:
                self.on_queue_wait(time.perf_counter() - queued_since)
                queued_since = None
            backend = free[0]
            pending.remove(backend)
            for attempt in range(self.max_retries + 1):
                if attempt:
                    await asyncio.sleep(self.retry_backoff_seconds * 2 ** (attempt - 1))
                yield backend, attempt

    async def run(self, model: str, op: Callable[[Backend], Awaitable[T]]) -> tuple[T, Backend]:
        """Run `op` on the best backend, retrying and then falling back on failure."""
        errors: list[BackendError] = []
        skip: set[str] = set()
        async for backend, attempt in self._attempts(model):
            if backend.name in skip:
                continue
            start = time.perf_counter()
            span = get_tracer().start_span(
                "router.attempt",
                attributes={
                    "llmgw.backend": backend.name,
                    "llmgw.attempt": attempt,
                    "gen_ai.request.model": model,
                },
            )
            self._acquire(backend)
            try:
                with use_span(span, end_on_exit=False):
                    result = await op(backend)
            except BackendError as exc:
                await self._release(backend)
                span.set_status(Status(StatusCode.ERROR, exc.kind))
                span.end()
                if exc.is_client_error:
                    raise
                self.record_failure(backend, exc)
                errors.append(exc)
                if not self.is_available(backend):
                    skip.add(backend.name)
                continue
            except BaseException:
                await self._release(backend)
                span.end()
                raise
            await self._release(backend)
            span.end()
            self.record_success(backend, time.perf_counter() - start)
            return result, backend
        raise NoBackendAvailable(model, errors)

    async def open_stream(
        self, model: str, payload: dict[str, Any]
    ) -> tuple[Backend, AsyncGenerator[str, None], float]:
        """Start a stream, falling back until one backend yields its first chunk.

        Returns the backend, the full line iterator, and time-to-first-token (seconds).
        Once the first chunk is out we are committed to that backend.
        """
        errors: list[BackendError] = []
        skip: set[str] = set()
        async for backend, attempt in self._attempts(model):
            if backend.name in skip:
                continue
            start = time.perf_counter()
            span = get_tracer().start_span(
                "router.attempt",
                attributes={
                    "llmgw.backend": backend.name,
                    "llmgw.attempt": attempt,
                    "llmgw.stream": True,
                    "gen_ai.request.model": model,
                },
            )
            stream = backend.chat_stream(payload)
            self._acquire(backend)
            try:
                with use_span(span, end_on_exit=False):
                    first = await anext(stream)
            except StopAsyncIteration:
                exc = BackendError(backend.name, "stream ended before any data", 502)
            except BackendError as err:
                if err.is_client_error:
                    await self._release(backend)
                    await stream.aclose()
                    span.end()
                    raise
                exc = err
            except BaseException:
                await self._release(backend)
                await stream.aclose()
                span.end()
                raise
            else:
                ttft = time.perf_counter() - start
                span.set_attribute("llmgw.ttft_ms", ttft * 1000)
                span.end()
                self.record_success(backend, ttft)
                return backend, _prepend(first, stream, lambda b=backend: self._release(b)), ttft
            await self._release(backend)
            span.set_status(Status(StatusCode.ERROR, exc.kind))
            span.end()
            await stream.aclose()
            self.record_failure(backend, exc)
            errors.append(exc)
            if not self.is_available(backend):
                skip.add(backend.name)
        raise NoBackendAvailable(model, errors)


async def _prepend(
    first: str,
    rest: AsyncGenerator[str, None],
    on_close: Callable[[], Awaitable[None]],
) -> AsyncGenerator[str, None]:
    """The stream with its first line put back; frees the backend slot when it ends."""
    try:
        yield first
        async for item in rest:
            yield item
    finally:
        try:
            await rest.aclose()
        finally:
            await on_close()
