"""Per-request usage ledger in PostgreSQL.

Every request (success, cache hit, rejection or error) becomes one row in `usage_events`.
Writes never block a response: rows go onto an in-memory queue and a background task
inserts them in batches. If PostgreSQL is down, rows stay queued up to `max_queue`;
beyond that the oldest are dropped and counted in `llmgw_ledger_dropped_total`.
"""

import asyncio
import contextlib
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import astuple, dataclass, field, fields
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                uuid PRIMARY KEY,
    ts                timestamptz NOT NULL,
    tenant            text NOT NULL,
    route             text NOT NULL,
    model             text NOT NULL,
    backend           text,
    cache             text,
    status            integer NOT NULL,
    prompt_tokens     integer NOT NULL DEFAULT 0,
    completion_tokens integer NOT NULL DEFAULT 0,
    cost_usd          double precision NOT NULL DEFAULT 0,
    latency_ms        double precision NOT NULL,
    guardrails        text
);
CREATE INDEX IF NOT EXISTS usage_events_tenant_ts ON usage_events (tenant, ts DESC);
"""

SUMMARY_SQL = """
SELECT model,
       count(*)                                                   AS requests,
       count(*) FILTER (WHERE status >= 400)                      AS errors,
       count(*) FILTER (WHERE cache IN ('HIT', 'SEMANTIC_HIT'))   AS cache_hits,
       coalesce(sum(prompt_tokens), 0)                            AS prompt_tokens,
       coalesce(sum(completion_tokens), 0)                        AS completion_tokens,
       coalesce(sum(cost_usd), 0)                                 AS cost_usd,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)    AS latency_p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)   AS latency_p95_ms
FROM usage_events
WHERE tenant = $1 AND ts >= $2
GROUP BY model
ORDER BY requests DESC
"""


@dataclass
class UsageEvent:
    tenant: str
    route: str
    model: str
    status: int
    latency_ms: float
    backend: str | None = None
    cache: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    guardrails: str | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


COLUMNS = [f.name for f in fields(UsageEvent)]
Writer = Callable[[list[UsageEvent]], Awaitable[None]]


class UsageLedger:
    def __init__(
        self,
        writer: Writer,
        metrics: Any,
        batch_size: int = 200,
        flush_interval_seconds: float = 1.0,
        max_queue: int = 10_000,
    ) -> None:
        self.writer = writer
        self.metrics = metrics
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.queue: deque[UsageEvent] = deque()
        self.max_queue = max_queue
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None

    def record(self, event: UsageEvent) -> None:
        if len(self.queue) >= self.max_queue:
            self.queue.popleft()
            self.metrics.ledger_dropped.inc()
        self.queue.append(event)
        if len(self.queue) >= self.batch_size:
            self._wakeup.set()

    async def flush(self) -> int:
        """Write everything queued, in batches. Returns rows written."""
        written = 0
        while self.queue:
            batch = [self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))]
            try:
                await self.writer(batch)
            except Exception as exc:  # DB down or bad row: keep the rows, retry later
                self.queue.extendleft(reversed(batch))
                self.metrics.ledger_errors.inc()
                logger.warning(
                    "usage ledger write failed (%d rows queued): %s", len(self.queue), exc
                )
                break
            written += len(batch)
            self.metrics.ledger_rows.inc(len(batch))
        return written

    async def _run(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), self.flush_interval_seconds)
            self._wakeup.clear()
            await self.flush()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.flush()


class PostgresStore:
    """asyncpg pool: schema setup, batched COPY inserts, per-tenant summaries."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.pool = None

    async def connect(self) -> None:
        import asyncpg  # imported here so the gateway runs without the ledger configured

        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4, timeout=5)
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def write(self, batch: list[UsageEvent]) -> None:
        if self.pool is None:
            await self.connect()
        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(
                "usage_events", records=[astuple(e) for e in batch], columns=COLUMNS
            )

    async def summary(self, tenant: str, since: datetime) -> list[dict[str, Any]]:
        if self.pool is None:
            await self.connect()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(SUMMARY_SQL, tenant, since)
        return [dict(r) for r in rows]

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
