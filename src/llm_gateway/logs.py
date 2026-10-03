"""Logging setup: readable text lines, or one JSON object per line for log shippers.

Access log records carry their structured fields in `record.fields` (request id,
status, latency, tenant, model, backend, cache); the JSON formatter emits them as
top-level keys, so a request can be found by the `x-request-id` header it returned.
"""

import json
import logging
from datetime import UTC, datetime

from llm_gateway.tracing import current_trace_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        trace_id = current_trace_id()
        if trace_id and "trace_id" not in entry:
            entry["trace_id"] = trace_id
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str, fmt: str = "text") -> None:
    """Configure the root logger once (called again, e.g. by tests, it is a no-op)."""
    root = logging.getLogger()
    if getattr(root, "_llmgw_configured", False):
        return
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers = [handler]
    root.setLevel(level.upper())
    # httpx logs every backend call at INFO; the access log already covers requests
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._llmgw_configured = True  # type: ignore[attr-defined]
