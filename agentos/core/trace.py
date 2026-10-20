from __future__ import annotations

from datetime import datetime

from agentos.schemas.trace import TraceContext


def is_current_trace(active: TraceContext | None, candidate: TraceContext, now: datetime | None = None) -> bool:
    if active is None:
        return False
    return active.same_async_source(candidate) and not candidate.is_expired(now)

