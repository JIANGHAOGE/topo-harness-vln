from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now
from agentos.schemas.enums import EventType, RuntimeState
from agentos.schemas.trace import TraceContext


class Event(SchemaModel):
    event_id: str = Field(min_length=1)
    event_type: EventType
    trace: TraceContext
    state: RuntimeState
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

