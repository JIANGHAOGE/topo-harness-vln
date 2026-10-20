from __future__ import annotations

from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel
from agentos.schemas.trace import TraceContext


class TaskRequest(SchemaModel):
    trace: TraceContext
    instruction: str = Field(min_length=1)
    user_id: str | None = None
    priority: int = Field(default=5, ge=0, le=10)
    success_conditions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
