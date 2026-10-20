from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now
from agentos.schemas.enums import SkillStatus
from agentos.schemas.trace import TraceContext


class RetryPolicy(SchemaModel):
    max_retries: int = Field(default=0, ge=0)
    retry_on_timeout: bool = False


class Evidence(SchemaModel):
    kind: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    ref: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class SkillRequest(SchemaModel):
    trace: TraceContext
    skill_name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    success_conditions: list[str] = Field(default_factory=list)
    context_summary: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = Field(default=30.0, gt=0.0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class SkillResult(SchemaModel):
    trace: TraceContext
    skill_name: str = Field(min_length=1)
    status: SkillStatus
    summary: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    compressed_history: str | None = None
    stop_reason: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    error: str | None = None

