from __future__ import annotations

from datetime import datetime

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now


class SkillRunnerSession(SchemaModel):
    """Local state for one delegated procedural subtask."""

    task_id: str = Field(min_length=1)
    skill_run_id: str = Field(min_length=1)
    skill_name: str = Field(min_length=1)
    subgoal: str = Field(min_length=1)
    success_conditions: list[str] = Field(default_factory=list)
    attempt: int = Field(default=0, ge=0)
    source_observation_id: str = Field(min_length=1)
    progress_summaries: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

