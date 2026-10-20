from __future__ import annotations

from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel


class PlanStep(SchemaModel):
    step_id: str = Field(min_length=1)
    skill_name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    success_conditions: list[str] = Field(default_factory=list)
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=0, ge=0)
    inputs: dict[str, Any] = Field(default_factory=dict)


class Plan(SchemaModel):
    task_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=0)
    rationale_summary: str = ""
    steps: list[PlanStep] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

