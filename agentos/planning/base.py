from __future__ import annotations

from typing import Protocol

from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest


class Planner(Protocol):
    async def plan(self, task: TaskRequest, observation: Observation) -> Plan:
        ...


class Replanner(Protocol):
    async def replan(
        self,
        task: TaskRequest,
        observation: Observation,
        previous_plan: Plan,
        skill_results: list[SkillResult],
        *,
        reason: str,
        next_revision: int,
    ) -> Plan:
        ...
