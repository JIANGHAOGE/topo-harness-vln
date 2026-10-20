from __future__ import annotations

from agentos.planning.base import Planner, Replanner
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest


class MainAgent:
    """Own the global task and plan without local Skill Runner history."""

    def __init__(self, planner: Planner, replanner: Replanner | None = None) -> None:
        self.planner = planner
        self.replanner = replanner

    async def plan(self, task: TaskRequest, observation: Observation) -> Plan:
        return await self.planner.plan(task, observation)

    async def replan(
        self,
        task: TaskRequest,
        observation: Observation,
        previous_plan: Plan,
        skill_results: list[SkillResult],
        *,
        reason: str,
        next_revision: int,
    ) -> Plan | None:
        if self.replanner is None:
            return None
        return await self.replanner.replan(
            task,
            observation,
            previous_plan,
            skill_results,
            reason=reason,
            next_revision=next_revision,
        )
