from __future__ import annotations

from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.task import TaskRequest


class StaticPlanner:
    def __init__(self, *, skill_name: str = "NavigationSkill", timeout_s: float = 5.0) -> None:
        self.skill_name = skill_name
        self.timeout_s = timeout_s

    async def plan(self, task: TaskRequest, observation: Observation) -> Plan:
        del observation
        skill_inputs = task.metadata.get("skill_inputs", {})
        if not isinstance(skill_inputs, dict):
            raise TypeError("task.metadata.skill_inputs must be a mapping")
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=task.trace.plan_revision,
            rationale_summary="MVP static plan: dispatch one bounded navigation skill.",
            steps=[
                PlanStep(
                    step_id="nav-001",
                    skill_name=self.skill_name,
                    command=task.instruction,
                    success_conditions=[
                        "Navigation skill returns a compressed result.",
                        "Skill verifier accepts local stop or trajectory evidence.",
                    ],
                    timeout_s=self.timeout_s,
                    max_retries=0,
                    inputs={"goal_text": task.instruction, **skill_inputs},
                )
            ],
        )
