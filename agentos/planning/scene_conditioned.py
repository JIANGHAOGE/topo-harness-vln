from __future__ import annotations

from typing import Any

from pydantic import Field

from agentos.memory.store import GraphMemoryStore
from agentos.providers.base import LLMProvider
from agentos.providers.vision import audit_observation_views, audited_observation_content
from agentos.schemas.base import SchemaModel
from agentos.schemas.capability import SkillDescriptor
from agentos.schemas.llm import ChatMessage
from agentos.schemas.memory import MemoryNodeType, MemoryQuery
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest


class _ProposedStep(SchemaModel):
    skill_name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    success_conditions: list[str] = Field(min_length=1)
    timeout_s: float = Field(default=300.0, gt=0.0, le=3600.0)
    max_retries: int = Field(default=0, ge=0, le=3)
    inputs: dict[str, Any] = Field(default_factory=dict)


class _ProposedPlan(SchemaModel):
    task_type: str = Field(min_length=1)
    rationale_summary: str = Field(min_length=1)
    task_success_conditions: list[str] = Field(min_length=1)
    steps: list[_ProposedStep] = Field(min_length=1, max_length=6)


class SceneConditionedPlanner:
    """ABot-AgentOS-style main-agent planning over scene, memory, and skills."""

    def __init__(
        self,
        provider: LLMProvider,
        skills: list[SkillDescriptor],
        *,
        memory: GraphMemoryStore | None = None,
        max_memory_hits: int = 6,
    ) -> None:
        self.provider = provider
        self.skills = {skill.name: skill for skill in skills}
        self.memory = memory
        self.max_memory_hits = max_memory_hits

    async def plan(self, task: TaskRequest, observation: Observation) -> Plan:
        return await self._generate(task, observation, plan_revision=task.trace.plan_revision)

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
        prior = "\n".join(
            f"- {item.skill_name}: {item.status.value}; {item.compressed_history or item.summary}"
            for item in skill_results[-6:]
        )
        recovery = (
            f"Previous plan revision: {previous_plan.plan_revision}\n"
            f"Verification failure: {reason}\n"
            f"Recent skill results:\n{prior or '- none'}\n"
            "Produce a materially revised recovery plan. Do not repeat a failed step unchanged."
        )
        return await self._generate(task, observation, plan_revision=next_revision, recovery=recovery)

    async def _generate(
        self,
        task: TaskRequest,
        observation: Observation,
        *,
        plan_revision: int,
        recovery: str = "",
    ) -> Plan:
        memory = self._memory_summary(task)
        capability_text = "\n".join(
            f"- {name}: {descriptor.description}; modalities={descriptor.required_modalities}; "
            f"side_effects={descriptor.side_effects}"
            for name, descriptor in self.skills.items()
        )
        prompt = (
            f"Original user task: {task.instruction}\n"
            f"Declared task success conditions: {task.success_conditions or ['infer explicit verifiable conditions']}\n"
            f"Available embodied skills:\n{capability_text}\n"
            f"Retrieved source-grounded memory (not current truth):\n{memory}\n"
            f"{recovery}\n"
            "Classify the task (navigation, search, interaction, reporting, manipulation, or observation). "
            "Create a short revisable high-level plan grounded in the current camera observations. "
            "Every step must select an available skill and include observable semantic completion conditions. "
            "A skill RPC success or OmniNav local STOP is evidence only, never global task completion. "
            "Use robot camera identity, not left/right position within an image."
        )
        view_audit = await audit_observation_views(self.provider, observation)
        response = await self.provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are the deliberative Main Agent in ABot-AgentOS. Plan only high-level embodied "
                        "subgoals; low-level motion belongs to isolated Skill Runners. Return strict JSON."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=audited_observation_content(prompt, observation, view_audit),
                ),
            ],
            response_schema=_ProposedPlan.model_json_schema(),
        )
        if not isinstance(response.parsed_json, dict):
            raise ValueError("Scene planner did not return a JSON object")
        proposed = _ProposedPlan.model_validate(response.parsed_json)
        unknown = sorted({step.skill_name for step in proposed.steps} - set(self.skills))
        if unknown:
            raise ValueError(f"Scene planner selected unknown skills: {unknown}")
        steps = [
            PlanStep(
                step_id=f"r{plan_revision}-step-{index:03d}",
                skill_name=step.skill_name,
                command=step.command,
                success_conditions=step.success_conditions,
                timeout_s=step.timeout_s,
                max_retries=step.max_retries,
                inputs={"instruction": step.command, **step.inputs},
            )
            for index, step in enumerate(proposed.steps, start=1)
        ]
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=plan_revision,
            rationale_summary=proposed.rationale_summary,
            steps=steps,
            metadata={
                "task_type": proposed.task_type,
                "task_success_conditions": proposed.task_success_conditions,
                "planner_model": response.model,
            },
        )

    def _memory_summary(self, task: TaskRequest) -> str:
        if self.memory is None:
            return "- none"
        hits = self.memory.search(
            MemoryQuery(
                text=task.instruction,
                node_types=[MemoryNodeType.EVIDENCE, MemoryNodeType.PLACE, MemoryNodeType.SEMANTIC_EVENT],
                limit=self.max_memory_hits,
            )
        )
        if not hits:
            return "- none"
        return "\n".join(
            f"- {hit.node.content or hit.node.title} [source={hit.node.source_ref}]" for hit in hits
        )
