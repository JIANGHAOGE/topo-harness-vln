from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentos.schemas.task_reasoning import (
    AgentOSSkillDelegation,
    NavigationSlowBrainOutput,
    SlowBrainTaskSpecification,
)


@dataclass
class AgentOSMainAgentSession:
    """Global planning state that never owns Skill-local execution history."""

    global_goal: str
    context_snapshot: dict[str, Any] | None = None
    last_delegation: AgentOSSkillDelegation | None = None
    main_agent_calls: int = 0
    active_skill_plan: list[dict[str, Any]] = field(default_factory=list)
    delegated_subtask_index: int | None = None
    progress_ledger: list[dict[str, Any]] = field(default_factory=list)
    mapping_observation_id: str | None = None
    region_search_state: dict[str, Any] | None = None

    def record_plan(self, delegation: AgentOSSkillDelegation) -> None:
        self.last_delegation = delegation
        self.active_skill_plan = [
            item.model_dump(mode="json") for item in delegation.planner_output.subtasks
        ]
        self.delegated_subtask_index = delegation.delegated_subtask_index
        region_search = delegation.planner_output.region_search
        self.region_search_state = (
            region_search.model_dump(mode="json") if region_search is not None else None
        )

    def record_outcome(
        self,
        *,
        observation_id: str,
        skill_name: str,
        status: str,
        summary: str,
        verifier_feedback: str,
        delegated_subtask: dict[str, Any] | None,
        verified_success: bool = False,
    ) -> None:
        self.progress_ledger.append(
            {
                "observation_id": observation_id,
                "skill_name": skill_name,
                "status": status,
                "summary": summary,
                "verifier_feedback": verifier_feedback,
                "delegated_subtask": delegated_subtask,
                "verified_success": verified_success,
            }
        )
        self.progress_ledger = self.progress_ledger[-8:]
        if skill_name == "MappingSkill" and status == "success":
            self.mapping_observation_id = observation_id


@dataclass
class N1NavigationSkillSession:
    """Private ABot-N1 state retained only by the Navigation Skill Runner."""

    global_goal: str
    handoff: dict[str, Any] | None = None
    active_task_specification: SlowBrainTaskSpecification | None = None
    previous_decision: NavigationSlowBrainOutput | None = None
    previous_decision_observation_id: str | None = None
    skill_runs: int = 0
    decision_count: int = 0
    recent_slow_outputs: list[dict[str, Any]] = field(default_factory=list)

    def bind_task(self, specification: SlowBrainTaskSpecification) -> bool:
        """Reset temporal feedback when AgentOS delegates a different N1 task g."""

        changed = self.active_task_specification != specification
        if changed:
            self.active_task_specification = specification
            self.previous_decision = None
            self.previous_decision_observation_id = None
            self.recent_slow_outputs.clear()
        return changed

    def record_slow_output(
        self,
        output: NavigationSlowBrainOutput,
        *,
        source_observation_id: str | None = None,
    ) -> None:
        self.previous_decision = output
        self.previous_decision_observation_id = source_observation_id
        self.decision_count += 1
        self.recent_slow_outputs.append(output.model_dump(mode="json"))
        self.recent_slow_outputs = self.recent_slow_outputs[-4:]
