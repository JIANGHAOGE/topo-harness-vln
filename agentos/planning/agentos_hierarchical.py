from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import ValidationError

from agentos.harness.navigation_session import AgentOSMainAgentSession
from agentos.memory.store import GraphMemoryStore
from agentos.planning.semantic_nodes import SemanticNodeRanker
from agentos.providers.base import LLMProvider
from agentos.providers.vision import audit_observation_views, audited_observation_content
from agentos.schemas.capability import SkillDescriptor
from agentos.schemas.llm import ChatMessage, LLMResponse
from agentos.schemas.memory import MemoryNodeType, MemoryQuery
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.task_reasoning import (
    AgentOSSkillDelegation,
    RegionSearchPlan,
    SemanticTaskDecompositionInput,
)


class AgentOSHierarchicalPlanner:
    """AgentOS Main Agent: plan and delegate without selecting local actions."""

    skill_name = "NavigationSkill"

    def __init__(
        self,
        provider: LLMProvider,
        service,
        session: AgentOSMainAgentSession,
        skill_catalog: list[SkillDescriptor] | None = None,
        memory: GraphMemoryStore | None = None,
        semantic_node_ranker: SemanticNodeRanker | None = None,
    ) -> None:
        self.provider = provider
        self.service = service
        self.session = session
        self.skill_catalog = skill_catalog or []
        self.memory = memory
        self.semantic_node_ranker = semantic_node_ranker
        self.skill_names = {
            descriptor.name for descriptor in self.skill_catalog
        } or {"MappingSkill", "NavigationSkill"}

    async def plan(self, task: TaskRequest, observation: Observation) -> Plan:
        if self.session.context_snapshot is None:
            packet = await asyncio.to_thread(
                self.service.begin_navigation,
                {"global_goal": task.instruction},
            )
            self.session.context_snapshot = compress_navigation_handoff(packet)
        return await self._delegate(task, observation, task.trace.plan_revision)

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
        verified_success = reason.startswith(
            ("MappingSkill succeeded.", "Delegated Skill succeeded.")
        )
        if skill_results:
            latest = skill_results[-1]
            snapshot = compress_navigation_handoff(latest.outputs or {})
            if snapshot.get("observation") or snapshot.get("slow_context"):
                previous = self.session.context_snapshot or {}
                self.session.context_snapshot = {
                    key: value if value not in (None, [], {}) else previous.get(key)
                    for key, value in snapshot.items()
                }
            self.session.record_outcome(
                observation_id=observation.trace.observation_id,
                skill_name=latest.skill_name,
                status=latest.status.value,
                summary=latest.compressed_history or latest.summary,
                verifier_feedback=reason,
                delegated_subtask=previous_plan.metadata.get("delegated_subtask"),
                verified_success=verified_success,
            )
        mapping_was_delegated = (
            previous_plan.metadata.get("delegated_skill") == "MappingSkill"
        )
        if verified_success and not mapping_was_delegated:
            advanced = self._advance_accepted_plan()
            if advanced is not None:
                return self._plan_from_delegation(
                    task,
                    observation,
                    next_revision,
                    advanced,
                    verifier_feedback=reason,
                    planner_model=str(previous_plan.metadata.get("planner_model") or "accepted-plan"),
                )
        return await self._delegate(
            task,
            observation,
            next_revision,
            verifier_feedback=reason,
            skill_results=skill_results,
        )

    async def _delegate(
        self,
        task: TaskRequest,
        observation: Observation,
        revision: int,
        *,
        verifier_feedback: str = "",
        skill_results: list[SkillResult] | None = None,
    ) -> Plan:
        handoff = self.session.context_snapshot or {}
        current_observation = observation
        topology_context = (handoff.get("slow_context") or {}).get(
            "topology_context"
        )
        context = {
            "stable_global_task": task.instruction,
            "skill_catalog": [
                {
                    "name": descriptor.name,
                    "description": descriptor.description,
                    "risk_level": descriptor.risk_level.value,
                    "side_effects": descriptor.side_effects,
                }
                for descriptor in self.skill_catalog
            ],
            "topo_memory": (
                topology_context
                if self.session.mapping_observation_id is not None
                else None
            ),
            "semantic_route_plan": (
                handoff.get("semantic_route_plan")
                if self.session.mapping_observation_id is not None
                else None
            ),
            "active_skill_plan": self.session.active_skill_plan,
            "active_delegated_subtask_index": self.session.delegated_subtask_index,
            "region_search_state": self.session.region_search_state,
            "region_candidate_progress": self._region_candidate_progress(
                topology_context
            ),
            "progress_ledger": self.session.progress_ledger,
            "mapping_evidence": {
                "available": self.session.mapping_observation_id is not None,
                "observation_id": self.session.mapping_observation_id,
                "matches_current_observation": (
                    self.session.mapping_observation_id
                    == current_observation.trace.observation_id
                ),
            },
            "previous_skill_results": [
                {
                    "skill_name": item.skill_name,
                    "status": item.status.value,
                    "summary": item.compressed_history or item.summary,
                }
                for item in (skill_results or [])[-3:]
            ],
            "verifier_feedback": verifier_feedback or None,
            "long_term_memory": self._retrieve_long_term_memory(
                task.instruction,
                map_id=(
                    str(topology_context.get("map_id"))
                    if isinstance(topology_context, dict)
                    and topology_context.get("map_id")
                    else None
                ),
            ),
        }
        prompt = (
            f"Original global task: {task.instruction}\n"
            f"Plan revision: {revision}\n"
            f"Bounded AgentOS context: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
            "Return one AgentOS Skill delegation. You are the Main Agent: arrange an ordered, revisable series "
            "of Skill subtasks needed to complete the global task, select the one that should execute now with "
            "delegated_subtask_index, and define a semantic success condition for every subtask. Choose only "
            "Skills in skill_catalog. MappingSkill is read-only and supplies current pose, semantic landmarks, map "
            "nodes, and connectivity. Delegate it when the global task may contain a map-addressable place or "
            "landmark but fresh mapping evidence is not yet present. After it returns, let the model compare the "
            "language with the returned semantic nodes; Runtime must not perform that semantic match. Its success "
            "condition must require "
            "only the requested pose/map/topology evidence, never robot motion, room arrival, or target visibility. "
            "Do not delegate MappingSkill again for the same observation when mapping_evidence says "
            "matches_current_observation=true; consume the existing evidence and continue the plan. "
            "After MappingSkill has acquired the fixed topology once, do not call it again merely because robot "
            "motion produced a new observation. A successful point_goal already returns authoritative arrival and "
            "localization evidence. Reacquire MappingSkill only when Verifier feedback explicitly reports missing, "
            "stale, or contradictory localization/topology evidence needed for the next decision. "
            "long_term_memory contains compact, source-grounded facts retrieved across earlier tasks. Use it as a "
            "hypothesis for selecting a likely place or Skill sequence, retain its source and confidence, and require "
            "fresh Skill/Verifier evidence before declaring the current task complete. Never treat remembered visual "
            "coordinates as valid in the current camera frame. "
            "Topo-Memory block nodes expose visited separately from goal_search. visited means the region was reached; "
            "goal_search is current-task, goal-conditioned evidence. no_target_observed means that goal was not visible "
            "in that attempt, not that it is permanently absent. Repeated no-target attempts without new target evidence "
            "should lower that node hypothesis relative to other semantically plausible nodes unless new evidence "
            "justifies revisiting it. Runtime execution may be exploring, but exploring is never a persisted memory fact. "
            "region_candidate_progress is the compact authoritative view of current-task evidence for the active "
            "candidate list. Static objects, landmarks, and task affordances are map priors only; never describe them "
            "as target_observed unless goal_search contains current-task target_observed or target_confirmed evidence. "
            "Before repeating an active candidate after no_target_observed or explicit Verifier rejection, compare the "
            "remaining candidates. Either advance the plan to a better candidate or state the fresh contradictory "
            "evidence that justifies another local attempt in region_search.update_reason. "
            "NavigationSkill exposes three task capabilities under one isolated Skill Runner. Topology nodes and route "
            "candidates are evidence: select a topology_node_id only when it is useful for that navigation "
            "subtask. Do not emit mandatory destination_node, next_node, or route_hypothesis fields, and do "
            "not assume Runtime will choose a semantic destination for you. Use point_goal for a known topology node, metric "
            "coordinate, or saved location that the deterministic navigation stack can reach. A point_goal success "
            "condition may establish only arrival/localization at that point; it cannot absorb an unresolved visual "
            "search, detection, or approach requirement. Use object_goal for continuous visual-semantic object "
            "search: successive affordance waypoints expose the requested object, then a grounded target is the "
            "required terminal endpoint. Do not emit topology_node_id on an object_goal: the Main Agent and Slow "
            "Brain may reason over Topo-Memory, but the ObjectGoal Skill contract remains semantic and coordinate-free. "
            "Use instruction_following for a natural-language route subtask whose ordered directional or landmark "
            "instructions should be executed by the OmniNav capability inside NavigationSkill. OmniNav local STOP "
            "ends only that Skill run and still requires independent verification. Do not emit poi_goal, "
            "person_following, or any other task type. "
            "Use destination_query and object_goal as an explicit task boundary: when the task names a separate "
            "map-addressable place and a visual target, keep the two values distinct. When there is no separate "
            "map-addressable destination, set destination_query exactly equal to object_goal. If the values are "
            "distinct and fresh mapping evidence is unavailable, emit the complete semantic task plan now: delegate "
            "MappingSkill first, retain a non-executable future point_goal placeholder for the named destination, and "
            "retain the terminal object_goal. The placeholder must use the semantic destination name as its target and "
            "keep topology_node_id null; never guess a backbone ID. Runtime executes only the delegated MappingSkill. "
            "After its SkillResult returns, replan from real map evidence and bind the pending point_goal to a concrete "
            "topology node before it can execute. This keeps the initial plan complete without treating an unresolved "
            "semantic destination as an executable coordinate. "
            "A compound instruction may "
            "use NavigationSkill more than once: when a known destination must be reached before a visual target can "
            "be searched, create an ordered point_goal subtask followed by an object_goal subtask. Do not create a "
            "point_goal prerequisite when no map-supported destination exists. Use object_goal when the unresolved "
            "work is finding a named physical object through visual-semantic search; use instruction_following when "
            "the unresolved work is primarily following an ordered natural-language route. Set "
            "task_specification.task_type and task_specification.goal to the currently delegated subtask's task_type "
            "and complete instruction. The Main Agent retains stable_global_task separately. "
            "You may select a "
            "topology node only when supported by Topo-Memory. Do not select a camera view, pixel, anchor "
            "action, controller action, or viewpoint turn. Those belong to the isolated Navigation Skill. "
            "For a task naming both an explicit destination place and an object, use a hierarchical evidence order. "
            "First select destination-compatible nodes from place_type; treat scene_summary, aliases, and function as "
            "secondary place evidence. Then use objects, landmarks, and task_affordances to rank candidates within "
            "that destination-compatible set. An object or alias at a node with a conflicting place_type must not "
            "override an explicit place_type match before the matching destination has been reached and searched. "
            "If goal_search records no_target_observed at the destination without new positive target evidence, the "
            "model may widen the hypothesis to connected or target-bearing nodes, while explaining that revision. "
            "If visual search in the current region fails and mapped semantics identify a better target-bearing node, "
            "revise the remaining Skill plan to a "
            "point_goal for that model-selected node followed by the unresolved object_goal. "
            "Decide semantically whether the unresolved object search needs node-guided regional exploration. "
            "When it does, populate region_search with one to four ordered candidate nodes and make "
            "active_topology_node_id the region selected for the current point_goal. Rank candidates jointly: "
            "place_type is the primary destination evidence, while object, landmark, function, and task relation "
            "evidence distinguish candidates within compatible places. Preserve and revise this task-scoped list "
            "across replans using progress_ledger and Topo-Memory goal_search. The initial candidate order is a stable "
            "search queue: do not reorder, drop, or replace its candidates on each observation. Update the active "
            "candidate and explain whether current-task evidence justifies continuing it or advancing through the "
            "remaining queue. no_target_observed should demote that "
            "hypothesis when alternatives remain; target_observed or target_confirmed should promote it; blocked "
            "means route execution failed and must not be treated as semantic absence. Set region_search=null when "
            "the Main Agent chooses direct visual object search without a mapped regional prerequisite, and for "
            "instruction_following plans. A point_goal placed before an unresolved object_goal is itself an explicit "
            "choice of node-guided regional search, so that plan must emit a non-null region_search. Runtime never "
            "creates, ranks, or switches these semantic hypotheses. "
            "Outside the map-acquisition planning turn, the full planner_output.subtasks list is the remaining "
            "Skill plan, and each item must name its skill_name; "
            "delegated_subtask_index must name exactly the subtask that should run next. After each delegated "
            "Skill, use progress_ledger, its compressed SkillResult, and verifier feedback to update or revise the "
            "remaining plan. Preserve completed progress instead of restarting the plan, and keep every local "
            "subtask explicitly connected to stable_global_task. When an object_goal remains unresolved after an "
            "intermediate affordance execution chunk, preserve that active subtask's instruction text and semantic "
            "objective exactly unless verifier evidence requires a real plan change; this keeps the isolated Slow "
            "Brain's previous decision and progress attached to the same Skill task. "
            "A blocked no_visible_pixel_goal is such replan evidence, not successful affordance progress. When it "
            "occurs, compare current-node semantics against mapped target-bearing nodes. If the model selects a "
            "better mapped region, materialize that decision as a point_goal with topology_node_id followed by the "
            "still-unresolved object_goal; mentioning the node only in rationale does not change robot state. "
            "Build this plan by reasoning backward from the "
            "terminal evidence that would satisfy stable_global_task. Infer the unresolved prerequisite world states "
            "and order them by dependency. Each subtask must represent one state transition that one Skill can "
            "execute and whose success condition can be independently verified. Emit the minimum complete plan, up "
            "to eight subtasks: do not split one atomic Skill call into narration, but do not copy the global task "
            "into one catch-all subtask when its success condition establishes only an intermediate or readiness "
            "state. If a subtask merely reaches, exposes, localizes, or prepares evidence needed for another part of "
            "the global task, include that unresolved follow-up in the same ordered plan. Before returning JSON, "
            "internally check that satisfying every listed success condition would satisfy the complete global task, "
            "and that the delegated index is the earliest subtask not already established by progress evidence. "
            "If a navigation subtask target is a topology node "
            "ID, put that same ID in topology_node_id; mentioning an ID only in target or rationale is not "
            "executable. "
            "approaching/reaching/interaction are descriptive plan stages, not "
            "a fixed mapping onto task types. When verifier feedback says a topology node was reached but "
            "semantic object work remains, preserve stable_global_task and delegate the planned object_goal rather "
            "than navigating to that same node again. Keep the rationale concise."
        )
        view_audit = await audit_observation_views(self.provider, current_observation)
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are the ABot-AgentOS Main Agent. Own global planning and Skill delegation only. "
                    "Never perform Skill-local visual grounding, control, self-verification, or memory writing. "
                    "Return strict JSON."
                ),
            ),
            ChatMessage(
                role="user",
                content=audited_observation_content(
                    prompt,
                    current_observation,
                    view_audit,
                ),
            ),
        ]
        delegation, response = await self._complete_delegation(
            messages,
            task.instruction,
            current_observation.trace.observation_id,
            planning_context=context,
        )
        self.session.main_agent_calls += 1
        return self._plan_from_delegation(
            task,
            current_observation,
            revision,
            delegation,
            verifier_feedback=verifier_feedback,
            planner_model=response.model,
        )

    def _advance_accepted_plan(self) -> AgentOSSkillDelegation | None:
        previous = self.session.last_delegation
        if previous is None:
            return None
        next_index = previous.delegated_subtask_index + 1
        if next_index >= len(previous.planner_output.subtasks):
            return None
        next_subtask = previous.planner_output.subtasks[next_index]
        return previous.model_copy(
            update={
                "skill_name": next_subtask.skill_name,
                "task_specification": previous.task_specification.model_copy(
                    update={
                        "task_type": next_subtask.task_type,
                        "goal": next_subtask.instruction,
                    }
                ),
                "delegated_subtask_index": next_index,
                "rationale_summary": (
                    "Advance the accepted Main Agent plan after the previous subtask passed "
                    "independent Skill verification."
                ),
            }
        )

    def _plan_from_delegation(
        self,
        task: TaskRequest,
        current_observation: Observation,
        revision: int,
        delegation: AgentOSSkillDelegation,
        *,
        verifier_feedback: str,
        planner_model: str,
    ) -> Plan:
        self.session.record_plan(delegation)
        subtask = delegation.delegated_subtask
        task_specification = delegation.task_specification
        execution_delegation = self._execution_delegation(delegation)
        inputs: dict[str, Any] = {
            "source_observation_id": current_observation.trace.observation_id,
            "verifier_feedback": verifier_feedback or None,
        }
        if subtask.skill_name == "NavigationSkill":
            inputs.update(
                {
                    "delegation": execution_delegation.model_dump(mode="json"),
                    "replace_active_plan": revision > 0,
                }
            )
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=revision,
            rationale_summary=delegation.rationale_summary,
            steps=[
                PlanStep(
                    step_id=f"delegation-{revision + 1:03d}",
                    skill_name=subtask.skill_name,
                    command=subtask.instruction,
                    success_conditions=[subtask.success_condition],
                    timeout_s=300.0,
                    max_retries=0,
                    inputs=inputs,
                )
            ],
            metadata={
                "agentos_role": "main_agent",
                "delegated_skill": delegation.skill_name,
                "delegated_subtask": subtask.model_dump(mode="json"),
                "delegated_subtask_index": delegation.delegated_subtask_index,
                "planned_skill_subtasks": [
                    item.model_dump(mode="json")
                    for item in delegation.planner_output.subtasks
                ],
                "n1_task_specification": task_specification.model_dump(mode="json"),
                "task_success_conditions": task.success_conditions,
                "planner_model": planner_model,
                "region_search": (
                    delegation.planner_output.region_search.model_dump(mode="json")
                    if delegation.planner_output.region_search is not None
                    else None
                ),
            },
        )

    def _retrieve_long_term_memory(
        self, instruction: str, *, map_id: str | None = None
    ) -> list[dict[str, Any]]:
        if self.memory is None:
            return []
        selected: list[tuple[Any, float]] = []
        seen: set[str] = set()

        def add(node, score: float) -> None:
            node_map = node.metadata.get("map_id")
            if node.node_id in seen or node.task_id is not None:
                return
            if map_id and node_map and str(node_map) != map_id:
                return
            selected.append((node, score))
            seen.add(node.node_id)

        entity_hits = self.memory.search(
            MemoryQuery(
                text=instruction,
                node_types=[MemoryNodeType.ENTITY],
                persistent_only=True,
                limit=3,
            )
        )
        for hit in entity_hits:
            add(hit.node, hit.score)
            for neighbor in self.memory.neighbors(hit.node.node_id, limit=4):
                if neighbor.node_type == MemoryNodeType.PLACE:
                    add(neighbor, hit.score * 0.9)
                    break
            if len(selected) >= 4:
                break
        for hit in self.memory.search(
            MemoryQuery(
                text=instruction,
                node_types=[MemoryNodeType.PLACE],
                persistent_only=True,
                limit=6,
            )
        ):
            add(hit.node, hit.score)
            if len(selected) >= 6:
                break

        return [
            {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "summary": (node.content or node.title)[:240],
                "source_ref": node.source_ref,
                "score": round(score, 3),
                "map_id": node.metadata.get("map_id"),
                "topology_node_id": node.metadata.get("topology_node_id"),
                "confidence": node.metadata.get("confidence"),
            }
            for node, score in selected[:6]
        ]

    async def _complete_delegation(
        self,
        messages: list[ChatMessage],
        global_instruction: str,
        current_observation_id: str,
        *,
        planning_context: dict[str, Any],
    ) -> tuple[AgentOSSkillDelegation, LLMResponse]:
        last_error = "unknown Main Agent protocol error"
        response_schema = AgentOSSkillDelegation.model_json_schema()
        planner_schema = response_schema["$defs"]["AgenticPlannerOutput"]
        planner_required = planner_schema.setdefault("required", [])
        if "region_search" not in planner_required:
            planner_required.append("region_search")
        for attempt in range(3):
            response = await self.provider.complete(
                messages,
                response_schema=response_schema,
            )
            if isinstance(response.parsed_json, dict):
                try:
                    payload = dict(response.parsed_json)
                    planner_output = payload.get("planner_output")
                    subtasks = (
                        planner_output.get("subtasks")
                        if isinstance(planner_output, dict)
                        else None
                    )
                    delegated_index = payload.get("delegated_subtask_index", 0)
                    if (
                        isinstance(subtasks, list)
                        and isinstance(delegated_index, int)
                        and 0 <= delegated_index < len(subtasks)
                        and isinstance(subtasks[delegated_index], dict)
                        and subtasks[delegated_index].get("skill_name")
                    ):
                        payload["skill_name"] = subtasks[delegated_index]["skill_name"]
                    delegation = AgentOSSkillDelegation.model_validate(payload)
                    planner_output = delegation.planner_output.model_copy(
                        update={"global_instruction": global_instruction}
                    )
                    delegated_subtask = delegation.delegated_subtask
                    task_specification = delegation.task_specification.model_copy(
                        update={
                            "task_type": delegated_subtask.task_type,
                            "goal": delegated_subtask.instruction,
                        }
                    )
                    delegation = delegation.model_copy(
                        update={
                            "planner_output": planner_output,
                            "task_specification": task_specification,
                        }
                    )
                    if self._requires_mapping_acquisition(
                        delegation,
                        planning_context=planning_context,
                    ):
                        delegation = self._mapping_acquisition_delegation(delegation)
                    delegation = await self._audit_region_selection(
                        delegation,
                        planning_context=planning_context,
                    )
                    self._validate_skills(delegation)
                    self._validate_topology_ids(delegation)
                    self._validate_plan_protocol(
                        delegation,
                        current_observation_id=current_observation_id,
                        planning_context=planning_context,
                    )
                    return delegation, response
                except (ValidationError, ValueError) as exc:
                    last_error = str(exc)
            else:
                last_error = "response did not contain one complete JSON object"
            if attempt < 2:
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "The previous candidate was rejected and is not a template. Recompute the plan "
                            "from the original task and current bounded evidence, then return only a corrected "
                            "AgentOSSkillDelegation whose skill_name matches the delegated subtask. Choose "
                            "only a Skill from skill_catalog. Do not add action, pixel, view, reflection, semantic_map, "
                            "or controller fields. Use only point_goal, object_goal, or instruction_following, and make the current "
                            "task_specification match the delegated subtask. "
                            "A topology node written as a navigation target must also be present in that "
                            "subtask's topology_node_id. MappingSkill is read-only and cannot use a "
                            "movement or visual-arrival success condition. Do not request localization twice "
                            "for the same observation. When mapping_evidence.available=false, never guess a "
                            "topology node ID. If MappingSkill is needed, this corrected planning turn must contain "
                            "a delegated MappingSkill first, followed by the complete semantic NavigationSkill plan. "
                            "A future point_goal must keep topology_node_id null and use the destination concept as a "
                            "non-executable placeholder. Runtime returns the MappingSkill result to this same Main "
                            "Agent, and the next planning turn must bind that placeholder from returned evidence. "
                            "Outside that acquisition turn, rebuild the minimum complete "
                            "Skill plan by reasoning backward from global completion evidence. Every subtask must be "
                            "one independently verifiable Skill state transition. "
                            "All other readiness-only steps require their unresolved follow-up in the same ordered plan. "
                            "For node-guided object search, provide region_search with ordered model-ranked candidates "
                            "and an active node matching the selected point_goal. Do not create region_search for direct "
                            "visual object search or instruction_following. Every response must explicitly include the "
                            "nullable region_search field. If point_goal precedes an unresolved object_goal, it must be "
                            "a non-null object rather than null. "
                            "When protocol or Verifier feedback says a mapped target-bearing region transition is missing, do "
                            "not merely discuss that node in rationale_summary. Emit the point_goal with its exact "
                            "topology_node_id as the earliest unresolved subtask and preserve the object_goal after "
                            "it. A no_visible_pixel_goal result is blocked execution and may justify this plan "
                            "change; it is not an intermediate affordance success that must be retried unchanged. "
                            f"Protocol feedback: {last_error}"
                        ),
                    )
                )
        raise ValueError(f"Main Agent delegation failed after 3 attempts: {last_error}")

    async def _audit_region_selection(
        self,
        delegation: AgentOSSkillDelegation,
        *,
        planning_context: dict[str, Any],
    ) -> AgentOSSkillDelegation:
        """Let a bounded semantic model audit Main Agent region ordering."""
        region_search = delegation.planner_output.region_search
        if self.session.region_search_state is not None:
            previous_region = RegionSearchPlan.model_validate(
                self.session.region_search_state
            )
            subtasks = delegation.planner_output.subtasks
            object_indexes = [
                index
                for index, subtask in enumerate(subtasks)
                if subtask.skill_name == "NavigationSkill"
                and subtask.task_type == "object_goal"
            ]
            if not object_indexes:
                return delegation
            regional_points = [
                subtask
                for index, subtask in enumerate(subtasks)
                if subtask.skill_name == "NavigationSkill"
                and subtask.task_type == "point_goal"
                and index < object_indexes[-1]
            ]
            active_node_id = (
                regional_points[0].topology_node_id
                if regional_points
                else (
                    region_search.active_topology_node_id
                    if region_search is not None
                    else previous_region.active_topology_node_id
                )
            )
            candidate_ids = {
                item.topology_node_id for item in previous_region.candidates
            }
            if active_node_id not in candidate_ids:
                raise ValueError(
                    "Main Agent selected a regional point outside the initial task-scoped "
                    "candidate queue. Finish or explicitly replace the regional search task before "
                    "introducing another node."
                )
            stable_region = previous_region.model_copy(
                update={
                    "active_topology_node_id": active_node_id,
                    "update_reason": (
                        region_search.update_reason
                        if region_search is not None
                        else (
                            "Preserved the initial task-scoped candidate queue while Main Agent "
                            f"selected {active_node_id}."
                        )
                    ),
                }
            )
            delegation = delegation.model_copy(
                update={
                    "planner_output": delegation.planner_output.model_copy(
                        update={"region_search": stable_region}
                    )
                }
            )
            return delegation
        if self.semantic_node_ranker is None:
            return delegation
        subtasks = delegation.planner_output.subtasks
        object_indexes = [
            index
            for index, subtask in enumerate(subtasks)
            if subtask.skill_name == "NavigationSkill"
            and subtask.task_type == "object_goal"
        ]
        regional_points = [
            subtask
            for index, subtask in enumerate(subtasks)
            if subtask.skill_name == "NavigationSkill"
            and subtask.task_type == "point_goal"
            and object_indexes
            and index < object_indexes[-1]
        ]
        if region_search is None and not regional_points:
            return delegation
        documents = self._semantic_node_documents(planning_context)
        if not documents:
            return delegation
        output = delegation.planner_output
        ranking = await self.semantic_node_ranker.rank_regions_async(
            SemanticTaskDecompositionInput(
                global_goal=output.global_instruction,
                destination_query=output.destination_query,
                object_goal=output.object_goal,
                node_local_radius_m=output.node_local_radius_m,
                anchor_max_depth_m=output.anchor_max_depth_m,
            ),
            documents,
        )
        selected = ranking.candidates[0]
        if selected.confidence < 0.70:
            return delegation
        planned_point_id = regional_points[0].topology_node_id if regional_points else None
        if region_search is None and planned_point_id == ranking.active_topology_node_id:
            planner_output = output.model_copy(update={"region_search": ranking})
            return delegation.model_copy(update={"planner_output": planner_output})
        ranked_first = (
            region_search.candidates[0].topology_node_id
            if region_search is not None
            else None
        )
        if (
            ranked_first != ranking.active_topology_node_id
            or region_search is None
            or region_search.active_topology_node_id != ranking.active_topology_node_id
        ):
            raise ValueError(
                "Independent semantic region audit selected "
                f"{ranking.active_topology_node_id} ({selected.confidence:.2f}): "
                f"{selected.relation_reason}. Re-rank region_search from this model evidence, "
                "make the audited node the first candidate and active point_goal, and retain "
                "other plausible alternatives. Audited candidates: "
                f"{ranking.model_dump(mode='json')}"
            )
        return delegation

    @staticmethod
    def _semantic_node_documents(
        planning_context: dict[str, Any],
    ) -> list[dict[str, object]]:
        topology_context = planning_context.get("topo_memory")
        if not isinstance(topology_context, dict):
            return []
        topo = topology_context.get("topo_memory")
        if not isinstance(topo, dict):
            topo = topology_context
        blocks = {
            str(item.get("topology_node_id")): item
            for item in topo.get("block_layer", [])
            if isinstance(item, dict) and item.get("topology_node_id")
        }
        functions = {
            str(item.get("topology_node_id")): item
            for item in topo.get("function_layer", [])
            if isinstance(item, dict) and item.get("topology_node_id")
        }
        objects = {
            str(item.get("topology_node_id")): item
            for item in topo.get("object_poi_layer", [])
            if isinstance(item, dict) and item.get("topology_node_id")
        }
        documents: list[dict[str, object]] = []
        for node_id, block in sorted(blocks.items()):
            function = functions.get(node_id, {})
            object_poi = objects.get(node_id, {})
            documents.append(
                {
                    "topology_node_id": node_id,
                    "node_semantics": {
                        "place_type": block.get("place_type"),
                        "scene_summary": block.get("scene_summary"),
                        "function": function.get("function"),
                        "task_affordances": function.get("task_affordances", []),
                        "objects": object_poi.get("objects", []),
                        "landmarks": object_poi.get("landmarks", []),
                    },
                    "visited": block.get("visited", False),
                    "goal_search": block.get("goal_search", []),
                }
            )
        return documents

    def _region_candidate_progress(
        self,
        topology_context: dict[str, Any] | None,
    ) -> list[dict[str, object]]:
        state = self.session.region_search_state
        if not isinstance(state, dict) or not isinstance(topology_context, dict):
            return []
        topo = topology_context.get("topo_memory")
        if not isinstance(topo, dict):
            topo = topology_context
        blocks = {
            str(item.get("topology_node_id")): item
            for item in topo.get("block_layer", [])
            if isinstance(item, dict) and item.get("topology_node_id")
        }
        progress: list[dict[str, object]] = []
        for rank, candidate in enumerate(state.get("candidates") or [], start=1):
            if not isinstance(candidate, dict):
                continue
            node_id = str(candidate.get("topology_node_id") or "")
            block = blocks.get(node_id, {})
            progress.append(
                {
                    "rank": rank,
                    "topology_node_id": node_id,
                    "active": node_id == state.get("active_topology_node_id"),
                    "prior_confidence": candidate.get("confidence"),
                    "visited": bool(block.get("visited")),
                    "visit_count": int(block.get("visit_count") or 0),
                    "goal_search": list(block.get("goal_search") or [])[-4:],
                }
            )
        return progress

    def _requires_mapping_acquisition(
        self,
        delegation: AgentOSSkillDelegation,
        *,
        planning_context: dict[str, Any],
    ) -> bool:
        mapping_evidence = planning_context.get("mapping_evidence") or {}
        if bool(mapping_evidence.get("available")):
            return False
        planner_output = delegation.planner_output
        return self._normalized_plan_concept(
            planner_output.destination_query
        ) != self._normalized_plan_concept(planner_output.object_goal)

    @staticmethod
    def _mapping_acquisition_delegation(
        candidate: AgentOSSkillDelegation,
    ) -> AgentOSSkillDelegation:
        """Expose the full semantic plan while delegating only map acquisition."""
        subtask = {
            "skill_name": "MappingSkill",
            "stage": "approaching",
            "task_type": "object_goal",
            "instruction": "Acquire current localization and semantic topology evidence.",
            "target": "current spatial evidence",
            "topology_node_id": None,
            "success_condition": (
                "Current pose, semantic landmarks, topology nodes, and connectivity are available."
            ),
        }
        payload = candidate.model_dump(mode="json")
        planner_output = payload["planner_output"]
        destination = str(planner_output.get("destination_query") or "destination")
        object_goal = str(planner_output.get("object_goal") or destination)
        followups: list[dict[str, Any]] = []
        for item in planner_output.get("subtasks") or []:
            if item.get("skill_name") == "MappingSkill":
                continue
            copied = dict(item)
            if copied.get("task_type") == "point_goal":
                copied.update(
                    {
                        "target": destination,
                        "topology_node_id": None,
                        "instruction": (
                            f"Resolve the mapped region for {destination} and navigate to it."
                        ),
                        "success_condition": (
                            f"The robot reaches the map-grounded region for {destination}."
                        ),
                    }
                )
            elif copied.get("task_type") == "object_goal":
                copied.update(
                    {
                        "target": object_goal,
                        "topology_node_id": None,
                        "instruction": f"Find and approach {object_goal} in {destination}.",
                        "success_condition": (
                            f"The robot visually grounds and approaches {object_goal}."
                        ),
                    }
                )
            followups.append(copied)

        if not any(item.get("task_type") == "point_goal" for item in followups):
            followups.insert(
                0,
                {
                    "skill_name": "NavigationSkill",
                    "stage": "approaching",
                    "task_type": "point_goal",
                    "instruction": (
                        f"Resolve the mapped region for {destination} and navigate to it."
                    ),
                    "target": destination,
                    "topology_node_id": None,
                    "success_condition": (
                        f"The robot reaches the map-grounded region for {destination}."
                    ),
                },
            )
        if not any(item.get("task_type") == "object_goal" for item in followups):
            followups.append(
                {
                    "skill_name": "NavigationSkill",
                    "stage": "interaction",
                    "task_type": "object_goal",
                    "instruction": f"Find and approach {object_goal} in {destination}.",
                    "target": object_goal,
                    "topology_node_id": None,
                    "success_condition": (
                        f"The robot visually grounds and approaches {object_goal}."
                    ),
                }
            )
        payload.update(
            {
                "skill_name": "MappingSkill",
                "task_specification": {
                    "task_type": "object_goal",
                    "goal": subtask["instruction"],
                },
                "delegated_subtask_index": 0,
                "rationale_summary": (
                    "Acquire source-grounded spatial evidence before selecting an executable "
                    "navigation destination."
                ),
            }
        )
        payload["planner_output"].update(
            {
                "node_selection": None,
                "region_search": None,
                "subtasks": [subtask, *followups],
            }
        )
        return AgentOSSkillDelegation.model_validate(payload)

    def _validate_skills(self, delegation: AgentOSSkillDelegation) -> None:
        proposed = {item.skill_name for item in delegation.planner_output.subtasks}
        unknown = sorted(proposed - self.skill_names)
        if unknown:
            raise ValueError(f"Main Agent selected unavailable Skills: {unknown}")

    def _validate_plan_protocol(
        self,
        delegation: AgentOSSkillDelegation,
        *,
        current_observation_id: str,
        planning_context: dict[str, Any],
    ) -> None:
        subtasks = delegation.planner_output.subtasks
        self._validate_completed_subtask_advance(delegation)
        if (
            delegation.delegated_subtask.skill_name == "MappingSkill"
            and self.session.mapping_observation_id == current_observation_id
        ):
            raise ValueError(
                "MappingSkill already returned evidence for the current observation"
            )
        for subtask in subtasks:
            target = subtask.target.strip()
            if subtask.task_type == "object_goal" and subtask.topology_node_id is not None:
                raise ValueError(
                    "object_goal must keep topology_node_id null; use a separate point_goal "
                    "for executable node navigation and provide Topo-Memory as reasoning context"
                )
            if (
                subtask.task_type == "instruction_following"
                and subtask.topology_node_id is not None
            ):
                raise ValueError(
                    "instruction_following is a semantic route contract and must keep topology_node_id null"
                )
            if (
                subtask.skill_name == "NavigationSkill"
                and re.fullmatch(r"backbone-[0-9]{3,}", target, re.IGNORECASE)
                and subtask.topology_node_id != target.lower()
            ):
                raise ValueError(
                    f"Navigation target {target} must also populate topology_node_id"
                )

        region_search = delegation.planner_output.region_search
        object_subtask_indexes = [
            index
            for index, subtask in enumerate(subtasks)
            if subtask.skill_name == "NavigationSkill"
            and subtask.task_type == "object_goal"
        ]
        if region_search is not None:
            if not object_subtask_indexes:
                raise ValueError(
                    "region_search is only valid while an object_goal remains unresolved"
                )
            selected = delegation.planner_output.node_selection
            if (
                selected is not None
                and selected.selected_topology_node_id
                != region_search.active_topology_node_id
            ):
                raise ValueError(
                    "node_selection and region_search must identify the same active node"
                )
        regional_point_indexes: list[int] = []
        if object_subtask_indexes:
            final_object_index = object_subtask_indexes[-1]
            regional_point_indexes = [
                index
                for index, subtask in enumerate(subtasks)
                if subtask.skill_name == "NavigationSkill"
                and subtask.task_type == "point_goal"
                and subtask.topology_node_id is not None
                and index < final_object_index
            ]
        if regional_point_indexes and region_search is None:
            raise ValueError(
                "A node-guided object search must include region_search with ordered "
                "candidate hypotheses and the active point_goal node"
            )
        if (
            self.session.region_search_state is not None
            and object_subtask_indexes
            and region_search is None
        ):
            raise ValueError(
                "An active region_search cannot be discarded while its object_goal remains "
                "unresolved. Preserve or revise the candidate list from verifier and goal_search evidence."
            )
        if region_search is not None:
            candidate_ids = {
                item.topology_node_id for item in region_search.candidates
            }
            point_node_ids = {
                subtasks[index].topology_node_id for index in regional_point_indexes
            }
            if None in point_node_ids or not point_node_ids.issubset(candidate_ids):
                raise ValueError(
                    "Every regional point_goal must reference a region_search candidate"
                )
            if (
                regional_point_indexes
                and subtasks[regional_point_indexes[0]].topology_node_id
                != region_search.active_topology_node_id
            ):
                raise ValueError(
                    "The earliest unresolved regional point_goal must match "
                    "region_search.active_topology_node_id"
                )

        destination = self._normalized_plan_concept(
            delegation.planner_output.destination_query
        )
        object_goal = self._normalized_plan_concept(
            delegation.planner_output.object_goal
        )
        if destination != object_goal:
            mapping_evidence = planning_context.get("mapping_evidence") or {}
            mapping_available = bool(mapping_evidence.get("available"))
            mapping_indexes = [
                index
                for index, subtask in enumerate(subtasks)
                if subtask.skill_name == "MappingSkill"
            ]
            if not mapping_available and (
                not mapping_indexes
                or delegation.delegated_subtask_index != mapping_indexes[0]
            ):
                raise ValueError(
                    "A compound destination/object task requires MappingSkill as the "
                    "delegated first subtask while fresh mapping evidence is unavailable. "
                    "Do not guess a topology node before that SkillResult returns."
                )
            if not mapping_available:
                if any(
                    subtask.topology_node_id is not None
                    for subtask in subtasks
                ):
                    raise ValueError(
                        "Main Agent guessed an executable topology node before MappingSkill "
                        "returned topology evidence. Keep future point_goal placeholders "
                        "semantic with topology_node_id=null, then bind them after MappingSkill."
                    )
                return
            object_indexes = [
                index
                for index, subtask in enumerate(subtasks)
                if subtask.skill_name == "NavigationSkill"
                and subtask.task_type == "object_goal"
                and self._normalized_plan_concept(subtask.target) == object_goal
            ]
            if not object_indexes:
                raise ValueError(
                    "Main Agent merged or dropped the terminal object_goal. A compound "
                    "destination/object task must preserve one NavigationSkill object_goal "
                    "whose target exactly matches planner_output.object_goal."
                )
            object_index = object_indexes[-1]
            if delegation.planner_output.node_selection is not None:
                selected_node_id = (
                    delegation.planner_output.node_selection.selected_topology_node_id
                )
                selected_point_indexes = [
                    index
                    for index, subtask in enumerate(subtasks)
                    if subtask.skill_name == "NavigationSkill"
                    and subtask.task_type == "point_goal"
                    and subtask.topology_node_id == selected_node_id
                ]
                if not selected_point_indexes or selected_point_indexes[0] >= object_index:
                    raise ValueError(
                        "Main Agent selected a mapped destination but did not materialize "
                        "the coarse-to-fine Skill plan. Emit a point_goal for the selected "
                        "topology node before the unresolved object_goal."
                    )

    def _validate_completed_subtask_advance(
        self,
        delegation: AgentOSSkillDelegation,
    ) -> None:
        if not self.session.progress_ledger:
            return
        latest = self.session.progress_ledger[-1]
        previous = latest.get("delegated_subtask")
        if not latest.get("verified_success") or not isinstance(previous, dict):
            return
        current = delegation.delegated_subtask
        same_subtask = (
            previous.get("skill_name") == current.skill_name
            and previous.get("task_type") == current.task_type
            and previous.get("topology_node_id") == current.topology_node_id
            and self._normalized_plan_concept(str(previous.get("target") or ""))
            == self._normalized_plan_concept(current.target)
        )
        if same_subtask:
            raise ValueError(
                "The previously delegated subtask already passed Skill Verifier. Advance to "
                "the earliest unresolved planned subtask instead of executing it again."
            )
    @staticmethod
    def _normalized_plan_concept(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _execution_delegation(
        delegation: AgentOSSkillDelegation,
    ) -> AgentOSSkillDelegation:
        """Scope a Skill Runner to its delegated subtask, not the full Main Agent plan."""
        subtask = delegation.delegated_subtask
        node_selection = delegation.planner_output.node_selection
        if (
            node_selection is not None
            and node_selection.selected_topology_node_id != subtask.topology_node_id
        ):
            node_selection = None
        region_search = delegation.planner_output.region_search
        if subtask.task_type == "instruction_following":
            region_search = None
        planner_output = delegation.planner_output.model_copy(
            update={
                "subtasks": [subtask],
                "node_selection": node_selection,
                "region_search": region_search,
            }
        )
        task_specification = delegation.task_specification.model_copy(
            update={
                "task_type": subtask.task_type,
                "goal": subtask.instruction,
            }
        )
        return delegation.model_copy(
            update={
                "skill_name": subtask.skill_name,
                "task_specification": task_specification,
                "planner_output": planner_output,
                "delegated_subtask_index": 0,
            }
        )

    def _validate_topology_ids(self, delegation: AgentOSSkillDelegation) -> None:
        topo = ((self.session.context_snapshot or {}).get("slow_context") or {}).get("topology_context") or {}
        memory = topo.get("topo_memory") or {}
        known_ids = {
            str(item.get("topology_node_id"))
            for layer in ("block_layer", "road_layer", "function_layer", "object_poi_layer")
            for item in (memory.get(layer) or [])
            if isinstance(item, dict) and item.get("topology_node_id")
        }
        if not known_ids:
            return
        selected = delegation.planner_output.node_selection
        proposed = [
            subtask.topology_node_id
            for subtask in delegation.planner_output.subtasks
            if subtask.topology_node_id is not None
        ]
        if selected is not None:
            proposed.append(selected.selected_topology_node_id)
        region_search = delegation.planner_output.region_search
        if region_search is not None:
            proposed.extend(
                item.topology_node_id for item in region_search.candidates
            )
        unknown = sorted({node_id for node_id in proposed if node_id not in known_ids})
        if unknown:
            raise ValueError(f"Main Agent invented unknown topology nodes: {unknown}")


def compress_navigation_handoff(packet: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded state shared across AgentOS component boundaries."""
    result = packet.get("result") or {}
    frames = result.get("trajectory_keyframes") or packet.get("trajectory_keyframes") or []
    refs: list[str] = []
    selected_keyframes: list[dict[str, Any]] = []
    for frame in frames[-4:]:
        ref = (frame.get("rgb_refs") or {}).get("front")
        if ref and ref not in refs:
            refs.append(str(ref))
            selected_keyframes.append(
                {
                    "observation_id": frame.get("observation_id"),
                    "front_image_ref": str(ref),
                }
            )
    return {
        "ok": bool(packet.get("ok", False)),
        "status": packet.get("status"),
        "observation": packet.get("post_observation") or packet.get("observation"),
        "slow_context": packet.get("slow_context"),
        "semantic_route_plan": packet.get("semantic_route_plan"),
        "execution_summary": {
            key: result.get(key)
            for key in (
                "controller",
                "reached",
                "control_steps",
                "distance_m",
                "remaining_distance_m",
                "collisions",
                "stalled",
                "termination_reason",
            )
        },
        "active_target_evidence": packet.get("active_target_evidence"),
        "topology_goal_search": packet.get("topology_goal_search"),
        "semantic_completion": packet.get("semantic_completion"),
        "selected_keyframe_refs": refs,
        "selected_keyframes": selected_keyframes,
    }
