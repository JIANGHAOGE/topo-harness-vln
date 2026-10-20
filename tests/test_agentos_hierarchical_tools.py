from __future__ import annotations

import asyncio

from agentos.core.events import InMemoryEventBus
from agentos.planning.hierarchical_agentos import (
    HierarchicalAgentSession,
    HierarchicalMacroPlanner,
    HierarchicalNavigationSkill,
)
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.skill import SkillRequest
from agentos.schemas.task import TaskRequest
from agentos.skills.base import CancellationToken, SkillContext


def _view_audit_response(messages, response_schema):
    if (response_schema or {}).get("title") != "SlowBrainSingleViewAudit":
        return None
    system = str(messages[0].content)
    view = next(
        item for item in ("left", "front", "right") if f"robot-{item.upper()}" in system
    )
    return LLMResponse(
        model="fake-qwen",
        content="{}",
        raw={},
        parsed_json={
            "view": view,
            "visible_objects": [f"{view} object"],
            "navigable_semantic_anchors": [f"{view} doorway"],
            "target_relevant_evidence": [],
            "scene_summary": f"Current {view} camera scene.",
        },
    )


def _decision(observation_id: str) -> dict:
    return {
        "observation_id": observation_id,
        "global_goal": "go to office to find a white fridge",
        "navigation_task_type": "point_goal",
        "cot": None,
        "self_reflection": None,
        "planner_output": {
            "global_instruction": "go to office to find a white fridge",
            "destination_query": "office",
            "object_goal": "white fridge",
            "node_selection": {
                "selected_topology_node_id": "backbone-002",
                "confidence": 0.95,
                "matched_concepts": ["office", "white fridge"],
                "selection_reason": "The node jointly matches the requested place and object.",
            },
            "subtasks": [
                {
                    "stage": "approaching",
                    "task_type": "point_goal",
                    "instruction": "Navigate to the office region.",
                    "target": "office region",
                    "topology_node_id": "backbone-002",
                    "success_condition": "The robot arrives at backbone-002.",
                },
                {
                    "stage": "reaching",
                    "task_type": "object_goal",
                    "instruction": "Find the white fridge in the office.",
                    "target": "white fridge",
                    "topology_node_id": "backbone-002",
                    "success_condition": "The white fridge is clearly visible.",
                },
            ],
        },
        "previous_assessment": {
            "execution_status": "not_started",
            "task_effect": "unknown",
            "evidence": "Initial synchronized observation.",
        },
        "route_hypothesis": {
            "statement": "Use the connected topology route to backbone-002.",
            "confidence": 0.9,
        },
        "candidates": [
            {
                "action_type": "known_location",
                "target": "backbone-002",
                "task_relevance": 0.95,
                "information_gain": 0.8,
                "revisit_penalty": 0.0,
            }
        ],
        "action": {
            "action_type": "known_location",
            "target": "backbone-002",
            "topology_node_id": "backbone-002",
            "expected_evidence": "Office-region landmarks and the white fridge.",
            "success_condition": "The robot arrives at backbone-002.",
            "fallback": "Adjust viewpoint at the node.",
        },
        "semantic_map": {
            "scene_summary": "Initial workspace",
            "left": ["cabinets"],
            "front": ["corridor"],
            "right": ["robot"],
        },
    }


class _Provider:
    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=_decision("model-invented-id"),
        )

    async def close(self):
        return None


class _RepairingProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        self.calls += 1
        decision = _decision("model-invented-id")
        if self.calls == 1:
            decision["action"].pop("topology_node_id")
            decision["action"]["action_type"] = "visible_anchor"
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _ViewpointWithNodeProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        self.calls += 1
        decision = _decision("model-invented-id")
        decision["candidates"] = [
            {
                "action_type": "viewpoint_adjustment",
                "target": "face route toward backbone-004",
                "task_relevance": 0.9,
                "information_gain": 0.8,
                "revisit_penalty": 0.0,
            }
        ]
        decision["action"] = {
            "action_type": "viewpoint_adjustment",
            "target": "face route toward backbone-004",
            "view": "front",
            "topology_node_id": "backbone-004",
            "viewpoint_action": "turn_left",
            "expected_evidence": "The connected route enters the camera coverage.",
            "success_condition": "A fresh observation covers the route direction.",
            "fallback": "Replan from the fresh observation.",
        }
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _KnownLocationWithTargetRoleProvider(_Provider):
    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        decision = _decision("model-invented-id")
        decision["action"]["anchor_role"] = "target"
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _RegionalRepairProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        self.calls += 1
        decision = _decision("model-invented-id")
        if self.calls == 1:
            decision["candidates"] = [
                {
                    "action_type": "visible_anchor",
                    "target": "open doorway ahead",
                    "task_relevance": 0.8,
                    "information_gain": 0.7,
                    "revisit_penalty": 0.0,
                }
            ]
            decision["action"] = {
                "action_type": "visible_anchor",
                "target": "open doorway ahead",
                "view": "front",
                "anchor_role": "affordance",
                "expected_evidence": "The doorway leads toward the target region.",
                "success_condition": "The robot reaches the doorway.",
                "fallback": "Use the connected topology route.",
            }
        else:
            decision["candidates"][0]["target"] = "backbone-004"
            decision["action"]["target"] = "backbone-004"
            decision["action"]["topology_node_id"] = "backbone-004"
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _ReflectionRepairProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        self.calls += 1
        decision = _decision("model-invented-id")
        decision["planner_output"] = None
        if self.calls > 1:
            decision["self_reflection"] = {
                "subtask_id": "subtask-001",
                "disposition": "continue_active_subtask",
                "feedback": "The intermediate route node was reached; continue toward the target node.",
            }
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _LocalSearchRepairProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        self.calls += 1
        decision = _decision("model-invented-id")
        decision["planner_output"] = None
        decision["self_reflection"] = {
            "subtask_id": "subtask-002",
            "disposition": (
                "replan" if self.calls == 1 else "continue_active_subtask"
            ),
            "feedback": "The bed is not visible yet; continue local search.",
        }
        if self.calls > 1:
            decision["candidates"] = [
                {
                    "action_type": "viewpoint_adjustment",
                    "target": "search the bedroom",
                    "task_relevance": 0.9,
                    "information_gain": 0.9,
                    "revisit_penalty": 0.0,
                }
            ]
            decision["action"] = {
                "action_type": "viewpoint_adjustment",
                "target": "search the bedroom",
                "viewpoint_action": "turn_left",
                "expected_evidence": "A new sector of the bedroom becomes visible.",
                "success_condition": "A fresh view covers an unseen sector.",
                "fallback": "Try another local affordance.",
            }
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _ViewpointLoopRepairProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        audit = _view_audit_response(messages, response_schema)
        if audit is not None:
            return audit
        self.calls += 1
        decision = _decision("model-invented-id")
        decision["planner_output"] = None
        decision["self_reflection"] = {
            "subtask_id": "subtask-002",
            "disposition": "continue_active_subtask",
            "feedback": "The target is not visible; continue local exploration.",
        }
        if self.calls == 1:
            decision["candidates"] = [
                {
                    "action_type": "viewpoint_adjustment",
                    "target": "sample another sector",
                    "task_relevance": 0.8,
                    "information_gain": 0.8,
                    "revisit_penalty": 0.0,
                }
            ]
            decision["action"] = {
                "action_type": "viewpoint_adjustment",
                "target": "sample another sector",
                "viewpoint_action": "turn_left",
                "expected_evidence": "Another sector becomes visible.",
                "success_condition": "A fresh sector is observed.",
                "fallback": "Move through a visible traversable anchor.",
            }
        else:
            decision["candidates"] = [
                {
                    "action_type": "visible_anchor",
                    "target": "open doorway ahead",
                    "task_relevance": 0.9,
                    "information_gain": 0.9,
                    "revisit_penalty": 0.0,
                }
            ]
            decision["action"] = {
                "action_type": "visible_anchor",
                "target": "open doorway ahead",
                "view": "front",
                "anchor_role": "affordance",
                "expected_evidence": "The robot enters the connected room.",
                "success_condition": "A fresh observation is captured beyond the doorway.",
                "fallback": "Reassess the local views.",
            }
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


class _Service:
    def __init__(self, observation: Observation) -> None:
        self.observation = observation
        self.calls: list[dict] = []

    def begin_navigation(self, payload=None):
        return {
            "ok": True,
            "observation": self.observation.model_dump(mode="json"),
            "slow_context": {"task_specification": (payload or {}).get("global_goal")},
            "next_call_contract": {
                "next_tool": "advance_navigation",
                "observation_id": self.observation.trace.observation_id,
                "planner_output": "required",
                "self_reflection": "null",
            },
        }

    def advance_navigation(self, payload):
        self.calls.append(payload)
        return {
            "ok": True,
            "status": "completed",
            "decision": payload["action"],
            "result": {
                "controller": "habitat_greedy_follower",
                "reached": True,
                "control_steps": 8,
                "distance_m": 1.6,
                "trajectory_keyframes": [
                    {
                        "rgb_refs": {"front": "mock://private-keyframe.jpg"},
                        "pose": {"x": 1.0, "y": 2.0},
                        "yaw": 0.5,
                        "location_id": "location-004",
                        "controller_ticks": [1, 2, 3],
                    }
                ],
            },
            "next_observation_id": self.observation.trace.observation_id,
            "post_observation": self.observation.model_dump(mode="json"),
            "next_call_contract": {"next_tool": "advance_navigation"},
        }


class _FinishWaitService(_Service):
    def advance_navigation(self, payload):
        self.calls.append(payload)
        return {
            "ok": True,
            "status": "awaiting_finish_verification",
            "next_observation_id": self.observation.trace.observation_id,
            "post_observation": self.observation.model_dump(mode="json"),
            "next_call_contract": {"next_tool": "finish_verification"},
        }


class _IntermediateRouteService(_Service):
    def advance_navigation(self, payload):
        packet = super().advance_navigation(payload)
        actual = dict(payload["action"])
        actual["target"] = "backbone-004 connected route waypoint"
        actual["topology_node_id"] = "backbone-004"
        packet["decision"] = actual
        return packet


def _observation(tmp_path, trace) -> Observation:
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"{view}.jpg"
        path.write_bytes(b"test-image")
        refs[view] = str(path)
    return Observation(
        trace=trace,
        rgb_refs=refs,
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0, source="test", frame="map"),
    )


def test_main_agent_emits_one_hierarchical_macro_skill(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    session = HierarchicalAgentSession(
        global_goal="go to office to find a white fridge",
        handoff=HierarchicalMacroPlanner.compress_handoff(
            service.begin_navigation({"global_goal": "go to office to find a white fridge"})
        ),
    )
    planner = HierarchicalMacroPlanner(_Provider(), service, session)
    task = TaskRequest(trace=trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert len(plan.steps) == 1
    assert plan.steps[0].skill_name == "HierarchicalNavigationSkill"
    assert plan.steps[0].inputs["observation_id"] == "obs-current"
    assert plan.metadata["macro_action"]["topology_node_id"] == "backbone-002"


def test_main_agent_asks_model_to_repair_invalid_protocol_once(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    provider = _RepairingProvider()
    planner = HierarchicalMacroPlanner(
        provider,
        service,
        HierarchicalAgentSession(global_goal="find a white fridge"),
    )
    task = TaskRequest(trace=trace, instruction="find a white fridge")

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.calls == 2
    assert plan.metadata["macro_action"]["action_type"] == "known_location"


def test_main_agent_drops_navigation_ids_from_viewpoint_action(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    provider = _ViewpointWithNodeProvider()
    planner = HierarchicalMacroPlanner(
        provider,
        service,
        HierarchicalAgentSession(global_goal="go to bedroom to find a bed"),
    )
    task = TaskRequest(trace=trace, instruction="go to bedroom to find a bed")

    plan = asyncio.run(planner.plan(task, observation))

    action = plan.metadata["macro_action"]
    assert provider.calls == 1
    assert action["action_type"] == "viewpoint_adjustment"
    assert action["viewpoint_action"] == "turn_left"
    assert action["view"] is None
    assert action["topology_node_id"] is None
    assert action["location_id"] is None


def test_main_agent_drops_visual_role_from_known_location(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    planner = HierarchicalMacroPlanner(
        _KnownLocationWithTargetRoleProvider(),
        _Service(observation),
        HierarchicalAgentSession(global_goal="find a white fridge"),
    )
    task = TaskRequest(trace=trace, instruction="find a white fridge")

    plan = asyncio.run(planner.plan(task, observation))

    assert plan.metadata["macro_action"]["action_type"] == "known_location"
    assert plan.metadata["macro_action"]["anchor_role"] == "affordance"


def test_main_agent_repairs_affordance_action_to_required_route_node(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    provider = _RegionalRepairProvider()
    session = HierarchicalAgentSession(
        global_goal="go to bedroom to find a bed",
        handoff={
            "observation": observation.model_dump(mode="json"),
            "semantic_route_plan": {
                "regional_guidance": {
                    "mode": "route_to_region",
                    "next_topology_node_id": "backbone-004",
                }
            },
        },
    )
    planner = HierarchicalMacroPlanner(provider, service, session)
    task = TaskRequest(trace=trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.calls == 2
    assert plan.metadata["macro_action"]["action_type"] == "known_location"
    assert plan.metadata["macro_action"]["topology_node_id"] == "backbone-004"


def test_replan_requires_model_generated_self_reflection(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    session = HierarchicalAgentSession(
        global_goal="go to bedroom to find a bed",
        macro_index=1,
        handoff={
            "observation": observation.model_dump(mode="json"),
            "next_call_contract": {
                "self_reflection": "required",
                "self_reflection_subtask_id": "subtask-001",
            },
        },
    )
    provider = _ReflectionRepairProvider()
    planner = HierarchicalMacroPlanner(provider, service, session)
    task = TaskRequest(trace=trace, instruction=session.global_goal)

    plan = asyncio.run(
        planner.replan(
            task,
            observation,
            None,  # type: ignore[arg-type]
            [],
            reason="Finish Verifier found no bed in the latest observation.",
            next_revision=2,
        )
    )

    reflection = plan.steps[0].inputs["self_reflection"]
    assert provider.calls == 2
    assert reflection["subtask_id"] == "subtask-001"
    assert reflection["should_replan"] is False
    assert "intermediate route node" in reflection["feedback"]


def test_target_region_requires_model_to_continue_local_search(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    provider = _LocalSearchRepairProvider()
    session = HierarchicalAgentSession(
        global_goal="go to bedroom to find a bed",
        macro_index=2,
        handoff={
            "observation": observation.model_dump(mode="json"),
            "next_call_contract": {
                "self_reflection": "required",
                "self_reflection_subtask_id": "subtask-002",
            },
            "semantic_route_plan": {
                "regional_guidance": {
                    "mode": "search_within_region",
                    "current_topology_node_id": "backbone-003",
                    "target_topology_node_id": "backbone-003",
                    "next_topology_node_id": "backbone-003",
                }
            },
            "execution_summary": {"stalled": False, "termination_reason": "reached"},
        },
    )
    planner = HierarchicalMacroPlanner(provider, service, session)
    task = TaskRequest(trace=trace, instruction=session.global_goal)

    plan = asyncio.run(
        planner.replan(
            task,
            observation,
            None,  # type: ignore[arg-type]
            [],
            reason="The bed is not visible in the current views.",
            next_revision=3,
        )
    )

    assert provider.calls == 2
    assert plan.metadata["macro_action"]["action_type"] == "viewpoint_adjustment"
    reflection = plan.steps[0].inputs["self_reflection"]
    assert reflection["outcome"] == "partial"
    assert reflection["should_replan"] is False


def test_local_search_forces_model_to_leave_repeated_viewpoint_loop(tmp_path, make_trace):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    provider = _ViewpointLoopRepairProvider()
    session = HierarchicalAgentSession(
        global_goal="go to bedroom to find a bed",
        macro_index=4,
        recent_macro_decisions=[
            {"action_type": "viewpoint_adjustment", "viewpoint_action": "turn_left"},
            {"action_type": "viewpoint_adjustment", "viewpoint_action": "turn_right"},
        ],
        handoff={
            "observation": observation.model_dump(mode="json"),
            "next_call_contract": {
                "self_reflection": "required",
                "self_reflection_subtask_id": "subtask-002",
            },
            "semantic_route_plan": {
                "regional_guidance": {
                    "mode": "search_within_region",
                    "current_topology_node_id": "backbone-003",
                    "target_topology_node_id": "backbone-003",
                }
            },
            "execution_summary": {"stalled": False, "termination_reason": "reached"},
        },
    )
    planner = HierarchicalMacroPlanner(provider, service, session)
    task = TaskRequest(trace=trace, instruction=session.global_goal)

    plan = asyncio.run(
        planner.replan(
            task,
            observation,
            None,  # type: ignore[arg-type]
            [],
            reason="The bed is not visible after another viewpoint adjustment.",
            next_revision=5,
        )
    )

    assert provider.calls == 2
    assert plan.metadata["macro_action"]["action_type"] == "visible_anchor"
    assert plan.metadata["macro_action"]["target"] == "open doorway ahead"


def test_hierarchical_skill_executes_previous_tool_pipeline_as_one_boundary(
    tmp_path, make_trace
):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _Service(observation)
    session = HierarchicalAgentSession(global_goal="find white fridge")
    skill = HierarchicalNavigationSkill(service, session)
    request = SkillRequest(
        trace=trace,
        skill_name=skill.name,
        command="Navigate to backbone-002.",
        inputs=_decision("obs-current"),
        timeout_s=30.0,
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(
                event_bus=InMemoryEventBus(),
                cancellation=CancellationToken(),
            ),
        )
    )

    assert result.status.value == "success"
    assert "NODE_NAV" in result.compressed_history
    assert result.outputs["result"]["control_steps"] == 8
    assert service.calls[0]["action"]["topology_node_id"] == "backbone-002"
    assert session.handoff is not None
    assert session.handoff["execution_summary"]["control_steps"] == 8
    assert "result" not in session.handoff
    assert session.handoff["selected_keyframe_refs"] == ["mock://private-keyframe.jpg"]
    assert result.outputs["result"]["control_steps"] == 8
    assert result.outputs["result"]["trajectory_keyframes"][0]["controller_ticks"] == [1, 2, 3]
    assert result.evidence[-1].kind == "trajectory_keyframe"


def test_hierarchical_skill_exposes_finish_boundary_without_fake_motion(
    tmp_path, make_trace
):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _FinishWaitService(observation)
    session = HierarchicalAgentSession(global_goal="find a bed")
    skill = HierarchicalNavigationSkill(service, session)
    request = SkillRequest(
        trace=trace,
        skill_name=skill.name,
        command="Verify completion.",
        inputs=_decision("obs-current"),
        timeout_s=30.0,
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(
                event_bus=InMemoryEventBus(),
                cancellation=CancellationToken(),
            ),
        )
    )

    assert result.status.value == "success"
    assert result.compressed_history.startswith("FINISH_CHECK_REQUIRED")
    assert "VIEWPOINT_ADJUST" not in result.compressed_history


def test_hierarchical_skill_reports_requested_and_executed_route_nodes(
    tmp_path, make_trace
):
    trace = make_trace(observation_id="obs-current")
    observation = _observation(tmp_path, trace)
    service = _IntermediateRouteService(observation)
    skill = HierarchicalNavigationSkill(
        service,
        HierarchicalAgentSession(global_goal="find a bed"),
    )
    inputs = _decision("obs-current")
    inputs["action"]["target"] = "backbone-003"
    inputs["action"]["topology_node_id"] = "backbone-003"
    inputs["candidates"][0]["target"] = "backbone-003"
    request = SkillRequest(
        trace=trace,
        skill_name=skill.name,
        command="Navigate to backbone-003.",
        inputs=inputs,
        timeout_s=30.0,
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(
                event_bus=InMemoryEventBus(),
                cancellation=CancellationToken(),
            ),
        )
    )

    assert "INTERMEDIATE_NODE_NAV" in result.compressed_history
    assert "requested=backbone-003" in result.compressed_history
    assert "executed=backbone-004" in result.compressed_history
    assert "requested_reached=False" in result.compressed_history
