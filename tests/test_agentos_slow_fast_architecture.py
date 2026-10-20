from __future__ import annotations

import asyncio
import time

import pytest
from PIL import Image

from agentos.core.events import InMemoryEventBus
from agentos.harness.navigation_session import (
    AgentOSMainAgentSession,
    N1NavigationSkillSession,
)
from agentos.planning.agentos_hierarchical import AgentOSHierarchicalPlanner
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation
from agentos.schemas.perception import (
    VisualDetection,
    VisualGroundingBatchProviderResponse,
    VisualGroundingQueryResult,
)
from agentos.schemas.enums import SkillStatus
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.task_reasoning import (
    AgentOSSkillDelegation,
    GroundedNavigationSlowBrainOutput,
    InstructionFollowingCoT,
    NavigationSlowBrainOutput,
    ObjectGoalSlowBrainSemanticOutput,
    ObjectGoalCoT,
    SlowBrainTaskSpecification,
    SlowBrainTopologyContext,
    SlowBrainSingleViewAudit,
    SlowBrainViewBindingVerification,
    SlowBrainVisualGrounding,
    RegionHypothesis,
    RegionSearchPlan,
    SemanticNodeSelection,
    SemanticTaskDecompositionInput,
)
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.navigation.agentos_hierarchical import (
    AgentOSHierarchicalNavigationSkill,
    AgentOSLocalizationMapSkill,
    _slow_topology_context,
)
from agentos.skills.navigation.persistent import GuidedOmniNavOutcome
from agentos.skills.perception import VisualGroundingSkill


def _delegation() -> dict:
    return {
        "skill_name": "NavigationSkill",
        "task_specification": {
            "task_type": "object_goal",
            "goal": "go to bedroom to find a bed",
        },
        "planner_output": {
            "global_instruction": "go to bedroom to find a bed",
            "destination_query": "bedroom",
            "object_goal": "bed",
            "node_selection": {
                "selected_topology_node_id": "backbone-003",
                "confidence": 0.95,
                "matched_concepts": ["bedroom", "bed"],
                "selection_reason": "The semantic node is the bedroom region.",
            },
            "subtasks": [
                {
                    "stage": "reaching",
                    "task_type": "object_goal",
                    "instruction": "Find the bed inside the bedroom region.",
                    "target": "bed",
                    "topology_node_id": None,
                    "success_condition": "A bed is clearly visible in the current observation.",
                }
            ],
        },
        "delegated_subtask_index": 0,
        "rationale_summary": "Search for the requested object within the bedroom region.",
    }


def _region_search(active_node_id: str = "backbone-003") -> dict:
    return {
        "candidates": [
            {
                "topology_node_id": active_node_id,
                "confidence": 0.92,
                "place_evidence": "The place type matches the requested bedroom.",
                "object_evidence": "The node semantics support searching for a bed.",
                "relation_reason": "Search this destination-compatible region first.",
            },
        ],
        "active_topology_node_id": active_node_id,
        "update_reason": "Rank place compatibility before object evidence.",
    }


def _slow_pixel_output() -> dict:
    return {
        "cot": {
            "target_localization_and_key_context": "The bed is visible in the front view.",
            "path_planning": "Move on the clear floor toward the bed.",
        },
        "affordance_pixel": {
            "left": [],
            "front": [500, 800],
            "right": [],
        },
        "target_pixel": {
            "left": [],
            "front": [500, 420],
            "right": [],
        },
    }


def _slow_semantic_output(
    *,
    target: bool = True,
) -> dict:
    return {
        "cot": {
            "target_localization_and_key_context": "The bed is visible in the front view.",
            "path_planning": "Move on the clear floor toward the bed.",
        },
        "affordance_query": {
            "role": "affordance",
            "label": "clear floor toward the bed",
            "preferred_view": "front",
            "visual_description": "Clear traversable floor leads toward the bed.",
            "spatial_relation": "Continue through the bedroom on clear floor toward the bed.",
            "excluded_labels": [],
        },
        "target_query": (
            {
                "role": "target",
                "label": "bed",
                "preferred_view": "front",
                "visual_description": "A bed clearly visible in the bedroom.",
                "spatial_relation": "Inside the bedroom ahead of the robot.",
                "excluded_labels": ["robot", "cabinet"],
            }
            if target
            else None
        ),
    }


def _instruction_slow_semantic_output() -> dict:
    return {
        "cot": {
            "finished_sub_instruction": "The robot has not completed a route segment yet.",
            "ongoing_sub_instruction": "Exit through the visible doorway and continue toward the robot.",
        },
        "affordance_query": {
            "role": "affordance",
            "label": "visible doorway",
            "preferred_view": "front",
            "visual_description": "A clear doorway leading out of the room.",
            "spatial_relation": "The doorway advances the instructed exit route.",
            "excluded_labels": [],
        },
        "target_query": {
            "role": "target",
            "label": "robot",
            "preferred_view": "front",
            "visual_description": "The requested robot at the end of the route.",
            "spatial_relation": "The robot is the terminal object in the route instruction.",
            "excluded_labels": [],
        },
    }


def _grounded_output(*, target: bool = True, target_confidence: float = 0.97) -> dict:
    return {
        "cot": _slow_semantic_output()["cot"],
        "affordance_grounding": {
            "label": "clear floor toward the bed",
            "view": "front",
            "bbox_normalized_1000": [400, 600, 600, 800],
            "confidence": 0.94,
            "evidence_summary": "Clear traversable floor leads toward the bed.",
        },
        "target_grounding": (
            {
                "label": "bed",
                "view": "front",
                "bbox_normalized_1000": [450, 250, 550, 420],
                "confidence": target_confidence,
                "evidence_summary": "The bed is clearly visible in the front view.",
            }
            if target
            else {
                "label": None,
                "view": None,
                "bbox_normalized_1000": [],
                "confidence": 0.0,
                "evidence_summary": "",
            }
        ),
    }


def _execution_telemetry(
    *,
    reached: bool,
    progress_m: float,
    stalled: bool = False,
    remaining_distance_m: float = 0.0,
) -> dict:
    return {
        "state": "reached" if reached else "stalled" if stalled else "progressing",
        "goal_reached": reached,
        "motion_attempted": progress_m > 0.0,
        "progress_m": progress_m,
        "remaining_distance_m": remaining_distance_m,
        "stalled": stalled,
        "termination_reason": "reached" if reached else "stalled" if stalled else "chunk_complete",
        "source": "test-adapter",
    }


class _Provider:
    def __init__(
        self,
        payload: dict | list[dict],
        *,
        view_bindings: dict | None = None,
    ) -> None:
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.schemas: list[dict | None] = []
        self.messages: list[list] = []
        self.delegation_messages: list[list] = []
        self.delegation_calls = 0
        self.view_binding_calls = 0
        self.view_bindings = view_bindings

    async def complete(self, messages, *, response_schema=None):
        self.messages.append(messages)
        self.schemas.append(response_schema)
        if (
            response_schema
            and response_schema.get("title") == SlowBrainSingleViewAudit.__name__
        ):
            system_text = str(messages[0].content)
            current_view = next(
                view
                for view in ("left", "front", "right")
                if f"robot-{view.upper()}" in system_text
            )
            payload = {
                "view": current_view,
                "scene_summary": f"Current {current_view} camera scene.",
                "visible_objects": ["bed", "doorway"],
                "navigable_semantic_anchors": ["visible doorway"],
                "target_relevant_evidence": ["A bed is visible."],
            }
        elif (
            response_schema
            and response_schema.get("title")
            == SlowBrainViewBindingVerification.__name__
        ):
            system_text = str(messages[0].content)
            current_view = next(
                view
                for view in ("left", "front", "right")
                if f"robot-{view.upper()}" in system_text
            )
            source = self.payloads[
                min(max(self.delegation_calls - 1, 0), len(self.payloads) - 1)
            ]
            if self.view_bindings is None:
                queries = [source["affordance_query"]]
                target = source.get("target_query")
                if target is not None and target.get("preferred_view") is not None:
                    queries.append(target)
                desired = {
                    "bindings": [
                        {
                            "role": query["role"],
                            "label": query["label"],
                            "visually_supported": True,
                            "verified_view": query["preferred_view"],
                            "confidence": 0.95,
                            "visual_description": query["visual_description"],
                            "evidence_summary": "The selected current view supports the query.",
                        }
                        for query in queries
                    ]
                }
            else:
                desired = self.view_bindings
            payload = {"bindings": []}
            for binding in desired["bindings"]:
                supported = bool(binding["visually_supported"]) and (
                    binding["verified_view"] == current_view
                )
                payload["bindings"].append(
                    {
                        **binding,
                        "visually_supported": supported,
                        "verified_view": current_view if supported else None,
                        "confidence": (
                            float(binding.get("confidence", 0.95)) if supported else 0.0
                        ),
                        "visual_description": (
                            binding.get("visual_description", "") if supported else ""
                        ),
                    }
                )
            self.view_binding_calls += 1
        else:
            payload = self.payloads[
                min(self.delegation_calls, len(self.payloads) - 1)
            ]
            self.delegation_calls += 1
            self.delegation_messages.append(messages)
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=payload,
        )

    async def close(self):
        return None


class _SemanticRegionAudit:
    def __init__(self, selected_topology_node_id: str) -> None:
        self.selected_topology_node_id = selected_topology_node_id
        self.calls: list[tuple[object, list[dict[str, object]]]] = []

    async def rank_regions_async(self, request, nodes):
        self.calls.append((request, nodes))
        return RegionSearchPlan(
            candidates=[
                RegionHypothesis(
                    topology_node_id=self.selected_topology_node_id,
                    confidence=0.96,
                    place_evidence="The requested place_type is an exact semantic match.",
                    object_evidence="The object remains unresolved in this region.",
                    relation_reason="The requested place_type is primary regional evidence.",
                )
            ],
            active_topology_node_id=self.selected_topology_node_id,
            update_reason="Ranked from place and object relation evidence.",
        )


class _TaskBoundaryRegionAudit(_SemanticRegionAudit):
    def __init__(self, selected_topology_node_id: str) -> None:
        super().__init__(selected_topology_node_id)
        self.decomposition_calls = 0

    async def decompose_task_async(self, global_goal):
        self.decomposition_calls += 1
        return SemanticTaskDecompositionInput(
            global_goal=global_goal,
            destination_query="office",
            object_goal="white refrigerator",
        )


def _business_schemas(provider: _Provider) -> list[dict | None]:
    return [
        schema
        for schema in provider.schemas
        if (schema or {}).get("title") != SlowBrainSingleViewAudit.__name__
    ]


class _CurrentObservationGrounder:
    def __init__(self, detections_by_role: dict[str, list[VisualDetection]]) -> None:
        self.detections_by_role = detections_by_role
        self.requests = []

    async def ground_batch(self, request):
        self.requests.append(request)
        return VisualGroundingBatchProviderResponse(
            provider="current-frame-grounder",
            results=[
                VisualGroundingQueryResult(
                    role=query.role,
                    query=query.query,
                    detections=self.detections_by_role.get(query.role, []),
                )
                for query in request.queries
            ],
        )

    async def close(self):
        return None


def _default_grounding_skill(
    *,
    affordance: bool = True,
    target: bool = True,
    target_confidence: float = 0.97,
) -> VisualGroundingSkill:
    detections: dict[str, list[VisualDetection]] = {
        "affordance": (
            [
                VisualDetection(
                    label="clear floor toward the bed",
                    view="front",
                    bbox_normalized_1000=[400, 600, 600, 800],
                    confidence=0.94,
                    evidence_summary="Clear traversable floor leads toward the bed.",
                )
            ]
            if affordance
            else []
        ),
        "target": (
            [
                VisualDetection(
                    label="bed",
                    view="front",
                    bbox_normalized_1000=[450, 250, 550, 420],
                    confidence=target_confidence,
                    evidence_summary="The bed is clearly visible in the front view.",
                )
            ]
            if target
            else []
        ),
    }
    return VisualGroundingSkill(lambda: _CurrentObservationGrounder(detections))


class _DelayedProvider(_Provider):
    async def complete(self, messages, *, response_schema=None):
        if self.schemas:
            await asyncio.sleep(0.02)
        return await super().complete(messages, response_schema=response_schema)


class _SlowTimeoutProvider(_Provider):
    async def complete(self, messages, *, response_schema=None):
        if self.schemas:
            await asyncio.sleep(0.1)
        return await super().complete(messages, response_schema=response_schema)


class _Service:
    def __init__(self, observation: Observation) -> None:
        self.observation = observation
        self.prepared: list[dict] = []
        self.pixel_calls: list[dict] = []

    def begin_navigation(self, payload=None):
        return {
            "ok": True,
            "observation": self.observation.model_dump(mode="json"),
            "slow_context": {
                "topology_context": {
                    "topo_memory": {
                        "block_layer": [
                            {"topology_node_id": "backbone-003", "place_type": "bedroom"}
                        ]
                    }
                }
            },
        }

    def localization_topology_snapshot(self, observation):
        return {
            "observation_id": observation.trace.observation_id,
            "pose": {"x": 1.0, "y": 2.0, "yaw": 0.5},
            "topdown_ref": "mock://topdown.png",
            "map_id": "test-floor",
            "current_topology_node_id": "backbone-001",
            "nearby_nodes": [
                {
                    "topology_node_id": "backbone-002",
                    "distance_m": 1.5,
                    "scene_summary": "Connected corridor.",
                }
            ],
            "topo_memory": self.begin_navigation()["slow_context"]["topology_context"]["topo_memory"],
            "localization_source": "test",
        }

    def prepare_agentic_plan(self, payload):
        self.prepared.append(payload)
        return {
            "ok": True,
            "status": "skill_ready",
            "observation": self.observation.model_dump(mode="json"),
            "semantic_route_plan": {
                "regional_guidance": {
                    "mode": "search_within_region",
                    "current_topology_node_id": "backbone-001",
                    "target_topology_node_id": "backbone-003",
                    "next_topology_node_id": "backbone-002",
                    "route_node_ids": [
                        "backbone-001",
                        "backbone-002",
                        "backbone-003",
                    ],
                    "target_region_reached": False,
                    "reason": "Follow the connected route into the bedroom region.",
                }
            },
            "slow_context": {
                "topology_context": {
                    "topo_memory": {
                        "map_id": "test-floor",
                        "revision": 3,
                        "current_topology_node_id": "backbone-001",
                        "block_layer": [
                            {
                                "topology_node_id": "backbone-001",
                                "place_type": "office",
                                "scene_summary": "Office starting region.",
                            },
                            {
                                "topology_node_id": "backbone-002",
                                "place_type": "corridor",
                                "scene_summary": "Corridor connecting office and bedroom.",
                            },
                            {
                                "topology_node_id": "backbone-003",
                                "place_type": "bedroom",
                                "scene_summary": "Bedroom region likely to contain a bed.",
                            },
                        ],
                        "road_layer": [
                            {
                                "source": "backbone-001",
                                "target": "backbone-002",
                                "distance_m": 2.0,
                                "transition": "office doorway",
                            },
                            {
                                "source": "backbone-002",
                                "target": "backbone-003",
                                "distance_m": 3.0,
                                "transition": "connected corridor",
                            },
                        ],
                        "function_layer": [
                            {
                                "topology_node_id": "backbone-003",
                                "function": "sleeping area",
                                "task_affordances": ["find bed"],
                            }
                        ],
                        "object_poi_layer": [
                            {
                                "topology_node_id": "backbone-003",
                                "objects": ["bed"],
                                "landmarks": ["wooden door"],
                            }
                        ],
                    }
                }
            },
        }

    def navigate_to_anchor(self, payload, *, allow_stale_slow_reference=False):
        assert allow_stale_slow_reference is False
        self.pixel_calls.append(payload)
        return {
            "ok": True,
            "status": "completed",
            "generated_point": {
                "annotated_ref": "mock://slow-target.jpg",
                "evidence_summary": "Slow Brain target pixel.",
            },
            "result": {
                "controller": "habitat_greedy_follower",
                "reached": True,
                "control_steps": 8,
                "distance_m": 1.5,
                "trajectory_keyframes": [],
            },
            "execution_telemetry": _execution_telemetry(
                reached=True,
                progress_m=1.5,
            ),
            "post_observation": self.observation.model_dump(mode="json"),
            "next_observation_id": self.observation.trace.observation_id,
        }

    def navigate_to_known_location(self, payload):
        raise AssertionError(f"unexpected known-location goal: {payload}")

    def adjust_viewpoint(self, payload):
        raise AssertionError(f"unexpected viewpoint goal: {payload}")


class _NodePointGoalService(_Service):
    def __init__(self, observation: Observation) -> None:
        super().__init__(observation)
        self.location_calls: list[dict] = []

    def prepare_agentic_plan(self, payload):
        packet = super().prepare_agentic_plan(payload)
        packet["semantic_route_plan"]["regional_guidance"]["mode"] = (
            "search_within_region" if self.location_calls else "route_to_region"
        )
        return packet

    def navigate_to_known_location(self, payload):
        self.location_calls.append(payload)
        return {
            "ok": True,
            "status": "completed",
            "result": {
                "controller": "habitat_greedy_follower",
                "reached": True,
                "control_steps": 12,
                "distance_m": 2.5,
                "termination_reason": "reached",
                "trajectory_keyframes": [],
            },
            "execution_telemetry": _execution_telemetry(
                reached=True,
                progress_m=2.5,
            ),
            "post_observation": self.observation.model_dump(mode="json"),
            "next_observation_id": self.observation.trace.observation_id,
        }


class _ChunkedService(_Service):
    def navigate_to_anchor(self, payload, *, allow_stale_slow_reference=False):
        assert allow_stale_slow_reference is False
        self.pixel_calls.append(payload)
        time.sleep(0.04)
        reached = len(self.pixel_calls) > 1
        return {
            "ok": reached,
            "status": "completed" if reached else "incomplete",
            "generated_point": {
                "annotated_ref": "mock://slow-target.jpg",
                "evidence_summary": "Slow Brain target pixel.",
            },
            "result": {
                "controller": "habitat_greedy_follower",
                "reached": reached,
                "control_steps": 8,
                "distance_m": 1.0,
                "remaining_distance_m": 0.0 if reached else 1.0,
                "stalled": False,
                "trajectory_keyframes": [],
            },
            "execution_telemetry": _execution_telemetry(
                reached=reached,
                progress_m=1.0,
                remaining_distance_m=0.0 if reached else 1.0,
            ),
            "post_observation": self.observation.model_dump(mode="json"),
            "next_observation_id": self.observation.trace.observation_id,
        }


class _DepthLimitedTargetService(_Service):
    anchor_max_depth_m = 5.0

    def navigate_to_anchor(self, payload, *, allow_stale_slow_reference=False):
        packet = super().navigate_to_anchor(
            payload,
            allow_stale_slow_reference=allow_stale_slow_reference,
        )
        depth_limited = len(self.pixel_calls) == 1
        packet["anchor_depth_limited"] = depth_limited
        packet["anchor_max_depth_m"] = payload["anchor_max_depth_m"]
        packet["raw_anchor_depth_m"] = 6.0 if depth_limited else 4.0
        packet["generated_point"]["depth_limited"] = depth_limited
        return packet


class _UnreachableAnchorService(_Service):
    def navigate_to_anchor(self, payload, *, allow_stale_slow_reference=False):
        del payload, allow_stale_slow_reference
        raise ValueError("pathfinder could not find a route to the grounded anchor")


def _observation(make_trace, tmp_path) -> Observation:
    trace = make_trace(observation_id="obs-current")
    rgb_refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"{view}.jpg"
        Image.new("RGB", (640, 480), color=(32, 32, 32)).save(path)
        rgb_refs[view] = str(path)
    return Observation(
        trace=trace,
        rgb_refs=rgb_refs,
        depth_refs={
            "left": "mock://left-depth.npy",
            "front": "mock://front-depth.npy",
            "right": "mock://right-depth.npy",
        },
    )


def test_main_agent_outputs_only_skill_delegation(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["planner_output"]["node_selection"] = None
    provider = _Provider(payload)
    service = _Service(observation)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=service.begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, service, session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    schema_properties = _business_schemas(provider)[0]["properties"]
    planner_schema = _business_schemas(provider)[0]["$defs"]["AgenticPlannerOutput"]
    assert "planner_output" in schema_properties
    assert "region_search" in planner_schema["required"]
    assert "action" not in schema_properties
    assert "self_reflection" not in schema_properties
    assert "semantic_map" not in schema_properties
    assert plan.steps[0].skill_name == "NavigationSkill"
    assert set(plan.steps[0].inputs) == {
        "delegation",
        "source_observation_id",
        "replace_active_plan",
        "verifier_feedback",
    }
    assert "action" not in plan.steps[0].inputs
    assert plan.metadata["region_search"] is None
    assert session.region_search_state is None


def test_main_agent_derives_redundant_skill_name_from_delegated_subtask(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["skill_name"] = "NavigationSkill"
    payload["planner_output"]["subtasks"] = [
        {
            "skill_name": "MappingSkill",
            "stage": "approaching",
            "task_type": "object_goal",
            "instruction": "Acquire current map and topology evidence.",
            "target": "current spatial evidence",
            "topology_node_id": None,
            "success_condition": "Current pose and topology evidence are available.",
        },
        payload["planner_output"]["subtasks"][0],
    ]
    payload["delegated_subtask_index"] = 0
    provider = _Provider(payload)
    session = AgentOSMainAgentSession(global_goal="go to bedroom to find a bed")
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 1
    assert plan.steps[0].skill_name == "MappingSkill"
    assert plan.metadata["delegated_skill"] == "MappingSkill"


def test_main_agent_delegation_always_uses_runtime_observation(make_trace, tmp_path):
    initial = _observation(make_trace, tmp_path)
    current = initial.model_copy(
        update={"trace": make_trace(observation_id="obs-after-skill")}
    )
    payload = _delegation()
    payload["planner_output"]["node_selection"] = None
    provider = _Provider(payload)
    service = _Service(initial)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot={"observation": initial.model_dump(mode="json")},
        mapping_observation_id=current.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, service, session)
    task = TaskRequest(trace=current.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, current))

    assert plan.steps[0].inputs["source_observation_id"] == "obs-after-skill"


def test_main_agent_plans_skill_series_and_delegates_only_current_skill(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["skill_name"] = "MappingSkill"
    payload["planner_output"]["subtasks"] = [
        {
            "skill_name": "MappingSkill",
            "stage": "approaching",
            "task_type": "object_goal",
            "instruction": "Inspect current localization and connected semantic regions.",
            "target": "current map context",
            "topology_node_id": None,
            "success_condition": "Current pose and connected topology evidence are available.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to the mapped bedroom region at backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "Current evidence confirms arrival near the bedroom region.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the bedroom region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and the robot has approached it.",
        },
    ]
    payload["planner_output"]["region_search"] = _region_search()
    payload["delegated_subtask_index"] = 0
    provider = _Provider(payload)
    service = _Service(observation)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=service.begin_navigation(),
        mapping_observation_id="obs-previous-map-snapshot",
    )
    planner = AgentOSHierarchicalPlanner(provider, service, session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert plan.steps[0].skill_name == "MappingSkill"
    assert plan.steps[0].command.startswith("Inspect current localization")
    assert set(plan.steps[0].inputs) == {
        "source_observation_id",
        "verifier_feedback",
    }
    assert [item["skill_name"] for item in plan.metadata["planned_skill_subtasks"]] == [
        "MappingSkill",
        "NavigationSkill",
        "NavigationSkill",
    ]
    planned = plan.metadata["planned_skill_subtasks"]
    assert [item["task_type"] for item in planned] == [
        "object_goal",
        "point_goal",
        "object_goal",
    ]
    assert planned[1]["topology_node_id"] == "backbone-003"
    assert planned[2]["target"] == "bed"


def test_main_agent_does_not_repeat_mapping_for_the_same_observation(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    localization = _delegation()
    localization["skill_name"] = "MappingSkill"
    localization["planner_output"]["subtasks"] = [
        {
            "skill_name": "MappingSkill",
            "stage": "approaching",
            "task_type": "object_goal",
            "instruction": "Inspect current localization and topology.",
            "target": "current spatial evidence",
            "topology_node_id": None,
            "success_condition": "Current pose and connected topology evidence are available.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the bedroom region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and approached.",
        },
    ]
    localization["delegated_subtask_index"] = 0
    navigation = _delegation()
    navigation["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to the mapped bedroom region at backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches the mapped bedroom region.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the bedroom region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and approached.",
        },
    ]
    navigation["planner_output"]["region_search"] = _region_search()
    navigation["delegated_subtask_index"] = 0
    provider = _Provider([localization, localization, navigation])
    session = AgentOSMainAgentSession(global_goal="go to bedroom to find a bed")
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    first_plan = asyncio.run(planner.plan(task, observation))
    localization_result = SkillResult(
        trace=observation.trace,
        skill_name="MappingSkill",
        status=SkillStatus.SUCCESS,
        summary=(
            'Mapping evidence: {"block_layer":[{"topology_node_id":'
            '"backbone-003","place_type":"bedroom"}]}'
        ),
        compressed_history=(
            'Mapping evidence: {"block_layer":[{"topology_node_id":'
            '"backbone-003","place_type":"bedroom"}]}'
        ),
    )
    revised = asyncio.run(
        planner.replan(
            task,
            observation,
            first_plan,
            [localization_result],
            reason="Localization evidence was acquired; global navigation remains.",
            next_revision=1,
        )
    )

    assert revised.steps[0].skill_name == "NavigationSkill"
    assert revised.steps[0].command == (
        "Navigate to the mapped bedroom region at backbone-003."
    )
    assert [
        item["task_type"] for item in revised.metadata["planned_skill_subtasks"]
    ] == ["point_goal", "object_goal"]
    assert provider.delegation_calls == 3
    assert session.mapping_observation_id == observation.trace.observation_id
    assert session.progress_ledger[-1]["delegated_subtask"]["skill_name"] == (
        "MappingSkill"
    )
    correction = provider.delegation_messages[2][-1].content
    assert "already returned evidence for the current observation" in correction
    assert any(
        "backbone-003" in str(message.content)
        for message in provider.delegation_messages[1]
    )


def test_main_agent_retries_when_node_target_is_not_executable(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    incomplete = _delegation()
    incomplete["planner_output"]["node_selection"] = None
    incomplete["planner_output"]["destination_query"] = "backbone-003"
    incomplete["planner_output"]["object_goal"] = "backbone-003"
    incomplete["planner_output"]["subtasks"][0].update(
        {
            "target": "backbone-003",
            "topology_node_id": None,
        }
    )
    corrected = _delegation()
    corrected["planner_output"]["destination_query"] = "backbone-003"
    corrected["planner_output"]["object_goal"] = "backbone-003"
    corrected["planner_output"]["node_selection"] = None
    corrected["task_specification"]["task_type"] = "point_goal"
    corrected["planner_output"]["subtasks"][0].update(
        {
            "task_type": "point_goal",
            "instruction": "Navigate to backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches backbone-003.",
        }
    )
    provider = _Provider([incomplete, corrected])
    session = AgentOSMainAgentSession(global_goal="go to bedroom to find a bed")
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 2
    assert plan.metadata["delegated_subtask"]["topology_node_id"] == "backbone-003"
    assert (
        "must also populate topology_node_id"
        in provider.delegation_messages[1][-1].content
    )


def test_main_agent_rejects_node_binding_inside_object_goal(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    invalid = _delegation()
    invalid["planner_output"]["destination_query"] = "bed"
    invalid["planner_output"]["node_selection"] = None
    invalid["planner_output"]["subtasks"][0]["topology_node_id"] = "backbone-003"
    corrected = _delegation()
    corrected["planner_output"]["destination_query"] = "bed"
    corrected["planner_output"]["node_selection"] = None
    provider = _Provider([invalid, corrected])
    session = AgentOSMainAgentSession(global_goal="go to bedroom to find a bed")
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 2
    assert plan.metadata["delegated_subtask"]["topology_node_id"] is None
    assert "object_goal must keep topology_node_id null" in (
        provider.delegation_messages[1][-1].content
    )


def test_slow_affordance_requires_a_supporting_current_view():
    payload = _slow_semantic_output()
    payload["affordance_query"]["preferred_view"] = None

    with pytest.raises(
        ValueError,
        match="affordance_query.preferred_view",
    ):
        ObjectGoalSlowBrainSemanticOutput.model_validate(payload)


def test_slow_grounding_query_rejects_a_camera_name_as_the_anchor_label():
    payload = _slow_semantic_output()
    payload["affordance_query"]["label"] = "FRONT"

    with pytest.raises(ValueError, match="not a camera view"):
        ObjectGoalSlowBrainSemanticOutput.model_validate(payload)


def test_main_agent_owns_complete_skill_plan_without_a_separate_plan_reviewer(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    corrected = _delegation()
    corrected["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to the mapped bedroom region at backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches the mapped bedroom region.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the bedroom region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and approached.",
        },
    ]
    corrected["planner_output"]["node_selection"] = {
        "selected_topology_node_id": "backbone-003",
        "confidence": 0.95,
        "matched_concepts": ["bedroom", "bed"],
        "selection_reason": "The semantic node is the mapped bedroom region.",
    }
    corrected["planner_output"]["region_search"] = _region_search()
    corrected["delegated_subtask_index"] = 0
    provider = _Provider(corrected)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=_Service(observation).begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 1
    assert [
        item["task_type"] for item in plan.metadata["planned_skill_subtasks"]
    ] == ["point_goal", "object_goal"]
    planning_prompt = provider.delegation_messages[0][1].content[0]["text"]
    assert "Emit the minimum complete plan" in planning_prompt
    assert "internally check" in planning_prompt
    assert "First select destination-compatible nodes from place_type" in planning_prompt
    assert "Decide semantically whether" in planning_prompt
    assert plan.metadata["region_search"]["active_topology_node_id"] == "backbone-003"
    assert session.region_search_state == plan.metadata["region_search"]
    execution = plan.steps[0].inputs["delegation"]
    assert execution["planner_output"]["region_search"] == plan.metadata["region_search"]


def test_main_agent_recomputes_region_order_from_model_semantic_audit(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)

    def candidate(node_id: str) -> dict:
        payload = _delegation()
        payload["planner_output"]["subtasks"] = [
            {
                "skill_name": "NavigationSkill",
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": f"Navigate to {node_id}.",
                "target": node_id,
                "topology_node_id": node_id,
                "success_condition": f"The robot reaches {node_id}.",
            },
            {
                "skill_name": "NavigationSkill",
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Search the office for the refrigerator.",
                "target": "refrigerator",
                "topology_node_id": None,
                "success_condition": "The refrigerator is grounded and reached.",
            },
        ]
        payload["planner_output"]["destination_query"] = "office"
        payload["planner_output"]["object_goal"] = "refrigerator"
        payload["planner_output"]["node_selection"] = {
            "selected_topology_node_id": node_id,
            "confidence": 0.94,
            "matched_concepts": ["office", "refrigerator"],
            "selection_reason": "Model-ranked regional hypothesis.",
        }
        payload["planner_output"]["region_search"] = _region_search(node_id)
        payload["delegated_subtask_index"] = 0
        payload["task_specification"] = {
            "task_type": "point_goal",
            "goal": f"Navigate to {node_id}.",
        }
        return payload

    context_snapshot = {
        "observation": observation.model_dump(mode="json"),
        "slow_context": {
            "topology_context": {
                "topo_memory": {
                    "block_layer": [
                        {
                            "topology_node_id": "backbone-001",
                            "place_type": "office area",
                            "scene_summary": "Office work area.",
                            "goal_search": [],
                        },
                        {
                            "topology_node_id": "backbone-004",
                            "place_type": "living room",
                            "scene_summary": "Living room with appliance landmark.",
                            "goal_search": [],
                        },
                    ],
                    "function_layer": [],
                    "object_poi_layer": [
                        {
                            "topology_node_id": "backbone-004",
                            "objects": ["tall refrigerator"],
                            "landmarks": [],
                        }
                    ],
                }
            }
        },
    }
    provider = _Provider([candidate("backbone-004"), candidate("backbone-001")])
    audit = _SemanticRegionAudit("backbone-001")
    session = AgentOSMainAgentSession(
        global_goal="go to office to find a refrigerator",
        context_snapshot=context_snapshot,
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(
        provider,
        _Service(observation),
        session,
        semantic_node_ranker=audit,
    )
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 2
    assert len(audit.calls) == 2
    assert plan.metadata["region_search"]["active_topology_node_id"] == "backbone-001"
    assert plan.metadata["delegated_subtask"]["topology_node_id"] == "backbone-001"
    assert "Independent semantic region audit selected backbone-001" in (
        provider.delegation_messages[1][-1].content
    )


def test_region_ranker_does_not_redecompose_the_main_agent_task(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["planner_output"]["destination_query"] = "white refrigerator"
    payload["planner_output"]["object_goal"] = "white refrigerator"
    payload["planner_output"]["region_search"] = _region_search("backbone-002")
    delegation = AgentOSSkillDelegation.model_validate(payload)
    ranker = _TaskBoundaryRegionAudit("backbone-001")
    planner = AgentOSHierarchicalPlanner(
        _Provider(payload),
        _Service(observation),
        AgentOSMainAgentSession(global_goal="go to office to find a white refrigerator"),
        semantic_node_ranker=ranker,
    )
    planning_context = {
        "stable_global_task": "go to office to find a white refrigerator",
        "topo_memory": {
            "topo_memory": {
                "block_layer": [
                    {
                        "topology_node_id": "backbone-001",
                        "place_type": "office area",
                    },
                    {
                        "topology_node_id": "backbone-002",
                        "place_type": "open hall",
                    },
                ],
                "function_layer": [],
                "object_poi_layer": [],
            }
        },
    }

    with pytest.raises(ValueError, match="semantic region audit selected"):
        asyncio.run(
            planner._audit_region_selection(
                delegation,
                planning_context=planning_context,
            )
        )
    assert ranker.decomposition_calls == 0


def test_main_agent_preserves_active_region_search_until_object_goal_resolves(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["planner_output"]["node_selection"] = None
    payload["planner_output"]["region_search"] = None
    provider = _Provider([payload, payload, payload])
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=_Service(observation).begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
        region_search_state=_region_search(),
    )
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert plan.metadata["region_search"]["active_topology_node_id"] == (
        "backbone-003"
    )
    assert session.region_search_state == plan.metadata["region_search"]


def test_main_agent_restores_region_candidates_when_advancing_to_known_candidate(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to the next candidate region.",
            "target": "backbone-002",
            "topology_node_id": "backbone-002",
            "success_condition": "The robot reaches backbone-002.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is visibly grounded and approached.",
        },
    ]
    payload["planner_output"]["region_search"] = None
    payload["delegated_subtask_index"] = 0
    region_search = _region_search("backbone-001")
    region_search["candidates"].append(
        {
            "topology_node_id": "backbone-002",
            "confidence": 0.84,
            "place_evidence": "A secondary destination-compatible region.",
            "object_evidence": "The node contains relevant object evidence.",
            "relation_reason": "Search after the first region is exhausted.",
        }
    )
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=_Service(observation).begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
        region_search_state=region_search,
    )
    planner = AgentOSHierarchicalPlanner(
        _Provider(payload),
        _Service(observation),
        session,
    )

    audited = asyncio.run(
        planner._audit_region_selection(
            AgentOSSkillDelegation.model_validate(payload),
            planning_context={},
        )
    )

    restored = audited.planner_output.region_search
    assert restored is not None
    assert restored.active_topology_node_id == "backbone-002"
    assert [item.topology_node_id for item in restored.candidates] == [
        "backbone-001",
        "backbone-002",
    ]


def test_main_agent_exposes_compact_current_task_region_progress(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    region_search = _region_search("backbone-001")
    region_search["candidates"].append(
        {
            "topology_node_id": "backbone-002",
            "confidence": 0.84,
            "place_evidence": "A secondary destination-compatible region.",
            "object_evidence": "Static map prior contains a refrigerator.",
            "relation_reason": "Search after the first region is exhausted.",
        }
    )
    planner = AgentOSHierarchicalPlanner(
        _Provider(_delegation()),
        _Service(observation),
        AgentOSMainAgentSession(
            global_goal="go to office to find a white refrigerator",
            region_search_state=region_search,
        ),
    )
    topology_context = {
        "topo_memory": {
            "block_layer": [
                {
                    "topology_node_id": "backbone-001",
                    "visited": True,
                    "visit_count": 2,
                    "goal_search": [
                        {
                            "goal": "white refrigerator",
                            "status": "no_target_observed",
                            "attempt_count": 2,
                        }
                    ],
                },
                {
                    "topology_node_id": "backbone-002",
                    "visited": False,
                    "visit_count": 0,
                    "goal_search": [],
                },
            ]
        }
    }

    progress = planner._region_candidate_progress(topology_context)

    assert progress == [
        {
            "rank": 1,
            "topology_node_id": "backbone-001",
            "active": True,
            "prior_confidence": 0.92,
            "visited": True,
            "visit_count": 2,
            "goal_search": [
                {
                    "goal": "white refrigerator",
                    "status": "no_target_observed",
                    "attempt_count": 2,
                }
            ],
        },
        {
            "rank": 2,
            "topology_node_id": "backbone-002",
            "active": False,
            "prior_confidence": 0.84,
            "visited": False,
            "visit_count": 0,
            "goal_search": [],
        },
    ]


def test_main_agent_keeps_initial_region_candidate_order_across_replans(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    initial = _region_search("backbone-001")
    second = {
        "topology_node_id": "backbone-002",
        "confidence": 0.84,
        "place_evidence": "A secondary destination-compatible region.",
        "object_evidence": "Static map prior contains a refrigerator.",
        "relation_reason": "Search after the first region is exhausted.",
    }
    initial["candidates"].append(second)
    payload = _delegation()
    payload["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to backbone-002.",
            "target": "backbone-002",
            "topology_node_id": "backbone-002",
            "success_condition": "The robot reaches backbone-002.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Find the refrigerator.",
            "target": "refrigerator",
            "topology_node_id": None,
            "success_condition": "The refrigerator is grounded and approached.",
        },
    ]
    payload["planner_output"]["region_search"] = {
        "candidates": [second, initial["candidates"][0]],
        "active_topology_node_id": "backbone-002",
        "update_reason": "Advance after current-task negative evidence.",
    }
    planner = AgentOSHierarchicalPlanner(
        _Provider(payload),
        _Service(observation),
        AgentOSMainAgentSession(
            global_goal="go to office to find a refrigerator",
            region_search_state=initial,
        ),
    )

    audited = asyncio.run(
        planner._audit_region_selection(
            AgentOSSkillDelegation.model_validate(payload),
            planning_context={},
        )
    )

    stable = audited.planner_output.region_search
    assert stable is not None
    assert [item.topology_node_id for item in stable.candidates] == [
        "backbone-001",
        "backbone-002",
    ]
    assert stable.active_topology_node_id == "backbone-002"
    assert stable.update_reason == "Advance after current-task negative evidence."


def test_main_agent_cannot_repeat_a_verified_completed_subtask(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["task_specification"] = {
        "task_type": "point_goal",
        "goal": "Navigate to backbone-003.",
    }
    payload["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches backbone-003.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Find the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "The bed is grounded and reached.",
        },
    ]
    payload["planner_output"]["region_search"] = _region_search()
    repeated = AgentOSSkillDelegation.model_validate(payload)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=_Service(observation).begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
        region_search_state=_region_search(),
    )
    session.record_outcome(
        observation_id=observation.trace.observation_id,
        skill_name="NavigationSkill",
        status="success",
        summary="PointGoal reached.",
        verifier_feedback="Delegated Skill succeeded.",
        delegated_subtask=payload["planner_output"]["subtasks"][0],
        verified_success=True,
    )
    planner = AgentOSHierarchicalPlanner(
        _Provider(payload),
        _Service(observation),
        session,
    )

    with pytest.raises(ValueError, match="already passed Skill Verifier"):
        planner._validate_plan_protocol(
            repeated,
            current_observation_id=observation.trace.observation_id,
            planning_context={"mapping_evidence": {"available": True}},
        )


def test_main_agent_corrects_merged_place_object_plan_before_mapping(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    merged = _delegation()
    merged["task_specification"]["goal"] = "go to office to find a fridge"
    merged["planner_output"].update(
        {
            "global_instruction": "go to office to find a fridge",
            "destination_query": "office",
            "object_goal": "fridge",
            "node_selection": None,
            "subtasks": [
                {
                    "skill_name": "NavigationSkill",
                    "stage": "approaching",
                    "task_type": "object_goal",
                    "instruction": "go to office to find a fridge",
                    "target": "office",
                    "topology_node_id": None,
                    "success_condition": (
                        "The robot reaches the office and visually identifies a fridge."
                    ),
                }
            ],
        }
    )

    corrected = _delegation()
    corrected["skill_name"] = "MappingSkill"
    corrected["task_specification"]["goal"] = "Acquire current map evidence."
    corrected["planner_output"].update(
        {
            "global_instruction": "go to office to find a fridge",
            "destination_query": "office",
            "object_goal": "fridge",
            "node_selection": None,
            "subtasks": [
                {
                    "skill_name": "MappingSkill",
                    "stage": "approaching",
                    "task_type": "object_goal",
                    "instruction": "Acquire current localization and semantic topology evidence.",
                    "target": "current spatial evidence",
                    "topology_node_id": None,
                    "success_condition": (
                        "Current pose and semantic topology evidence are available."
                    ),
                },
            ],
        }
    )
    corrected["delegated_subtask_index"] = 0
    provider = _Provider([merged, corrected])
    session = AgentOSMainAgentSession(global_goal="go to office to find a fridge")
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 1
    assert plan.steps[0].skill_name == "MappingSkill"
    assert [
        (item["skill_name"], item["task_type"], item["target"])
        for item in plan.metadata["planned_skill_subtasks"]
    ] == [
        ("MappingSkill", "object_goal", "current spatial evidence"),
        ("NavigationSkill", "point_goal", "office"),
        ("NavigationSkill", "object_goal", "fridge"),
    ]
    assert plan.metadata["planned_skill_subtasks"][1]["topology_node_id"] is None
    planning_prompt = provider.delegation_messages[0][1].content[0]["text"].lower()
    assert "emit the complete semantic task plan now" in planning_prompt
    assert "non-executable future point_goal placeholder" in planning_prompt


def test_main_agent_corrects_selected_region_without_point_then_object_plan(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    incomplete = _delegation()
    incomplete["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Find the bed in the bedroom.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and approached.",
        }
    ]

    corrected = _delegation()
    corrected["task_specification"]["task_type"] = "point_goal"
    corrected["task_specification"]["goal"] = (
        "Navigate to the mapped bedroom region at backbone-003."
    )
    corrected["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to the mapped bedroom region at backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches the mapped bedroom region.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the bedroom region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and approached.",
        },
    ]
    corrected["planner_output"]["region_search"] = _region_search()
    corrected["delegated_subtask_index"] = 0
    provider = _Provider([incomplete, corrected])
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=_Service(observation).begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 2
    assert [
        item["task_type"] for item in plan.metadata["planned_skill_subtasks"]
    ] == ["point_goal", "object_goal"]
    assert "point_goal" in provider.delegation_messages[1][-1].content
    assert "object_goal" in provider.delegation_messages[1][-1].content


def test_main_agent_requires_region_hypotheses_for_node_guided_object_search(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    missing = _delegation()
    missing["task_specification"]["task_type"] = "point_goal"
    missing["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches backbone-003.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Search the region and approach the bed.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is visibly grounded and approached.",
        },
    ]
    corrected = dict(missing)
    corrected["planner_output"] = dict(missing["planner_output"])
    corrected["planner_output"]["region_search"] = _region_search()
    provider = _Provider([missing, corrected])
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=_Service(observation).begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, _Service(observation), session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert provider.delegation_calls == 2
    assert "must include region_search" in provider.delegation_messages[1][-1].content
    assert plan.metadata["region_search"]["active_topology_node_id"] == "backbone-003"


def test_main_agent_does_not_expand_a_point_goal_into_route_hops(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    candidate = _delegation()
    candidate["task_specification"]["task_type"] = "point_goal"
    candidate["planner_output"]["subtasks"] = [
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "point_goal",
            "instruction": "Navigate to backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches backbone-003.",
        },
        {
            "skill_name": "NavigationSkill",
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": "Find the bed in the mapped region.",
            "target": "bed",
            "topology_node_id": None,
            "success_condition": "A bed is clearly visible and approached.",
        },
    ]
    candidate["planner_output"]["region_search"] = _region_search()
    candidate["delegated_subtask_index"] = 0
    provider = _Provider(candidate)
    service = _Service(observation)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot={
            "slow_context": {
                "topology_context": {
                    "topo_memory": {
                        "current_topology_node_id": "backbone-001",
                        "road_layer": [
                            {"source": "backbone-001", "target": "backbone-002"},
                            {"source": "backbone-002", "target": "backbone-003"},
                        ],
                    }
                }
            }
        },
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, service, session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))

    assert plan.metadata["delegated_subtask"]["topology_node_id"] == "backbone-003"
    assert provider.delegation_calls == 1
    assert len(plan.metadata["planned_skill_subtasks"]) == 2


def test_navigation_runner_receives_only_delegated_skill_subtask(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["planner_output"]["node_selection"] = None
    payload["planner_output"]["subtasks"].insert(
        0,
        {
            "skill_name": "MappingSkill",
            "stage": "approaching",
            "task_type": "object_goal",
            "instruction": "Inspect current localization.",
            "target": "map context",
            "topology_node_id": None,
            "success_condition": "Spatial evidence is available.",
        },
    )
    payload["delegated_subtask_index"] = 1
    provider = _Provider(payload)
    service = _Service(observation)
    session = AgentOSMainAgentSession(
        global_goal="go to bedroom to find a bed",
        context_snapshot=service.begin_navigation(),
        mapping_observation_id=observation.trace.observation_id,
    )
    planner = AgentOSHierarchicalPlanner(provider, service, session)
    task = TaskRequest(trace=observation.trace, instruction=session.global_goal)

    plan = asyncio.run(planner.plan(task, observation))
    execution_delegation = plan.steps[0].inputs["delegation"]

    assert plan.steps[0].skill_name == "NavigationSkill"
    assert execution_delegation["delegated_subtask_index"] == 0
    assert execution_delegation["task_specification"] == {
        "task_type": "object_goal",
        "goal": "Find the bed inside the bedroom region.",
    }
    assert execution_delegation["planner_output"]["global_instruction"] == (
        "go to bedroom to find a bed"
    )
    assert len(execution_delegation["planner_output"]["subtasks"]) == 1
    assert execution_delegation["planner_output"]["subtasks"][0]["skill_name"] == (
        "NavigationSkill"
    )


def test_agentos_localization_skill_returns_bounded_spatial_evidence(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    service = _Service(observation)
    skill = AgentOSLocalizationMapSkill(service)
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Inspect current localization.",
        success_conditions=["Current pose and topology evidence are available."],
        inputs={"source_observation_id": observation.trace.observation_id},
    )
    context = SkillContext(
        event_bus=InMemoryEventBus(),
        cancellation=CancellationToken(),
    )

    result = asyncio.run(skill.run(request, observation, context))

    assert result.status.value == "success"
    assert result.outputs["map_id"] == "test-floor"
    assert result.outputs["nearby_nodes"][0]["topology_node_id"] == "backbone-002"
    assert result.evidence[0].kind == "localization_topology"


def test_high_confidence_target_preempts_affordance_for_fast_execution(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    current_output = _slow_semantic_output()
    current_output["cot"] = {
        "target_localization_and_key_context": (
            "The current front view independently confirms the bed in the bedroom."
        ),
        "path_planning": "Use the current clear floor evidence to approach the bed.",
    }
    provider = _Provider(current_output)
    service = _Service(observation)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        context_summary=(
            "Retrieved evidence: bed was visually grounded at backbone-003 on map test-floor."
        ),
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
            "replace_active_plan": False,
        },
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

    slow_schema = _business_schemas(provider)[0]
    schema_properties = slow_schema["properties"]
    assert "affordance_query" in schema_properties
    assert "target_query" in schema_properties
    assert "bbox_normalized_1000" not in str(slow_schema)
    assert "goal" not in schema_properties
    assert "planner_output" not in schema_properties
    assert "self_reflection" not in schema_properties
    assert "semantic_map" not in schema_properties
    prompt = str(provider.delegation_messages[0][-1].content)
    assert "explicit object must always populate target_query" in prompt
    assert service.prepared[0]["planner_output"]["object_goal"] == "bed"
    assert service.pixel_calls[0]["target_pixel"] == [320, 201]
    assert service.pixel_calls[0]["target_bbox_px"] == [288, 120, 351, 201]
    assert service.pixel_calls[0]["target_pixel_confidence"] == 0.97
    assert service.pixel_calls[0]["n1_target_pixel"] == [320, 201]
    assert service.pixel_calls[0]["n1_target_view"] == "front"
    assert service.pixel_calls[0]["n1_target_label"] == "bed"
    assert service.pixel_calls[0]["n1_target_bbox_px"] == [288, 120, 351, 201]
    assert service.pixel_calls[0]["n1_target_confidence"] == 0.97
    assert service.pixel_calls[0]["n1_affordance_view"] == "front"
    assert service.pixel_calls[0]["n1_affordance_pixel"] == [320, 383]
    assert service.pixel_calls[0]["n1_affordance_label"] == "clear floor toward the bed"
    assert service.pixel_calls[0]["n1_affordance_bbox_px"] == [256, 287, 383, 383]
    assert service.pixel_calls[0]["n1_affordance_confidence"] == 0.94
    assert service.pixel_calls[0]["view"] == "front"
    assert service.pixel_calls[0]["anchor_role"] == "target"
    assert service.pixel_calls[0]["anchor"] == "bed"
    assert len(_business_schemas(provider)) == 4
    assert provider.view_binding_calls == 3
    assert result.status.value == "success"
    assert result.outputs["slow_brain_output"]["target_pixel"]["front"] == [500, 420]
    assert result.outputs["slow_brain_output"]["affordance_pixel"]["front"] == [500, 800]
    assert result.outputs["semantic_completion"]["target_grounded"] is True
    assert result.outputs["semantic_completion"]["target_executed"] is True
    assert result.outputs["semantic_completion"]["target_reached"] is True
    assert result.outputs["semantic_completion"]["satisfied"] is True


def test_visual_grounding_skill_replaces_slow_bbox_with_current_observation(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    provider = _Provider(_slow_semantic_output())
    grounder = _CurrentObservationGrounder(
        {
            "affordance": [
                VisualDetection(
                    label="clear floor toward the bed",
                    view="front",
                    bbox_normalized_1000=[400, 600, 600, 800],
                    confidence=0.94,
                )
            ],
            "target": [VisualDetection(
                label="bed",
                view="right",
                bbox_normalized_1000=[100, 300, 300, 650],
                confidence=0.98,
                evidence_summary="The latest right view contains the bed.",
            )],
        }
    )
    grounding_skill = VisualGroundingSkill(lambda: grounder)
    service = _Service(observation)
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
        visual_grounding_skill=grounding_skill,
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
            "replace_active_plan": False,
        },
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

    assert grounder.requests[0].observation_id == observation.trace.observation_id
    assert grounder.requests[0].image_refs == observation.rgb_refs
    assert len(grounder.requests) == 1
    assert [query.role for query in grounder.requests[0].queries] == [
        "affordance",
        "target",
    ]
    target_query = next(
        query for query in grounder.requests[0].queries if query.role == "target"
    )
    assert target_query.constraints.model_dump(mode="json") == {
            "preferred_view": "front",
            "visual_description": "A bed clearly visible in the bedroom.",
            "excluded_labels": ["robot", "cabinet"],
    }
    assert service.pixel_calls[0]["view"] == "right"
    assert service.pixel_calls[0]["target_bbox_px"] == [64, 144, 192, 311]
    assert service.pixel_calls[0]["target_pixel"] == [128, 311]
    assert result.outputs["visual_grounding_skill_result"]["outputs"]["provider"] == (
        "current-frame-grounder"
    )


def test_slow_view_binding_verifier_corrects_conflicting_camera(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    semantic = _slow_semantic_output()
    semantic["cot"]["target_localization_and_key_context"] = (
        "CURRENT_VIEW_AUDIT: LEFT=wooden doorway; FRONT=robot testing area; "
        "RIGHT=cabinets and boxes."
    )
    semantic["affordance_query"].update(
        {
            "label": "wooden doorway",
            "preferred_view": "front",
            "visual_description": "A doorway near the front robot testing area.",
        }
    )
    provider = _Provider(
        semantic,
        view_bindings={
            "bindings": [
                {
                    "role": "affordance",
                    "label": "wooden doorway",
                    "visually_supported": True,
                    "verified_view": "left",
                    "visual_description": "A brown wooden doorway centered in the left image.",
                    "evidence_summary": "The exact doorway is visible only in LEFT.",
                },
                {
                    "role": "target",
                    "label": "bed",
                    "visually_supported": True,
                    "verified_view": "front",
                    "visual_description": "A bed visible in the front image.",
                    "evidence_summary": "The bed is visible in FRONT.",
                },
            ]
        },
    )
    grounder = _CurrentObservationGrounder(
        {
            "affordance": [
                VisualDetection(
                    label="wooden doorway",
                    view="left",
                    bbox_normalized_1000=[300, 200, 600, 850],
                    confidence=0.95,
                )
            ],
            "target": [
                VisualDetection(
                    label="bed",
                    view="front",
                    bbox_normalized_1000=[450, 250, 550, 420],
                    confidence=0.97,
                )
            ],
        }
    )
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        _Service(observation),
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
        visual_grounding_skill=VisualGroundingSkill(lambda: grounder),
    )
    result = asyncio.run(
        skill.run(
            SkillRequest(
                trace=observation.trace,
                skill_name=skill.name,
                command="Find the bed inside the bedroom region.",
                inputs={
                    "delegation": _delegation(),
                    "source_observation_id": observation.trace.observation_id,
                },
            ),
            observation,
            SkillContext(
                event_bus=InMemoryEventBus(),
                cancellation=CancellationToken(),
            ),
        )
    )

    affordance = result.outputs["slow_brain_semantic_decision"]["affordance_query"]
    assert affordance["preferred_view"] == "left"
    assert affordance["visual_description"] == (
        "A brown wooden doorway centered in the left image."
    )
    grounded_query = grounder.requests[0].queries[0]
    assert grounded_query.constraints.preferred_view == "left"
    assert grounded_query.constraints.visual_description == affordance["visual_description"]
    assert result.outputs["slow_view_binding_verification"]["bindings"][0][
        "verified_view"
    ] == "left"


def test_visual_grounding_miss_removes_stale_slow_target(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    grounder = _CurrentObservationGrounder(
        {
            "affordance": [
                VisualDetection(
                    label="clear floor toward the bed",
                    view="front",
                    bbox_normalized_1000=[400, 600, 600, 800],
                    confidence=0.94,
                )
            ],
            "target": [],
        }
    )
    service = _Service(observation)
    skill = AgentOSHierarchicalNavigationSkill(
        _Provider(_slow_semantic_output()),
        service,
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
        visual_grounding_skill=VisualGroundingSkill(lambda: grounder),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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

    assert service.pixel_calls[0]["anchor_role"] == "affordance"
    assert service.pixel_calls[0]["n1_target_pixel"] is None
    assert result.outputs["slow_brain_grounding"]["target_grounding"] == {
        "label": None,
        "view": None,
        "bbox_normalized_1000": [],
        "confidence": 0.0,
        "evidence_summary": "",
    }


def test_unreachable_grounded_pixel_returns_control_for_replanning(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    grounder = _CurrentObservationGrounder(
        {
            "affordance": [
                VisualDetection(
                    label="doorway",
                    view="front",
                    bbox_normalized_1000=[400, 300, 600, 700],
                    confidence=0.94,
                )
            ],
            "target": [],
        }
    )
    skill = AgentOSHierarchicalNavigationSkill(
        _Provider(_slow_semantic_output(target=False)),
        _UnreachableAnchorService(observation),
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
        visual_grounding_skill=VisualGroundingSkill(lambda: grounder),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Search the current region for the requested object.",
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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

    assert result.status == SkillStatus.RETRYABLE_FAILURE
    assert result.stop_reason == "fast_goal_invalid"
    assert "pathfinder could not find a route" in result.error


def test_low_confidence_target_falls_back_to_affordance(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    grounded = GroundedNavigationSlowBrainOutput.model_validate(
        _grounded_output(target_confidence=0.85)
    )
    output = NavigationSlowBrainOutput.model_validate(
        {
            "cot": grounded.cot,
            "affordance_pixel": {"left": [], "front": [500, 800], "right": []},
            "target_pixel": {"left": [], "front": [500, 420], "right": []},
        }
    )
    service = _Service(observation)
    skill = AgentOSHierarchicalNavigationSkill(
        _Provider(_slow_semantic_output()),
        service,
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
    )

    packet = skill._execute_fast(
        output,
        grounded,
        observation,
        AgentOSSkillDelegation.model_validate(_delegation()),
        allow_stale_slow_reference=False,
    )

    assert packet["ok"] is True
    assert service.pixel_calls[0]["target_pixel"] == [320, 383]
    assert service.pixel_calls[0]["target_bbox_px"] == [256, 287, 383, 383]
    assert service.pixel_calls[0]["target_pixel_confidence"] == 0.94
    assert service.pixel_calls[0]["anchor_role"] == "affordance"
    assert service.pixel_calls[0]["anchor"] == "clear floor toward the bed"
    assert service.pixel_calls[0]["n1_target_pixel"] == [320, 201]
    assert service.pixel_calls[0]["n1_target_confidence"] == 0.85
    assert service.pixel_calls[0]["n1_affordance_view"] == "front"
    assert service.pixel_calls[0]["n1_affordance_pixel"] == [320, 383]
    assert service.pixel_calls[0]["n1_affordance_label"] == "clear floor toward the bed"


def test_affordance_arrival_returns_control_for_semantic_verification():
    output = NavigationSlowBrainOutput.model_validate(_slow_pixel_output())

    assert AgentOSHierarchicalNavigationSkill._should_return_control(
        "object_goal",
        output,
        {
            "anchor_role": "affordance",
            "execution_telemetry": _execution_telemetry(
                reached=True, progress_m=1.0
            ),
        },
    ) is True
    assert AgentOSHierarchicalNavigationSkill._return_control_reason(
        "object_goal",
        output,
        {
            "anchor_role": "affordance",
            "execution_telemetry": _execution_telemetry(
                reached=True, progress_m=1.0
            ),
        },
    ) == "affordance_execution_chunk_reached"
    assert AgentOSHierarchicalNavigationSkill._should_return_control(
        "object_goal",
        output,
        {
            "anchor_role": "target",
            "execution_telemetry": _execution_telemetry(
                reached=True, progress_m=1.0
            ),
        },
    ) is True


def test_object_goal_natural_route_without_final_object_uses_only_affordance(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    delegation = _delegation()
    delegation["task_specification"] = {
        "task_type": "object_goal",
        "goal": "Go straight through the doorway, then turn left.",
    }
    delegation["planner_output"]["subtasks"][0].update(
        {
            "task_type": "object_goal",
            "instruction": "Go straight through the doorway, then turn left.",
            "target": "left corridor after the doorway",
        }
    )
    semantic_output = {
        "cot": {
            "target_localization_and_key_context": (
                "No final physical object is named; use the visible route description."
            ),
            "path_planning": "Move through the front doorway before turning left.",
        },
        "affordance_query": {
            "role": "affordance",
            "label": "traversable floor through the front doorway",
            "preferred_view": "front",
            "visual_description": "Open floor continuing through a doorway.",
            "spatial_relation": "Straight ahead through the doorway before the left turn.",
            "excluded_labels": [],
        },
        "target_query": None,
    }
    grounder = _CurrentObservationGrounder(
        {
            "affordance": [
                VisualDetection(
                    label="traversable floor through the front doorway",
                    view="front",
                    bbox_normalized_1000=[380, 500, 620, 900],
                    confidence=0.95,
                )
            ]
        }
    )
    service = _Service(observation)
    skill = AgentOSHierarchicalNavigationSkill(
        _Provider(semantic_output),
        service,
        N1NavigationSkillSession(global_goal=delegation["task_specification"]["goal"]),
        max_local_cycles=1,
        visual_grounding_skill=VisualGroundingSkill(lambda: grounder),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command=delegation["task_specification"]["goal"],
        inputs={
            "delegation": delegation,
            "source_observation_id": observation.trace.observation_id,
        },
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(InMemoryEventBus(), CancellationToken()),
        )
    )

    assert [query.role for query in grounder.requests[0].queries] == ["affordance"]
    assert result.outputs["slow_brain_semantic_decision"]["target_query"] is None
    assert result.outputs["slow_brain_output"]["target_pixel"] == {
        "left": [],
        "front": [],
        "right": [],
    }
    assert service.pixel_calls[0]["anchor_role"] == "affordance"
    assert service.pixel_calls[0]["target_pixel"] == [320, 431]


def test_navigation_skill_public_schema_exposes_point_object_and_instruction_following():
    schema = SlowBrainTaskSpecification.model_json_schema()

    assert schema["properties"]["task_type"]["enum"] == [
        "point_goal",
        "object_goal",
        "instruction_following",
    ]


class _StubOmniCapability:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def run_guided(
        self,
        request,
        observation,
        context,
        *,
        initial_guidance,
        refresh_guidance,
        on_guidance_accepted,
    ):
        del observation, context, refresh_guidance
        self.commands.append(initial_guidance.instruction)
        await on_guidance_accepted(initial_guidance)
        result = SkillResult(
            trace=request.trace,
            skill_name="NavigationSkill",
            status=SkillStatus.SUCCESS,
            summary="Slow guidance selected the terminal target.",
            outputs={"final_observation_id": "obs-current"},
            stop_reason=initial_guidance.terminal_reason,
        )
        return GuidedOmniNavOutcome(result=result, guidance=initial_guidance)


def test_instruction_following_routes_to_omninav_inside_navigation_skill(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    payload = _delegation()
    payload["task_specification"] = {
        "task_type": "instruction_following",
        "goal": "Exit the room, turn left, and continue to the robot.",
    }
    payload["planner_output"]["destination_query"] = "robot"
    payload["planner_output"]["object_goal"] = "robot"
    payload["planner_output"]["node_selection"] = None
    payload["planner_output"]["subtasks"] = [
        {
            "stage": "reaching",
            "skill_name": "NavigationSkill",
            "task_type": "instruction_following",
            "instruction": "Exit the room, turn left, and continue to the robot.",
            "target": "robot",
            "topology_node_id": None,
            "success_condition": "The instructed route has been followed and the robot is reached.",
        }
    ]
    omni = _StubOmniCapability()
    skill = AgentOSHierarchicalNavigationSkill(
        _Provider(_instruction_slow_semantic_output()),
        _Service(observation),
        N1NavigationSkillSession(global_goal=payload["task_specification"]["goal"]),
        visual_grounding_skill=_default_grounding_skill(),
        omninav_skill=omni,
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name="NavigationSkill",
        command=payload["task_specification"]["goal"],
        inputs={
            "delegation": payload,
            "source_observation_id": observation.trace.observation_id,
        },
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(InMemoryEventBus(), CancellationToken()),
        )
    )

    assert omni.commands == ["Hold for precise approach to the visible robot."]
    assert result.outputs["navigation_capability"] == "slow_affordance_omni_then_object_goal"
    assert result.outputs["anchor_role"] == "target"
    assert result.outputs["semantic_completion"]["satisfied"] is True


def test_topology_region_change_returns_control_after_anchor_execution():
    output = NavigationSlowBrainOutput.model_validate(_slow_pixel_output())
    topology = SlowBrainTopologyContext(
        map_id="test-floor",
        revision=1,
        current_topology_node_id="backbone-001",
    )

    reason = AgentOSHierarchicalNavigationSkill._return_control_reason(
        "object_goal",
        output,
        {
            "anchor_role": "affordance",
            "execution_telemetry": _execution_telemetry(
                reached=True, progress_m=1.0
            ),
            "semantic_route_plan": {
                "regional_guidance": {
                    "current_topology_node_id": "backbone-002",
                    "target_region_reached": False,
                }
            },
        },
        topology_context=topology,
    )

    assert reason == "topology_region_changed:backbone-001->backbone-002"


def test_reached_affordance_chunk_returns_to_agentos_for_verification(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    provider = _Provider(
        [
            _slow_semantic_output(target=False),
            _slow_semantic_output(target=True),
        ]
    )
    service = _Service(observation)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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

    assert len(_business_schemas(provider)) == 4
    assert provider.view_binding_calls == 3
    assert len(service.pixel_calls) == 1
    assert result.outputs["local_slow_cycles"] == 1
    assert result.outputs["return_control_reason"] == (
        "affordance_execution_chunk_reached"
    )
    assert result.outputs["slow_brain_output"]["target_pixel"] == {
        "left": [],
        "front": [],
        "right": [],
    }
    assert session.decision_count == 1


def test_each_fast_execution_uses_one_fresh_serial_slow_decision(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    provider = _DelayedProvider(
        [
            _slow_semantic_output(target=False),
            _slow_semantic_output(target=True),
        ]
    )
    service = _ChunkedService(observation)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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

    scheduler = result.outputs["slow_fast_scheduler"]
    assert scheduler["mode"] == "sequential"
    assert scheduler["policy"] == "one_observation_one_decision_one_execution"
    assert [event["event"] for event in scheduler["events"]] == [
        "slow_applied",
        "slow_applied",
        "control_returned",
    ]
    assert scheduler["events"][-1]["reason"] == "target_pixel_reached"
    assert len(service.pixel_calls) == 2
    assert service.pixel_calls[0]["target_pixel"] != service.pixel_calls[1]["target_pixel"]
    assert all("max_control_steps" not in call for call in service.pixel_calls)
    assert result.outputs["slow_brain_grounding"]["target_grounding"]["label"] == "bed"


def test_depth_limited_target_is_reobserved_before_object_goal_completion(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    provider = _Provider(
        [
            _slow_semantic_output(target=True),
            _slow_semantic_output(target=True),
        ]
    )
    service = _DepthLimitedTargetService(observation)
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        N1NavigationSkillSession(global_goal="find the brown cabinet"),
        visual_grounding_skill=_default_grounding_skill(),
    )
    delegation = _delegation()
    delegation["planner_output"]["anchor_max_depth_m"] = 1.5
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find and approach the brown cabinet.",
        inputs={
            "delegation": delegation,
            "source_observation_id": observation.trace.observation_id,
        },
        timeout_s=30.0,
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(InMemoryEventBus(), CancellationToken()),
        )
    )

    completion = result.outputs["semantic_completion"]
    assert len(service.pixel_calls) == 2
    assert {call["anchor_max_depth_m"] for call in service.pixel_calls} == {5.0}
    assert result.outputs["local_slow_cycles"] == 2
    assert result.outputs["return_control_reason"] == "target_pixel_reached"
    assert completion["target_projection_limited"] is False
    assert completion["target_proxy_reached"] is False
    assert completion["target_reached"] is True
    assert completion["satisfied"] is True


def test_slow_latency_does_not_enable_cached_async_execution(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    provider = _SlowTimeoutProvider(
        [
            _slow_semantic_output(target=False),
            _slow_semantic_output(target=True),
        ]
    )
    service = _ChunkedService(observation)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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

    scheduler = result.outputs["slow_fast_scheduler"]
    assert scheduler["mode"] == "sequential"
    assert [event["event"] for event in scheduler["events"]] == [
        "slow_applied",
        "slow_applied",
        "control_returned",
    ]
    assert scheduler["events"][-1]["reason"] == "target_pixel_reached"
    assert all("max_control_steps" not in call for call in service.pixel_calls)


def test_slow_topology_context_keeps_bounded_semantic_map_without_node_binding(
    make_trace,
    tmp_path,
):
    observation = _observation(make_trace, tmp_path)
    handoff = _Service(observation).prepare_agentic_plan({})
    handoff["semantic_route_plan"] = {}
    handoff["slow_context"]["topology_context"]["topo_memory"][
        "current_topology_node_id"
    ] = None
    delegation_payload = _delegation()
    delegation_payload["planner_output"]["node_selection"] = None
    delegation_payload["planner_output"]["subtasks"][0]["topology_node_id"] = None
    context = _slow_topology_context(
        handoff,
        AgentOSSkillDelegation.model_validate(delegation_payload),
    )

    assert context is not None
    assert context.current_topology_node_id is None
    assert context.target_topology_node_id is None
    assert [node.topology_node_id for node in context.nodes] == [
        "backbone-001",
        "backbone-002",
        "backbone-003",
    ]
    assert [(edge.source, edge.target) for edge in context.edges] == [
        ("backbone-001", "backbone-002"),
        ("backbone-002", "backbone-003"),
    ]


def test_slow_brain_input_contains_all_four_abot_n1_categories(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    history = tmp_path / "history-front.jpg"
    Image.new("RGB", (640, 480), color=(24, 24, 24)).save(history)
    current_output = _slow_semantic_output()
    current_output["cot"] = {
        "target_localization_and_key_context": (
            "The current front view independently confirms the bed in the bedroom."
        ),
        "path_planning": "Use the current clear floor evidence to approach the bed.",
    }
    provider = _Provider(current_output)
    service = _Service(observation)
    session = N1NavigationSkillSession(
        global_goal="go to bedroom to find a bed",
        handoff={
            "selected_keyframes": [
                {
                    "observation_id": "obs-history",
                    "front_image_ref": str(history),
                }
            ]
        },
    )
    session.bind_task(
        SlowBrainTaskSpecification(
            task_type="object_goal",
            goal="go to bedroom to find a bed",
        )
    )
    session.record_slow_output(
        NavigationSlowBrainOutput.model_validate(_slow_pixel_output()),
        source_observation_id="obs-previous",
    )
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        context_summary=(
            "Retrieved evidence: bed was visually grounded at backbone-003 on map test-floor."
        ),
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
            "replace_active_plan": False,
        },
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

    slow_input = result.outputs["slow_brain_input"]
    assert set(slow_input) == {
        "reference_memory",
        "reference_observation",
        "current_view_audit",
        "task_specification",
        "previous_decision",
        "topology_context",
    }
    assert slow_input["reference_memory"] == [
        {
            "slot": 0,
            "front_image_ref": str(history),
            "observation_id": "obs-history",
        }
    ]
    assert slow_input["reference_observation"]["observation_id"] == "obs-current"
    assert [
        slow_input["current_view_audit"][view]["view"]
        for view in ("left", "front", "right")
    ] == ["left", "front", "right"]
    assert slow_input["task_specification"] == {
        "task_type": "object_goal",
        "goal": "go to bedroom to find a bed",
    }
    assert slow_input["previous_decision"] == _slow_pixel_output()
    topology = slow_input["topology_context"]
    assert topology["current_topology_node_id"] == "backbone-001"
    assert topology["target_topology_node_id"] == "backbone-003"
    assert topology["next_topology_node_id"] == "backbone-002"
    assert [node["topology_node_id"] for node in topology["nodes"]] == [
        "backbone-001",
        "backbone-003",
        "backbone-002",
    ]
    assert topology["edges"] == [
        {
            "source": "backbone-001",
            "target": "backbone-002",
            "distance_m": 2.0,
            "transition": "office doorway",
        },
        {
            "source": "backbone-002",
            "target": "backbone-003",
            "distance_m": 3.0,
            "transition": "connected corridor",
        },
    ]
    labels = [
        item.get("text")
        for item in provider.delegation_messages[0][1].content
        if item.get("type") == "text"
    ]
    assert (
        "HISTORICAL_REFERENCE_MEMORY_1_OBSERVATION_obs-history_FRONT_"
        "NO_CURRENT_COORDINATES"
    ) in labels
    assert all(not label.startswith("CURRENT_OBSERVATION_") for label in labels)
    prompt = provider.delegation_messages[0][1].content[0]["text"]
    assert '"current_view_audit"' in prompt
    assert "previous_decision was produced from observation obs-previous" in prompt
    assert "current observation is obs-current" in prompt
    assert "Current visual evidence overrides previous_decision" in prompt
    assert "CURRENT_VIEW_AUDIT: LEFT=<visible evidence>" in prompt
    assert "Never reinterpret image-local left, center, and right as camera identities" in prompt
    assert "Stable AgentOS global task" in prompt
    assert "Current delegated Navigation Skill subtask" in prompt
    assert "bed was visually grounded at backbone-003" in prompt
    assert "Prefer a distinctive directional anchor" in prompt
    assert "rather than defaulting to generic phrases such as 'clear floor'" in prompt
    assert "Never output a bbox, pixel, depth, world coordinate" in prompt


def test_slow_brain_accepts_stable_cot_when_current_semantics_are_unchanged(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    previous = NavigationSlowBrainOutput.model_validate(_slow_pixel_output())
    stable = _slow_semantic_output()
    provider = _Provider(stable)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    session.bind_task(
        SlowBrainTaskSpecification(
            task_type="object_goal",
            goal="go to bedroom to find a bed",
        )
    )
    session.record_slow_output(previous, source_observation_id="obs-previous")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        _Service(observation),
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
            "replace_active_plan": False,
        },
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

    assert len(_business_schemas(provider)) == 4
    assert provider.view_binding_calls == 3
    assert result.outputs["slow_brain_output"]["cot"] == stable["cot"]


def test_slow_grounding_rejects_zero_area_bbox_placeholder():
    try:
        SlowBrainVisualGrounding.model_validate(
            {
                "label": "placeholder",
                "view": "front",
                "bbox_normalized_1000": [500, 500, 500, 500],
                "confidence": 0.9,
                "evidence_summary": "Zero-area placeholder should be rejected.",
            }
        )
    except ValueError as exc:
        assert "positive width and height" in str(exc)
    else:
        raise AssertionError("zero-area bbox placeholders must be rejected")


def test_grounding_protocol_compacts_long_evidence_without_losing_geometry(
    make_trace,
):
    semantic_decision = ObjectGoalSlowBrainSemanticOutput.model_validate(
        _slow_semantic_output()
    )
    long_evidence = "  current   frame evidence " * 30
    result = SkillResult(
        trace=make_trace(observation_id="obs-current"),
        skill_name="VisualGroundingSkill",
        status=SkillStatus.SUCCESS,
        summary="Grounded target and affordance.",
        outputs={
            "available": True,
            "observation_id": "obs-current",
            "provider": "test-grounder",
            "results": [
                {
                    "role": "affordance",
                    "query": "bedroom doorway",
                    "detections": [
                        {
                            "label": "A verbose doorway description that should never replace the bounded Slow query label in the navigation protocol",
                            "view": "right",
                            "bbox_normalized_1000": [100, 200, 500, 900],
                            "confidence": 0.93,
                            "evidence_summary": long_evidence,
                        }
                    ],
                }
            ],
        },
    )

    grounded = AgentOSHierarchicalNavigationSkill._grounding_from_batch(
        semantic_decision,
        result,
    ).affordance_grounding

    assert len(grounded.evidence_summary) == 240
    assert "  " not in grounded.evidence_summary
    assert grounded.label == "bedroom doorway"
    assert grounded.view == "right"
    assert grounded.bbox_normalized_1000 == [100, 200, 500, 900]
    assert grounded.confidence == 0.93


def test_slow_grounding_rejects_invalid_view_name():
    try:
        SlowBrainVisualGrounding.model_validate(
            {
                "label": "doorway",
                "view": "center",
                "bbox_normalized_1000": [400, 400, 600, 800],
                "confidence": 0.9,
            }
        )
    except ValueError as exc:
        assert "left" in str(exc) and "front" in str(exc) and "right" in str(exc)
    else:
        raise AssertionError("grounding view must use the synchronized tri-view names")


def test_slow_grounding_schema_only_allows_empty_or_four_coordinate_bbox():
    schema = GroundedNavigationSlowBrainOutput.model_json_schema()
    grounding_ref = schema["properties"]["affordance_grounding"]["$ref"].split("/")[-1]
    bbox_schema = schema["$defs"][grounding_ref]["properties"][
        "bbox_normalized_1000"
    ]

    assert bbox_schema["anyOf"] == [
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "maxItems": 0,
        },
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
    ]


def test_n1_natural_language_cot_fields_allow_complete_bounded_reasoning():
    instruction_schema = InstructionFollowingCoT.model_json_schema()["properties"]
    object_schema = ObjectGoalCoT.model_json_schema()["properties"]

    assert instruction_schema["finished_sub_instruction"]["maxLength"] == 512
    assert instruction_schema["ongoing_sub_instruction"]["maxLength"] == 512
    assert object_schema["target_localization_and_key_context"]["maxLength"] == 512
    assert object_schema["path_planning"]["maxLength"] == 512


def test_main_agent_rationale_allows_one_complete_bounded_explanation():
    schema = AgentOSSkillDelegation.model_json_schema()

    assert schema["properties"]["rationale_summary"]["maxLength"] == 512


def test_slow_brain_schema_has_cot_affordance_and_target_pixel_interface():
    schema = NavigationSlowBrainOutput.model_json_schema()
    assert set(schema["properties"]) == {
        "cot",
        "affordance_pixel",
        "target_pixel",
    }


def test_slow_semantic_schema_has_queries_but_no_localization_coordinates():
    schema = ObjectGoalSlowBrainSemanticOutput.model_json_schema()

    assert set(schema["properties"]) == {
        "cot",
        "affordance_query",
        "target_query",
    }
    assert "bbox" not in str(schema).lower()
    assert "pixel" not in str(schema).lower()


def test_changed_task_spec_resets_previous_n1_decision_and_reference_memory(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    history = tmp_path / "previous-task.jpg"
    Image.new("RGB", (640, 480), color=(24, 24, 24)).save(history)
    session = N1NavigationSkillSession(
        global_goal="go to bedroom to find a bed",
        handoff={
            "selected_keyframes": [
                {"observation_id": "obs-old", "front_image_ref": str(history)}
            ]
        },
    )
    session.bind_task(
        SlowBrainTaskSpecification(
            task_type="object_goal",
            goal="Find the bed inside the bedroom region.",
        )
    )
    session.record_slow_output(
        NavigationSlowBrainOutput.model_validate(_slow_pixel_output()),
        source_observation_id="obs-old",
    )
    skill = AgentOSHierarchicalNavigationSkill(_Provider(_slow_semantic_output()), _Service(observation), session)
    delegation = _delegation()
    delegation["task_specification"] = {
        "task_type": "point_goal",
        "goal": "Go to the saved topology point backbone-002.",
    }
    delegation["planner_output"]["subtasks"][0].update(
        {
            "stage": "reaching",
            "task_type": "point_goal",
            "instruction": "Go to the saved topology point backbone-002.",
            "target": "backbone-002",
            "topology_node_id": "backbone-002",
        }
    )

    slow_input = skill.slow_brain._build_input(
        observation,
        AgentOSSkillDelegation.model_validate(delegation),
    )

    assert slow_input.previous_decision is None
    assert session.previous_decision_observation_id is None
    assert slow_input.reference_memory == []
    assert slow_input.task_specification.task_type == "point_goal"


def test_object_goal_does_not_hide_a_topology_point_goal_inside_skill_runner(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    provider = _Provider(_slow_semantic_output())
    service = _NodePointGoalService(observation)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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
    assert "PointGoal node=" not in result.summary
    assert service.location_calls == []
    assert len(service.pixel_calls) == 1
    assert len(_business_schemas(provider)) == 4
    assert provider.view_binding_calls == 3
    assert session.decision_count == 1


def test_explicit_point_goal_uses_topology_navigation_without_slow_grounding(
    make_trace, tmp_path
):
    observation = _observation(make_trace, tmp_path)
    provider = _Provider(_slow_semantic_output())
    service = _NodePointGoalService(observation)
    session = N1NavigationSkillSession(global_goal="go to bedroom to find a bed")
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        session,
        visual_grounding_skill=_default_grounding_skill(),
    )
    delegation = _delegation()
    delegation["task_specification"] = {
        "task_type": "point_goal",
        "goal": "Navigate to the mapped bedroom region at backbone-003.",
    }
    delegation["planner_output"]["subtasks"][0].update(
        {
            "stage": "approaching",
            "task_type": "point_goal",
            "instruction": "Navigate to the mapped bedroom region at backbone-003.",
            "target": "backbone-003",
            "topology_node_id": "backbone-003",
            "success_condition": "The robot reaches the mapped bedroom region.",
        }
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Navigate to the mapped bedroom region at backbone-003.",
        success_conditions=["The robot reaches the mapped bedroom region."],
        inputs={
            "delegation": delegation,
            "source_observation_id": observation.trace.observation_id,
        },
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
    assert "PointGoal node=backbone-002" in result.summary
    assert result.outputs["distance_m"] == 2.5
    assert result.outputs["execution_mode"] == "topology_node_point_goal"
    assert service.location_calls[0]["topology_node_id"] == "backbone-002"
    assert service.pixel_calls == []
    assert provider.schemas == []
    assert session.decision_count == 0


def test_no_projectable_pixel_returns_recoverable_skill_failure(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    provider = _Provider(_slow_semantic_output(target=False))
    service = _Service(observation)
    skill = AgentOSHierarchicalNavigationSkill(
        provider,
        service,
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
        visual_grounding_skill=_default_grounding_skill(
            affordance=False,
            target=False,
        ),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        success_conditions=["A bed is clearly visible in the current observation."],
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
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

    assert result.status.value == "retryable_failure"
    assert result.stop_reason == "no_visible_pixel_goal"
    assert service.pixel_calls == []


def test_invalid_slow_protocol_returns_recoverable_skill_failure(make_trace, tmp_path):
    observation = _observation(make_trace, tmp_path)
    skill = AgentOSHierarchicalNavigationSkill(
        _Provider({"invalid": "slow output"}),
        _Service(observation),
        N1NavigationSkillSession(global_goal="go to bedroom to find a bed"),
        visual_grounding_skill=_default_grounding_skill(),
    )
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="Find the bed inside the bedroom region.",
        inputs={
            "delegation": _delegation(),
            "source_observation_id": observation.trace.observation_id,
        },
        timeout_s=30.0,
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(InMemoryEventBus(), CancellationToken()),
        )
    )

    assert result.status.value == "retryable_failure"
    assert result.stop_reason == "slow_decision_invalid"
    assert "Slow/Grounding decision needs replanning" in result.summary
