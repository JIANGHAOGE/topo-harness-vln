from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentos.core.trace import is_current_trace
from agentos.schemas.enums import SkillStatus
from agentos.schemas.hierarchical import HierarchicalNavigateInput
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.task_reasoning import (
    AdvanceNavigationInput,
    AgenticPlannerOutput,
    FinishVerifierAssessment,
    FinishVerifierDecision,
    SelectedNavigationAction,
    RegionSearchPlan,
    SlowBrainAffordanceGroundingQuery,
    SubtaskReflection,
)
from agentos.schemas.trace import TraceContext


def test_trace_schema_requires_async_identity_fields(make_trace):
    schema = TraceContext.model_json_schema()
    required = set(schema["required"])
    assert {
        "episode_id",
        "task_id",
        "plan_revision",
        "skill_run_id",
        "observation_id",
        "frame_timestamp",
        "pose_timestamp",
        "deadline",
    } <= required


def test_core_messages_validate_and_forbid_extra_fields(make_trace):
    trace = make_trace()
    task = TaskRequest(trace=trace, instruction="Go to the kitchen.")
    observation = Observation(
        trace=trace,
        rgb_refs={"front": "mock://front.jpg"},
        pose=Pose2D(x=1.0, y=2.0, yaw=0.1, source="mock", frame="robot_odom_xy"),
    )
    skill_request = SkillRequest(trace=trace, skill_name="NavigationSkill", command=task.instruction)
    skill_result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="done",
        evidence=[Evidence(kind="mock", summary="ok")],
    )
    assert observation.pose.frame == "robot_odom_xy"
    assert skill_request.trace.episode_id == trace.episode_id
    assert skill_result.compressed_history is None

    with pytest.raises(ValidationError):
        TaskRequest(trace=trace, instruction="Go.", unexpected=True)


def test_pixel_anchor_rejects_topology_and_location_ids():
    base = {
        "view": "front",
        "semantic_map": {
            "scene_summary": "Open traversable corridor.",
            "traversable_directions": ["front"],
        },
    }
    for anchor in ("backbone-003", "location-025"):
        with pytest.raises(ValidationError, match="not a topology/location ID"):
            HierarchicalNavigateInput(anchor=anchor, **base)

    valid = HierarchicalNavigateInput(
        anchor="traversable route affordance",
        anchor_role="affordance",
        **base,
    )
    assert valid.anchor == "traversable route affordance"


def test_slow_affordance_rejects_camera_placeholder_labels():
    with pytest.raises(ValidationError, match="navigable semantic waypoint"):
        SlowBrainAffordanceGroundingQuery(
            label="right_camera",
            preferred_view="right",
            visual_description="The current right camera image.",
            spatial_relation="Inspect the right side.",
        )

    doorway = SlowBrainAffordanceGroundingQuery(
        label="open wooden doorway",
        preferred_view="left",
        visual_description="An open wooden doorway with traversable floor beneath it.",
        spatial_relation="The doorway leads into the next room.",
    )
    assert doorway.label == "open wooden doorway"


def test_trace_guard_drops_mismatched_or_expired_results(make_trace):
    active = make_trace(skill_run_id="active")
    same = active.model_copy()
    different_skill = active.for_skill(skill_run_id="old-skill")
    expired = make_trace(skill_run_id="active", deadline_offset_s=-1.0)

    assert is_current_trace(active, same)
    assert not is_current_trace(active, different_skill)
    assert not is_current_trace(active, expired)


def test_region_search_plan_requires_unique_candidates_and_active_membership():
    candidate = {
        "topology_node_id": "backbone-003",
        "confidence": 0.9,
        "place_evidence": "The node place type is bedroom.",
        "object_evidence": "A bed is associated with this region.",
        "relation_reason": "This is the strongest task-compatible region.",
    }
    plan = RegionSearchPlan(
        candidates=[candidate],
        active_topology_node_id="backbone-003",
        update_reason="Start with the strongest joint place-object hypothesis.",
    )
    assert plan.active_topology_node_id == "backbone-003"

    with pytest.raises(ValidationError, match="must be unique"):
        RegionSearchPlan(
            candidates=[candidate, candidate],
            active_topology_node_id="backbone-003",
            update_reason="Invalid duplicate candidates.",
        )

    with pytest.raises(ValidationError, match="must reference"):
        RegionSearchPlan(
            candidates=[candidate],
            active_topology_node_id="backbone-004",
            update_reason="Invalid active candidate.",
        )

    second_candidate = {
        **candidate,
        "topology_node_id": "backbone-004",
        "confidence": 0.8,
    }
    advanced = RegionSearchPlan(
        candidates=[candidate, second_candidate],
        active_topology_node_id="backbone-004",
        update_reason="Advance through the stable candidate queue after checking the first region.",
    )
    assert advanced.active_topology_node_id == "backbone-004"
    assert advanced.candidates[0].topology_node_id == "backbone-003"


def test_abot_agentic_planner_enforces_typed_subtasks_and_preserves_instruction_order():
    plan = AgenticPlannerOutput(
        global_instruction="go to the bedroom and find the bed",
        destination_query="bedroom",
        object_goal="bed",
        subtasks=[
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Approach the bedroom region.",
                "target": "bedroom",
                "topology_node_id": "backbone-003",
                "success_condition": "Reach the bedroom region.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Locate the bed inside the bedroom.",
                "target": "bed",
                "success_condition": "Current visual evidence confirms the bed.",
            },
        ],
    )

    assert [subtask.stage for subtask in plan.subtasks] == [
        "approaching",
        "reaching",
    ]

    descriptive_stage = AgenticPlannerOutput(
        global_instruction="find the bed",
        destination_query="bedroom",
        object_goal="bed",
        subtasks=[
            {
                "stage": "approaching",
                "task_type": "object_goal",
                "instruction": "Search for the object while approaching its likely region.",
                "target": "bedroom",
                "success_condition": "Reach the bedroom and obtain visual evidence.",
            }
        ],
    )
    assert descriptive_stage.subtasks[0].task_type == "object_goal"

    long_instruction = AgenticPlannerOutput(
        global_instruction="go out of the room, turn left, and find the robot",
        destination_query="area reached after leaving the room and turning left",
        object_goal="robot",
        subtasks=[
            {
                "stage": "interaction",
                "task_type": "object_goal",
                "instruction": "Exit the current room, then turn left and go straight.",
                "target": "ordered route instruction",
                "success_condition": "The ordered route segment is complete.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the robot after completing the route segment.",
                "target": "robot",
                "success_condition": "Current visual evidence confirms the robot.",
            },
        ],
    )

    assert [subtask.stage for subtask in long_instruction.subtasks] == [
        "interaction",
        "reaching",
    ]


def test_finish_verifier_requires_structured_evidence_for_authorized_done():
    with pytest.raises(ValidationError, match="observed_evidence"):
        FinishVerifierAssessment(
            evidence_satisfies_task=True,
            terminal_failure_supported=False,
            observed_evidence=[],
        )

    with pytest.raises(ValidationError, match="observed_evidence"):
        FinishVerifierDecision(
            requested_outcome="done",
            authorized=True,
            recommended_action="finish",
            summary="The target was found.",
            observed_evidence=[],
        )

    rejected = FinishVerifierDecision(
        requested_outcome="done",
        authorized=False,
        recommended_action="replan",
        summary="The target is not confirmed.",
        observed_evidence=["Current views do not show the requested object."],
    )
    assert rejected.authorized is False


def test_abot_n1_cot_schema_is_task_specific():
    payload = {
        "observation_id": "obs-001",
        "global_goal": "follow the instruction",
        "previous_assessment": {
            "execution_status": "not_started",
            "task_effect": "unknown",
            "evidence": "Initial observation.",
        },
        "route_hypothesis": {
            "statement": "Continue through the doorway.",
            "confidence": 0.8,
        },
        "candidates": [
            {
                "action_type": "visible_anchor",
                "target": "doorway",
                "task_relevance": 0.9,
                "information_gain": 0.7,
                "revisit_penalty": 0.0,
            }
        ],
        "action": {
            "action_type": "visible_anchor",
            "target": "doorway",
            "view": "front",
            "expected_evidence": "The next room becomes visible.",
            "success_condition": "Reach the doorway.",
            "fallback": "Reassess.",
        },
        "semantic_map": {
            "scene_summary": "A corridor with an open doorway.",
            "left": [],
            "front": ["doorway"],
            "right": [],
        },
    }
    object_goal = AdvanceNavigationInput.model_validate(
        {
            **payload,
            "navigation_task_type": "object_goal",
            "cot": {
                "target_localization_and_key_context": (
                    "The doorway is the next semantic route target."
                ),
                "path_planning": "Pass through the open doorway.",
            },
        }
    )
    assert object_goal.cot.path_planning == "Pass through the open doorway."

    point_goal = AdvanceNavigationInput.model_validate(
        {**payload, "navigation_task_type": "point_goal", "cot": None}
    )
    assert point_goal.cot is None

    with pytest.raises(ValidationError, match="does not emit linguistic CoT"):
        AdvanceNavigationInput.model_validate(
            {
                **payload,
                "navigation_task_type": "point_goal",
                "cot": {
                    "finished_sub_instruction": "No previous step.",
                    "ongoing_sub_instruction": "Approach the map point.",
                },
            }
        )

    with pytest.raises(ValidationError, match="object_goal requires"):
        AdvanceNavigationInput.model_validate(
            {
                **payload,
                "navigation_task_type": "object_goal",
                "cot": {
                    "finished_sub_instruction": "Reached the corridor end.",
                    "ongoing_sub_instruction": "Pass through the open doorway.",
                },
            }
        )


def test_subtask_reflection_rejects_inconsistent_outcome():
    with pytest.raises(ValidationError, match="outcome=failed requires should_replan=true"):
        SubtaskReflection(
            subtask_id="subtask-001",
            completed=False,
            outcome="failed",
            feedback="The route is blocked.",
            should_replan=False,
        )
    with pytest.raises(ValidationError, match="completed must agree"):
        SubtaskReflection(
            subtask_id="subtask-001",
            completed=True,
            outcome="partial",
            feedback="The intermediate waypoint was reached.",
            should_replan=False,
        )


def test_navigation_action_types_reject_fields_from_other_execution_modes():
    common = {
        "target": "doorway",
        "expected_evidence": "The route remains visible.",
        "success_condition": "The action completes.",
        "fallback": "Reassess.",
    }
    with pytest.raises(ValidationError, match="known-location target ID"):
        SelectedNavigationAction(
            action_type="visible_anchor",
            view="front",
            topology_node_id="backbone-002",
            **common,
        )
    with pytest.raises(ValidationError, match="visually groundable semantic phrase"):
        SelectedNavigationAction(
            action_type="visible_anchor",
            target="backbone-004",
            view="front",
            expected_evidence="The route remains visible.",
            success_condition="The action completes.",
            fallback="Reassess.",
        )
    with pytest.raises(ValidationError, match="camera or viewpoint fields"):
        SelectedNavigationAction(
            action_type="known_location",
            topology_node_id="backbone-002",
            view="front",
            **common,
        )
