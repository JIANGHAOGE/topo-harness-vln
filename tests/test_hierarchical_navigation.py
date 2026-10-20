from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from agentos.api.hierarchical_navigation_server import (
    HierarchicalNavigationService,
    _compact_error_message,
)
from agentos.core.events import InMemoryEventBus
from agentos.mapping import StaticNavigabilityMapProvider
from agentos.memory.semantic_map import SemanticMapStore
from agentos.schemas.abot import ABotTargetEvidence
from agentos.schemas.hierarchical import (
    AnchorPixel,
    HierarchicalNavigateInput,
    SemanticMapUpdate,
)
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.semantic_map import EditableSemanticMap, NodeSemantics
from agentos.schemas.skill import SkillRequest
from agentos.schemas.task_reasoning import (
    AdvanceNavigationInput,
    AgenticPlannerOutput,
    SelectedNavigationAction,
    SemanticNodeSelection,
)
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.navigation import (
    AnchorGenerationSkill,
    AnchorGroundingSkill,
    LocalizationMapSkill,
    PathExecutionSkill,
    PathPlanningSkill,
    PixelToPointSkill,
    PointNavigationSkill,
    ViewpointAdjustmentSkill,
)
from agentos.skills.registry import SkillRegistry
from agentos.topology import (
    BackboneEdge,
    BackboneNode,
    BackboneTopology,
    BackboneTopologyCache,
    NavigabilityGrid,
)


class FakeProvider:
    def __init__(
        self,
        calls: list[dict],
        *,
        confidence: float = 0.91,
        view: str = "front",
        anchor_visible: bool = True,
    ) -> None:
        self.calls = calls
        self.confidence = confidence
        self.view = view
        self.anchor_visible = anchor_visible

    async def complete(self, messages, *, response_schema=None):
        self.calls.append({"messages": messages, "schema": response_schema})
        image_labels = [
            item.get("text", "")
            for item in messages[-1].content
            if item.get("type") == "text" and "IMAGE/PANEL" in item.get("text", "")
        ]
        current_view = next(
            view
            for view in ("left", "front", "right")
            if image_labels and view.upper() in image_labels[-1]
        )
        visible = self.anchor_visible and current_view == self.view
        payload = {
            "anchor": "doorway",
            "panel_index": {"left": 0, "front": 1, "right": 2}[current_view],
            "x_min": 450,
            "y_min": 620,
            "x_max": 550,
            "y_max": 720,
            "anchor_visible": visible,
            "confidence": self.confidence if visible else 0.0,
            "evidence_summary": (
                "Floor contact just inside the visible doorway."
                if visible
                else "The requested anchor is absent from this camera."
            ),
        }
        return LLMResponse(model="fake-qwen", content="{}", raw={}, parsed_json=payload)

    async def close(self):
        return None


class FakeSemanticNodeProvider:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls

    async def complete(self, messages, *, response_schema=None):
        self.calls.append({"messages": messages, "schema": response_schema})
        return LLMResponse(
            model="fake-semantic-model",
            content="{}",
            raw={},
            parsed_json={
                "selected_topology_node_id": "backbone-002",
                "confidence": 0.96,
                "matched_concepts": ["fridge", "refrigerator"],
                "selection_reason": "Explicit refrigerator evidence best satisfies the object goal.",
            },
        )

    async def close(self):
        return None


def _view_audit_response(messages, response_schema):
    if (response_schema or {}).get("title") != "SlowBrainSingleViewAudit":
        return None
    system = str(messages[0].content)
    view = next(
        item for item in ("left", "front", "right") if f"robot-{item.upper()}" in system
    )
    return LLMResponse(
        model="fake-view-audit",
        content="{}",
        raw={},
        parsed_json={
            "view": view,
            "visible_objects": ["target"],
            "navigable_semantic_anchors": ["doorway"],
            "target_relevant_evidence": [f"Current {view} evidence."],
            "scene_summary": f"Current {view} camera scene.",
        },
    )


def semantic_map(scene_summary="white corridor", *, left=None, front=None, right=None):
    return {
        "scene_summary": scene_summary,
        "left": list(["white cabinets"] if left is None else left),
        "front": list(["open corridor"] if front is None else front),
        "right": list(["robot equipment"] if right is None else right),
    }


def viewpoint_decision(
    observation_id,
    *,
    task_type="point_goal",
    reflection=None,
    planner_output=None,
):
    cot = None
    if task_type == "object_goal":
        cot = {
            "target_localization_and_key_context": "Search the active target region.",
            "path_planning": "Adjust the viewpoint before selecting the next anchor.",
        }
    elif task_type == "instruction_following":
        cot = {
            "finished_sub_instruction": "The previous macro action finished.",
            "ongoing_sub_instruction": "Continue the active navigation subtask.",
        }
    payload = {
        "observation_id": observation_id,
        "global_goal": "find the target",
        "navigation_task_type": task_type,
        "cot": cot,
        "self_reflection": reflection,
        "previous_assessment": {
            "execution_status": "completed" if reflection is not None else "not_started",
            "task_effect": "progress" if reflection is not None else "unknown",
            "evidence": "Current synchronized observation.",
        },
        "route_hypothesis": {
            "statement": "Inspect the connected route from the current observation.",
            "confidence": 0.8,
        },
        "candidates": [
            {
                "action_type": "viewpoint_adjustment",
                "target": "turn left",
                "task_relevance": 0.8,
                "information_gain": 0.8,
                "revisit_penalty": 0.0,
            }
        ],
        "action": {
            "action_type": "viewpoint_adjustment",
            "target": "turn left",
            "viewpoint_action": "turn_left",
            "expected_evidence": "A connected route becomes visible.",
            "success_condition": "The left-side view is refreshed.",
            "fallback": "Reassess the current views.",
        },
        "semantic_map": semantic_map(),
    }
    if planner_output is not None:
        payload["planner_output"] = planner_output
    return payload


class FakeHierarchicalAdapter:
    name = "habitat-gs"

    def __init__(self, tmp_path: Path, make_trace) -> None:
        self.sim = object()
        self.tmp_path = tmp_path
        self.make_trace = make_trace
        self.execution_calls = 0
        self.viewpoint_actions = []
        self.yaw = 0.0
        self.before_control_step = None

    async def observe(self, trace):
        refs = {}
        depth_refs = {}
        for view in ("left", "front", "right"):
            rgb = self.tmp_path / f"{view}.jpg"
            depth = self.tmp_path / f"depth_{view}.png"
            Image.new("RGB", (640, 480), color=(80, 90, 100)).save(rgb)
            depth.write_bytes(b"depth")
            refs[view] = str(rgb)
            depth_refs[view] = str(depth)
        topdown = self.tmp_path / "topdown.png"
        Image.new("RGB", (320, 480), color=(30, 35, 30)).save(topdown)
        return Observation(
            trace=trace,
            rgb_refs=refs,
            depth_refs=depth_refs,
            topdown_ref=str(topdown),
            pose=Pose2D(
                x=0.0,
                y=-self.execution_calls * 0.18,
                yaw=self.yaw,
                source="test",
                frame="habitat_world_xz",
            ),
            metadata={
                "camera": {"output_resolution": [640, 480]},
                "habitat_position_xyz": [0.0, 0.0, -self.execution_calls * 0.18],
                "metric_position_xyz": [0.0, 0.0, -self.execution_calls * 0.18],
                "map_id": "test-map",
                "topdown_map": {
                    "origin_world_xz_m": [-16.0, -24.0],
                    "meters_per_pixel": 0.1,
                    "pixel_axes": ["+world_x", "+world_z"],
                },
            },
        )

    async def observe_live(self, trace):
        return await self.observe(trace)

    async def pixel_to_world(self, **kwargs):
        assert kwargs["view"] == "front"
        assert kwargs["x_px"] == 320
        return {
            "view": "front",
            "pixel": [320, 360],
            "depth_pixel": [416, 360],
            "depth_m": 2.0,
            "camera_point_right_up_forward_m": [0.0, -0.4, 2.0],
            "robot_point_forward_left_m": [2.0, 0.0],
            "world_point_xyz_m": [0.0, 0.05, -2.0],
            "navigation_candidate_xyz_m": [0.0, 0.0, -2.0],
        }

    async def world_to_pixel(self, **kwargs):
        assert kwargs["world_point"][2] <= 0.0
        return {
            "view": "front",
            "pixel": [320, 360],
            "depth_m": 2.0,
            "sensed_depth_m": 2.5,
            "visible": True,
        }

    async def plan_path(self, world_point):
        assert world_point == [0.0, 0.0, -2.0]
        return {
            "requested_goal_xyz_m": world_point,
            "snapped_goal_xyz_m": world_point,
            "world_path_xyz_m": [[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]],
            "robot_path_forward_left_m": [[0.0, 0.0], [2.0, 0.0]],
            "geodesic_distance_m": 2.0,
            "waypoint_count": 2,
        }

    async def execute_world_path(self, world_path, **kwargs):
        raise AssertionError("hierarchical execution must use bounded Habitat navmesh actions")

    async def follow_navmesh_path_step(self, world_path, **kwargs):
        assert world_path[-1] == [0.0, 0.0, -2.0]
        if self.before_control_step is not None:
            self.before_control_step()
        self.execution_calls += 1
        reached = self.execution_calls >= 10
        return {
            "reached": reached,
            "remaining_distance_m": 0.2 if reached else 2.0 - self.execution_calls * 0.18,
            "stalled": False,
            "pose_after": {"x": 0.0, "y": -self.execution_calls * 0.18, "yaw": self.yaw},
            "control": {
                "action": "move_forward",
                "controller": "habitat_greedy_follower",
                "distance_m": 0.18,
                "collided": False,
            },
        }

    async def observe_keyframe(self, trace, run_id, frame_index):
        frame_dir = self.tmp_path / run_id / str(frame_index)
        frame_dir.mkdir(parents=True, exist_ok=True)
        refs = {}
        for view in ("left", "front", "right"):
            path = frame_dir / f"rgb_{view}.jpg"
            Image.new("RGB", (640, 480), color=(frame_index, 20, 30)).save(path)
            refs[view] = str(path)
        return Observation(
            trace=trace,
            rgb_refs=refs,
            pose=Pose2D(
                x=0.0,
                y=-frame_index * 0.18,
                yaw=self.yaw,
                source="test",
                frame="habitat_world_xz",
            ),
            metadata={
                "metric_position_xyz": [0.0, 0.0, -frame_index * 0.18],
                "map_id": "test-map",
                "pose_confidence": 1.0,
            },
        )

    def step(self, action):
        self.viewpoint_actions.append(action)
        pose_before = {"x": 0.0, "y": 0.0, "yaw": self.yaw}
        if action == "turn_left":
            self.yaw += 0.261799
        elif action == "turn_right":
            self.yaw -= 0.261799
        return {
            "action": action,
            "collided": False,
            "distance_m": 0.2 if action in {"move_forward", "move_backward"} else 0.0,
            "pose_before": pose_before,
            "pose_after": {"x": 0.0, "y": 0.0, "yaw": self.yaw},
        }

    def close(self):
        return None

    def start(self):
        return None


def test_begin_loads_frozen_backbone_without_adding_start_node(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    free = np.zeros((30, 40), dtype=bool)
    free[4:26, 5:35] = True
    provider = StaticNavigabilityMapProvider(
        grid=NavigabilityGrid(
            map_id="test-map",
            geometry_revision="test-map-v1",
            source="habitat_navmesh",
            frame="habitat_world_xz",
            resolution_m=0.2,
            origin_xy_m=(-4.0, -3.0),
            free_mask=free,
            navigation_plane="xz",
        ),
        cache_dir=tmp_path / "topology-cache",
    )
    service = HierarchicalNavigationService(
        adapter,
        SkillRegistry(),
        topology_provider=provider,
    )
    load_calls = []
    original_load_grid = provider.load_grid

    def counted_load_grid():
        load_calls.append(True)
        return original_load_grid()

    provider.load_grid = counted_load_grid

    first = service.begin_navigation()

    assert first["topology"]["status"] == "loaded"
    assert first["topology"]["frozen"] is True
    assert first["topology"]["cache_hit"] is False
    assert first["topology"]["node_count"] > 0
    assert first["route_graph"]["nodes"][0]["location_id"] == "location-001"
    assert not any(node["is_global_anchor"] for node in first["route_graph"]["nodes"])
    assert all(node["source"] == "map_backbone" for node in first["route_graph"]["nodes"])

    reset = service.reset()
    assert reset["topology"]["cache_hit"] is True
    assert reset["route_graph"]["nodes"][0]["location_id"] == "location-001"
    second = service.begin_navigation()

    assert second["topology"]["cache_hit"] is True
    assert second["route_graph"]["nodes"][0]["location_id"] == "location-001"
    assert len(load_calls) == 1
    assert not hasattr(adapter, "navigability_grid")
    assert len(list(provider.cache_dir.glob("test-map-*.json"))) == 1


def test_semantic_backbone_stays_frozen_across_reset_and_reload_request(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    free = np.zeros((30, 40), dtype=bool)
    free[4:26, 5:35] = True
    provider = StaticNavigabilityMapProvider(
        grid=NavigabilityGrid(
            map_id="test-map",
            geometry_revision="test-map-v1",
            source="habitat_navmesh",
            frame="habitat_world_xz",
            resolution_m=0.2,
            origin_xy_m=(-4.0, -3.0),
            free_mask=free,
            navigation_plane="xz",
        ),
        cache_dir=tmp_path / "topology-cache",
    )
    topology_cache = BackboneTopologyCache(provider.cache_dir)
    topology, _ = topology_cache.get_or_build(provider.load_grid())
    semantic_store = SemanticMapStore(tmp_path / "semantic-maps")
    target_node_id = topology.nodes[0].node_id
    semantic_store.save(
        EditableSemanticMap(
            map_id=topology.map_id,
            geometry_revision=topology.geometry_revision,
            topology_cache_key=topology.cache_key,
            nodes={
                target_node_id: NodeSemantics(
                    place_type="kitchen",
                    objects=["white humanoid robot"],
                    source="human",
                )
            },
        )
    )
    service = HierarchicalNavigationService(
        adapter,
        SkillRegistry(),
        topology_provider=provider,
        topology_cache=topology_cache,
        semantic_map_store=semantic_store,
    )

    initial = service.load_environment_map()
    stored_path = semantic_store.path_for(topology.map_id)
    edited = EditableSemanticMap.model_validate_json(stored_path.read_text(encoding="utf-8"))
    edited.nodes[target_node_id] = edited.nodes[target_node_id].model_copy(
        update={"place_type": "office"}
    )
    stored_path.write_text(edited.model_dump_json(indent=2), encoding="utf-8")

    reload_result = service.reload_semantic_map()
    reset_result = service.reset()
    reloaded_place = service.navigation_memory.get_topology_node(target_node_id)

    assert initial["frozen"] is True
    assert reload_result["reloaded"] is False
    assert reset_result["topology"]["frozen"] is True
    assert reloaded_place is not None
    assert reloaded_place.node_semantics is not None
    assert reloaded_place.node_semantics.place_type == "kitchen"
    assert not any(node["is_global_anchor"] for node in reset_result["route_graph"]["nodes"])


def test_hierarchical_pipeline_audits_each_camera_then_runs_geometry_and_control(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGroundingSkill(lambda: FakeProvider(calls)))
    skills.register(PixelToPointSkill(adapter))
    skills.register(PathPlanningSkill(adapter))
    skills.register(PathExecutionSkill(adapter))
    skills.register(AnchorGenerationSkill(lambda: FakeProvider(calls), adapter))
    skills.register(PointNavigationSkill(adapter))
    skills.register(ViewpointAdjustmentSkill(adapter))
    skills.register(LocalizationMapSkill())
    service = HierarchicalNavigationService(adapter, skills)

    result = service.navigate_to_anchor(
        {
            "anchor": "the visible doorway",
            "view": "front",
            "semantic_map": semantic_map(),
            "max_control_steps": 24,
        }
    )

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert [stage["skill"] for stage in result["stages"]] == [
        "AnchorGenerationSkill",
        "PointNavigationSkill",
    ]
    assert len(calls) == 3
    assert all(
        sum(part.get("type") == "image_url" for part in call["messages"][-1].content)
        == 1
        for call in calls
    )
    assert adapter.execution_calls == 10
    assert Path(result["anchor_annotation_ref"]).is_file()
    execution = result["result"]
    assert execution["controller"] == "habitat_greedy_follower"
    assert execution["control_steps"] == 10
    assert execution["pixel_goal"]["affordance_pixel"] == {
        "left": [],
        "front": [320, 360],
        "right": [],
    }
    assert execution["pixel_goal"]["target_pixel"] == {
        "left": [],
        "front": [],
        "right": [],
    }
    assert [frame["frame_index"] for frame in execution["trajectory_keyframes"]] == [0, 3, 7, 10]
    assert all(len(frame["rgb_refs"]) == 3 for frame in execution["trajectory_keyframes"])
    assert all(frame["location_id"].startswith("location-") for frame in execution["trajectory_keyframes"])
    assert all(len(frame["navigation_point_xyz_m"]) == 3 for frame in execution["trajectory_keyframes"])
    assert [item["location_id"] for item in result["trajectory_memory"]] == [
        frame["location_id"] for frame in execution["trajectory_keyframes"]
    ]
    remembered_frame = service.navigation_memory.get_place(
        execution["trajectory_keyframes"][-1]["location_id"]
    )
    assert remembered_frame is not None
    assert remembered_frame.source == "trajectory_keyframe"
    assert remembered_frame.is_route_node is False
    assert remembered_frame.pose.yaw == 0.0
    assert remembered_frame.navigation_point_xyz_m == [0.0, 0.0, -1.8]
    assert set(remembered_frame.image_refs) == {"left", "front", "right"}
    assert result["next_observation_id"] == "hierarchical-obs-1"
    assert result["post_observation"]["trace"]["observation_id"] == "hierarchical-obs-1"
    assert set(result["post_observation"]["rgb_refs"]) == {"left", "front", "right"}
    assert service._latest_observation["trace"]["observation_id"] == "hierarchical-obs-1"
    assert service.health()["version_profile"] == "hierarchical"
    assert result["memory_delta"]["current_place"]["place_id"] == "location-001"
    assert service.navigation_memory.get_place("location-001").navigation_point_xyz_m == [0.0, 0.0, 0.0]
    recalled = service.recall_visual_memory({"scope": "task_overview", "limit": 4})
    assert recalled["count"] == 4
    assert recalled["archive_count"] == 12
    assert recalled["memories"][0]["frame_index"] == 0
    assert recalled["memories"][-1]["frame_index"] == 10
    recalled_location = service.recall_visual_memory(
        {
            "scope": "task",
            "location_id": execution["trajectory_keyframes"][-1]["location_id"],
            "limit": 4,
        }
    )
    assert recalled_location["count"] == 3
    assert {memory["view"] for memory in recalled_location["memories"]} == {
        "left",
        "front",
        "right",
    }
    assert Path(result["post_observation"]["topdown_ref"]).name == "semantic_topdown.png"
    assert [skill["name"] for skill in service.capabilities()["skills"]] == [
        "AnchorGenerationSkill",
        "AnchorGroundingSkill",
        "LocalizationMapSkill",
        "PathExecutionSkill",
        "PathPlanningSkill",
        "PixelToPointSkill",
        "PointNavigationSkill",
        "ViewpointAdjustmentSkill",
    ]


def test_anchor_grounding_searches_all_three_views_and_uses_qwen_selected_view(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGroundingSkill(lambda: FakeProvider(calls, view="right")))
    service = HierarchicalNavigationService(adapter, skills)
    observed = service.observe()["observation"]

    result = service.run_anchor_grounding(
        {
            "observation_id": observed["trace"]["observation_id"],
            "anchor": "doorway beside the green chair",
            "view": "front",
        }
    )

    outputs = result["result"]["outputs"]
    assert outputs["view"] == "right"
    assert outputs["anchor_visible"] is True
    assert outputs["x_px"] == 320
    assert outputs["y_px"] == 345
    assert outputs["bbox_px"] == [288, 297, 351, 345]
    assert Path(outputs["annotated_ref"]).name.endswith("_right.jpg")
    assert len(calls) == 3
    contents = [call["messages"][-1].content for call in calls]
    assert all(
        sum(part.get("type") == "image_url" for part in content) == 1
        for content in contents
    )
    assert all(
        part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for content in contents
        for part in content
        if part.get("type") == "image_url"
    )


def test_slow_annotation_draws_visual_target_with_executed_affordance(
    tmp_path,
):
    source = tmp_path / "right.jpg"
    Image.new("RGB", (640, 480), color=(80, 80, 80)).save(source)
    affordance = AnchorPixel(
        anchor="clear floor",
        view="right",
        x_px=320,
        y_px=460,
        anchor_visible=True,
        confidence=0.95,
        evidence_summary="Traversable floor toward the target.",
        bbox_px=[240, 320, 400, 470],
    )
    target = AnchorPixel(
        anchor="bed",
        view="right",
        x_px=110,
        y_px=280,
        anchor_visible=True,
        confidence=0.90,
        evidence_summary="The bed is visibly grounded.",
        bbox_px=[20, 120, 200, 280],
    )

    output = AnchorGroundingSkill._annotate_anchor(
        str(source),
        affordance,
        "dual-pixel",
        primary_role="affordance",
        secondary_target=target,
    )

    pixels = np.asarray(Image.open(output))
    target_border = pixels[116:125, 16:205]
    affordance_border = pixels[316:325, 236:405]
    assert np.any(
        (target_border[:, :, 0] < 140)
        & (target_border[:, :, 1] > 150)
        & (target_border[:, :, 2] > 170)
    )
    assert np.any(
        (affordance_border[:, :, 0] > 170)
        & (affordance_border[:, :, 1] > 140)
        & (affordance_border[:, :, 2] < 140)
    )


def test_target_anchor_uses_bbox_bottom_center_not_affordance_pixel(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skill = AnchorGenerationSkill(lambda: FakeProvider(calls), adapter)
    observation = asyncio.run(adapter.observe(make_trace(observation_id="obs-target")))
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="generate target anchor",
        inputs={
            "source_observation_id": "obs-target",
            "anchor": "bed",
            "anchor_role": "target",
            "view": "front",
        },
    )

    result = asyncio.run(
        skill.run(
            request,
            observation,
            SkillContext(InMemoryEventBus(), CancellationToken()),
        )
    )

    assert result.outputs["grounding_pixel"] == [320, 345]
    assert result.outputs["target_pixel"] == [320, 345]
    assert "affordance_pixel" not in result.outputs


def test_anchor_grounding_does_not_annotate_an_anchor_missing_from_all_views(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(
        AnchorGroundingSkill(
            lambda: FakeProvider(calls, confidence=0.99, anchor_visible=False)
        )
    )
    service = HierarchicalNavigationService(adapter, skills)
    observed = service.observe()["observation"]

    result = service.run_anchor_grounding(
        {
            "observation_id": observed["trace"]["observation_id"],
            "anchor": "bed",
            "view": "front",
        }
    )

    outputs = result["result"]["outputs"]
    assert outputs["anchor_visible"] is False
    assert outputs["confidence"] == 0.0
    assert outputs["annotated_ref"] is None


def test_anchor_grounding_rejects_a_panel_outside_topology_guided_views(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGroundingSkill(lambda: FakeProvider(calls, view="right")))
    service = HierarchicalNavigationService(adapter, skills)
    observed = service.observe()["observation"]

    result = service.run_anchor_grounding(
        {
            "observation_id": observed["trace"]["observation_id"],
            "anchor": "corridor toward the bedroom",
            "view": "front",
            "allowed_views": ["front"],
            "direction_hint": "The next topology edge is in the front view.",
        }
    )

    outputs = result["result"]["outputs"]
    assert outputs["view"] == "front"
    assert outputs["anchor_visible"] is False
    assert outputs["confidence"] == 0.0
    content = calls[0]["messages"][1].content
    assert sum(part.get("type") == "image_url" for part in content) == 1
    assert "only these camera views are allowed: FRONT" in content[0]["text"]


def test_localization_map_binds_semantics_and_detects_revisit(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(LocalizationMapSkill())
    service = HierarchicalNavigationService(adapter, skills)

    first = service.inspect_localization_map(
        semantic_map(
            "white corridor near cabinets",
            left=["white cabinets"],
            front=["corridor"],
            right=[],
        )
    )
    second = service.inspect_localization_map(
        semantic_map(
            "same corridor facing doorway",
            left=["doorway"],
            front=["white cabinets"],
            right=[],
        )
    )

    assert first["memory_delta"]["revisited"] is False
    assert second["memory_delta"]["revisited"] is True
    assert second["memory_delta"]["current_place"]["place_id"] == "location-001"
    location = service.navigation_memory.get_place("location-001")
    assert location.visit_count == 2
    assert location.semantic_labels == ["white cabinets", "corridor", "doorway"]
    assert location.semantic_observations[-1].left == ["doorway"]
    assert len(second["memory_view"]["relevant_places"]) <= 5
    assert second["next_observation_id"] == "hierarchical-obs-1"
    assert second["result"]["topdown_ref"].endswith("semantic_topdown.png")
    assert set(second["observation"]["rgb_refs"]) == {"left", "front", "right"}
    service.reset()
    assert service.health()["explored_location_count"] == 0
    assert service.health()["topology_route_node_count"] == 0


def test_known_location_navigation_reuses_saved_world_point_without_qwen(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    target_observation = asyncio.run(adapter.observe(make_trace(observation_id="old-observation")))
    target_observation = target_observation.model_copy(
        update={
            "pose": Pose2D(x=0.0, y=-2.0, yaw=1.57, source="test", frame="habitat_world_xz"),
            "metadata": {**target_observation.metadata, "metric_position_xyz": [0.0, 0.0, -2.0]},
        }
    )
    service.navigation_memory.record_place(
        target_observation,
        SemanticMapUpdate.model_validate(
            semantic_map(
                "doorway to the main corridor",
                left=["doorway"],
                front=["main corridor"],
                right=[],
            )
        ),
    )
    observed = service.observe()["observation"]

    result = service.navigate_to_known_location(
        {
            "location_id": "location-001",
            "observation_id": observed["trace"]["observation_id"],
            "semantic_map": semantic_map(),
            "max_control_steps": 24,
        }
    )

    assert result["ok"] is True
    assert result["location_id"] == "location-001"
    assert result["target_location"]["scene_summary"] == "doorway to the main corridor"
    assert [stage["skill"] for stage in result["stages"]] == ["PointNavigationSkill"]
    assert adapter.execution_calls == 10
    assert result["result"]["controller"] == "habitat_greedy_follower"
    assert result["result"]["target_yaw"] == 1.57
    assert result["result"]["orientation_steps"] == 6
    assert result["result"]["orientation_error_rad"] < 0.14
    assert set(result["post_observation"]["rgb_refs"]) == {"left", "front", "right"}
    assert service.navigation_memory.get_place("location-001").navigation_point_xyz_m == [0.0, 0.0, -2.0]


def test_topology_node_navigation_continues_control_chunks_without_agent_boundary(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    service.navigation_memory.record_backbone_topology(
        BackboneTopology(
            map_id="test-map",
            geometry_revision="test-v1",
            source="habitat_navmesh",
            frame="habitat_world_xz",
            cache_key="atomic-node-navigation",
            nodes=[
                BackboneNode(
                    node_id="backbone-003",
                    node_type="open_space_center",
                    x=0.0,
                    y=-2.0,
                    row=0,
                    column=0,
                    clearance_m=1.2,
                    navigation_point_xyz_m=[0.0, 0.0, -2.0],
                )
            ],
        )
    )
    observed = service.observe()["observation"]

    result = service.navigate_to_known_location(
        {
            "topology_node_id": "backbone-003",
            "observation_id": observed["trace"]["observation_id"],
            "semantic_map": semantic_map(),
            "max_control_steps": 4,
        }
    )

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert len(result["stages"]) == 3
    assert result["result"]["execution_chunks"] == 3
    assert result["result"]["termination_reason"] == "reached"
    assert result["result"]["control_steps"] == 10
    assert result["result"]["target_yaw"] is None
    assert result["result"]["orientation_steps"] == 0
    assert adapter.execution_calls == 10
    assert [frame["frame_index"] for frame in result["result"]["trajectory_keyframes"]] == [
        0,
        3,
        7,
        10,
    ]


def test_advance_navigation_routes_structured_viewpoint_decision(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation()
    observation_id = started["observation"]["trace"]["observation_id"]
    assert started["memory_view"]["global_anchor"] is None
    assert service.navigation_memory.place_count == 0
    assert service.navigation_memory.visual_memories() == []
    assert started["topology"] == {"status": "unsupported", "loaded": False}
    assert started["route_graph"]["nodes"] == []

    decision = {
            "observation_id": observation_id,
            "global_goal": "find the bed in the bedroom",
            "target_evidence": ["bed", "bedroom doorway"],
            "previous_assessment": {
                "execution_status": "not_started",
                "task_effect": "unknown",
                "evidence": "initial observation",
            },
            "route_hypothesis": {
                "statement": "turning left may reveal a room entrance",
                "confidence": 0.6,
            },
            "candidates": [
                {
                    "action_type": "viewpoint_adjustment",
                    "target": "left-side room entrance",
                    "task_relevance": 0.7,
                    "information_gain": 0.9,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "viewpoint_adjustment",
                "target": "left-side room entrance",
                "viewpoint_action": "turn_left",
                "expected_evidence": "a doorway or bedroom furniture",
                "success_condition": "the left-side space becomes visible",
                "fallback": "choose a visible corridor anchor",
            },
            "semantic_map": semantic_map(),
        }
    result = service.advance_navigation(decision)

    assert adapter.viewpoint_actions == ["turn_left"] * 6
    assert result["decision"]["action_type"] == "viewpoint_adjustment"
    assert result["slow_output"]["reference_observation_id"] == observation_id
    assert result["slow_context"]["previous_decision"] == result["slow_output"]
    assert result["memory_view"]["global_goal"] == "find the bed in the bedroom"
    assert result["memory_view"]["route_hypothesis"] == "turning left may reveal a room entrance"
    assert result["memory_view"]["global_anchor"] is None
    assert all(not place.is_global_anchor for place in service.navigation_memory.places())
    assert all(
        visual.kind != "global_anchor"
        for visual in service.navigation_memory.visual_memories()
    )
    relay_graph = service.relay_state()["route_graph"]
    assert relay_graph["nodes"] == []
    assert service.health()["topology_route_node_count"] == 0
    revision = service.navigation_memory.revision
    try:
        service.advance_navigation(decision)
    except ValueError as exc:
        assert "stale observation_id" in str(exc)
    else:
        raise AssertionError("stale structured decisions must be rejected")
    assert service.navigation_memory.revision == revision


def test_first_slow_decision_plans_after_topology_context_and_is_cached_before_control(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(
        AnchorGenerationSkill(lambda: FakeProvider(calls, confidence=0.95), adapter)
    )
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)

    started = service.begin_navigation(
        {"global_goal": "go to the bedroom and find the bed"}
    )
    observation_id = started["observation"]["trace"]["observation_id"]
    assert started["planner_required"] is True
    assert started["semantic_route_plan"] is None
    assert started["slow_context"]["topology_context"]["agentic_plan"] is None

    cached_before_control = []
    adapter.before_control_step = lambda: cached_before_control.append(
        service.navigation_memory.last_slow_output
    )
    result = service.advance_navigation(
        {
            "observation_id": observation_id,
            "global_goal": "ignored repeated task text",
            "navigation_task_type": "object_goal",
            "cot": {
                "target_localization_and_key_context": (
                    "The bed is visible in the front view inside the bedroom."
                ),
                "path_planning": "Use the visible bed as the final local target.",
            },
            "self_reflection": None,
            "planner_output": {
                "global_instruction": "go to the bedroom and find the bed",
                "destination_query": "bedroom",
                "object_goal": "bed",
                "subtasks": [
                    {
                        "stage": "reaching",
                        "task_type": "object_goal",
                        "instruction": "Locate and reach the bed in the bedroom.",
                        "target": "bed",
                        "success_condition": "Current visual evidence confirms the bed.",
                    }
                ],
            },
            "previous_assessment": {
                "execution_status": "not_started",
                "task_effect": "unknown",
                "evidence": "Initial synchronized observation.",
            },
            "route_hypothesis": {
                "statement": "The visible bed is the final target.",
                "confidence": 0.95,
            },
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "target": "bed",
                    "task_relevance": 1.0,
                    "information_gain": 1.0,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "target": "bed",
                "view": "front",
                "anchor_role": "target",
                "expected_evidence": "The bed remains grounded.",
                "success_condition": "Reach the planned bed approach point.",
                "fallback": "Reassess the current views.",
            },
            "semantic_map": semantic_map("bedroom with a visible bed"),
            "max_control_steps": 16,
        }
    )

    assert cached_before_control
    assert all(item is not None for item in cached_before_control)
    first_cached = cached_before_control[0]
    assert first_cached["reference_observation_id"] == observation_id
    assert first_cached["pixel_goal"]["affordance_pixel"]["front"] == [320, 360]
    assert first_cached["pixel_goal"]["target_pixel"]["front"] == [320, 345]
    assert result["slow_output"] == service.navigation_memory.last_slow_output
    assert result["fast_execution"]["slow_step_id"] == result["slow_output"][
        "slow_step_id"
    ]
    assert result["result"]["slow_step_id"] == result["slow_output"]["slow_step_id"]
    assert result["agentic_plan"]["global_instruction"] == (
        "go to the bedroom and find the bed"
    )
    assert result["agentic_plan"]["subtasks"][0]["task_type"] == "object_goal"
    target_evidence = result["slow_context"]["active_target_evidence"]
    assert target_evidence["target"] == "bed"
    assert target_evidence["confidence"] == 0.95
    assert target_evidence["execution_reached"] is True
    assert target_evidence["source_observation_id"] == observation_id
    assert result["next_call_contract"]["self_reflection_subtask_id"] == "subtask-001"
    assert result["next_call_contract"]["navigation_task_type"] == "object_goal"


def test_runtime_completes_point_only_plan_with_object_reaching_stage(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation(
        {"global_goal": "go to office to find a white fridge"}
    )
    planner = {
        "global_instruction": "go to office to find a white fridge",
        "destination_query": "office",
        "object_goal": "white fridge",
        "subtasks": [
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Reach the office region.",
                "target": "office",
                "success_condition": "The office region is reached.",
            }
        ],
    }

    result = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            planner_output=planner,
        )
    )

    assert [item["task_type"] for item in result["agentic_plan"]["subtasks"]] == [
        "point_goal",
        "object_goal",
    ]
    assert result["agentic_plan"]["subtasks"][1]["target"] == "white fridge"
    assert result["planner_output"]["subtasks"][1]["stage"] == "reaching"
    assert result["next_call_contract"] == {
        "next_tool": "advance_navigation",
        "observation_id": result["next_observation_id"],
        "planner_output": "only_when_self_reflection.should_replan",
        "self_reflection": "required",
        "self_reflection_subtask_id": "subtask-001",
        "navigation_task_type": "point_goal",
        "active_target_evidence": False,
        "prose_completion_allowed": False,
    }


def test_redundant_planner_output_is_ignored_without_replacing_plan(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation({"global_goal": "find the target"})
    planner = {
        "global_instruction": "find the target",
        "destination_query": "target region",
        "object_goal": "target object",
        "subtasks": [
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Reach the target region.",
                "target": "target region",
                "success_condition": "The target region is reached.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the target object.",
                "target": "target object",
                "success_condition": "The target object is visible.",
            },
        ],
    }
    first = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            planner_output=planner,
        )
    )
    plan_id = service._semantic_route_plan.agentic_plan.plan_id

    result = service.advance_navigation(
        viewpoint_decision(
            first["next_observation_id"],
            task_type="object_goal",
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The region appears reached.",
                "should_replan": False,
            },
            planner_output=planner,
        )
    )

    assert result["protocol_normalization"]["ignored_planner_output"] is True
    assert result["planner_output"] is None
    assert result["agentic_plan"]["plan_id"] == plan_id
    assert result["agentic_plan"]["active_subtask_index"] == 1


def test_transition_uses_active_plan_task_type_without_model_retry(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation({"global_goal": "find the target"})
    planner = {
        "global_instruction": "find the target",
        "destination_query": "target region",
        "object_goal": "target object",
        "subtasks": [
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Reach the target region.",
                "target": "target region",
                "success_condition": "The target region is reached.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the target object.",
                "target": "target object",
                "success_condition": "The target object is visible.",
            },
        ],
    }
    first_payload = viewpoint_decision(
        started["observation"]["trace"]["observation_id"],
        task_type="object_goal",
        planner_output=planner,
    )
    first = service.advance_navigation(first_payload)
    assert first["protocol_normalization"] == {
        "corrected_navigation_task_type": True,
        "from": "object_goal",
        "to": "point_goal",
        "reason": "Agentic Plan active-subtask contract is authoritative.",
    }

    transition = service.advance_navigation(
        viewpoint_decision(
            first["next_observation_id"],
            task_type="point_goal",
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The target region was reached.",
                "should_replan": False,
            },
        )
    )
    assert transition["protocol_normalization"][
        "corrected_navigation_task_type"
    ] is True
    assert transition["protocol_normalization"]["to"] == "object_goal"
    assert transition["slow_output"]["navigation_task_type"] == "object_goal"
    assert transition["slow_output"]["cot"]["path_planning"]


def test_point_goal_reflection_is_kept_partial_until_metric_node_arrival(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="reflection-arrival",
        nodes=[
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=-4.0,
                y=0.0,
                row=0,
                column=0,
                clearance_m=1.2,
                navigation_point_xyz_m=[-4.0, 0.0, 0.0],
            )
        ],
    )
    service._topology_model = topology
    service._topology_loaded_for_task = True
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-001": NodeSemantics(
                    place_type="target region",
                    objects=["target object"],
                    source="human",
                )
            },
        )
    )
    started = service.begin_navigation(
        {
            "global_goal": "find the target",
            "destination_query": "target region",
            "object_goal": "target object",
        }
    )
    observation = service._observations[
        started["observation"]["trace"]["observation_id"]
    ]
    request = AdvanceNavigationInput.model_validate(
        viewpoint_decision(
            observation.trace.observation_id,
            task_type="point_goal",
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The target node appears reached.",
                "should_replan": False,
            },
        )
    )
    normalized, protocol_normalization = service._normalize_point_goal_reflection(
        request,
        observation,
    )
    assert service._preflight_advance_transition(
        normalized,
        observation,
        validate_task_type=True,
    ) is False
    service._apply_subtask_reflection(normalized.self_reflection, observation)

    assert protocol_normalization["corrected_self_reflection"] is True
    assert "cannot complete point-goal" in protocol_normalization["reason"]
    assert normalized.self_reflection.completed is False
    assert normalized.self_reflection.outcome == "partial"
    plan = service._semantic_route_plan.agentic_plan
    assert plan.active_subtask_index == 0
    assert plan.subtasks[0].status == "active"


def test_subtask_transition_synchronizes_regional_node(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="active-node-sync",
        nodes=[
            BackboneNode(
                node_id="backbone-001",
                node_type="junction",
                x=0.0,
                y=0.0,
                row=0,
                column=0,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-002",
                node_type="open_space_center",
                x=-3.0,
                y=0.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[-3.0, 0.0, 0.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-001",
                target="backbone-002",
                distance_m=3.0,
                path_world_xy_m=[(0.0, 0.0), (-3.0, 0.0)],
            )
        ],
    )
    service._topology_model = topology
    service._topology_loaded_for_task = True
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-001": NodeSemantics(place_type="office", source="human"),
                "backbone-002": NodeSemantics(
                    place_type="open hall",
                    objects=["refrigerator"],
                    source="human",
                ),
            },
        )
    )
    started = service.begin_navigation({"global_goal": "find the refrigerator"})
    planner = {
        "global_instruction": "find the refrigerator",
        "destination_query": "open hall with refrigerator",
        "object_goal": "refrigerator",
        "subtasks": [
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Reach the office junction.",
                "target": "office junction",
                "topology_node_id": "backbone-001",
                "success_condition": "backbone-001 is reached.",
            },
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Reach the refrigerator hall.",
                "target": "refrigerator hall",
                "topology_node_id": "backbone-002",
                "success_condition": "backbone-002 is reached.",
            },
        ],
    }
    first = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            planner_output=planner,
        )
    )
    assert first["semantic_route_plan"]["selected_topology_node_id"] == "backbone-001"

    second = service.advance_navigation(
        viewpoint_decision(
            first["next_observation_id"],
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "Metric localization confirms backbone-001.",
                "should_replan": False,
            },
        )
    )

    assert second["agentic_plan"]["active_subtask_index"] == 1
    assert second["semantic_route_plan"]["selected_topology_node_id"] == "backbone-002"
    assert second["semantic_route_plan"]["regional_guidance"][
        "target_topology_node_id"
    ] == "backbone-002"


def test_final_subtask_reflection_waits_for_finish_verifier(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation({"global_goal": "find the target"})
    planner = {
        "global_instruction": "find the target",
        "destination_query": "target region",
        "object_goal": "target object",
        "subtasks": [
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the target object.",
                "target": "target object",
                "success_condition": "The target object is confirmed.",
            }
        ],
    }
    first = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            task_type="object_goal",
            planner_output=planner,
        )
    )
    service._active_target_evidence = ABotTargetEvidence(
        target="target object",
        confidence=0.95,
        source_observation_id=first["next_observation_id"],
        post_observation_id=first["next_observation_id"],
        view="front",
        target_pixel=(320, 240),
        affordance_pixel=(320, 245),
        execution_reached=True,
        remaining_distance_m=0.2,
        evidence_summary="The target object is visually confirmed in the front view.",
    )
    action_count = len(adapter.viewpoint_actions)

    result = service.advance_navigation(
        viewpoint_decision(
            first["next_observation_id"],
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The final subtask appears complete.",
                "should_replan": False,
            },
        )
    )

    assert result["status"] == "awaiting_finish_verification"
    assert result["finish_required"] is True
    assert len(adapter.viewpoint_actions) == action_count
    assert result["agentic_plan"]["subtasks"][0]["status"] == "active"
    assert result["semantic_route_plan"]["phase"] == "verify_goal"
    assert result["next_call_contract"]["next_tool"] == "finish_navigation_task"


def test_object_subtask_completion_delegates_semantics_to_finish_verifier(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation({"global_goal": "find the target"})
    planner = {
        "global_instruction": "find the target",
        "destination_query": "target region",
        "object_goal": "target object",
        "subtasks": [
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the target object.",
                "target": "target object",
                "success_condition": "The target object is confirmed.",
            }
        ],
    }
    first = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            task_type="object_goal",
            planner_output=planner,
        )
    )

    result = service.advance_navigation(
        viewpoint_decision(
            first["next_observation_id"],
            task_type="object_goal",
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The target object is visible in the current observation.",
                "should_replan": False,
            },
        )
    )

    assert result["status"] == "awaiting_finish_verification"
    assert result["finish_required"] is True


def test_repeated_begin_for_same_goal_returns_existing_plan(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation({"global_goal": "find the target"})
    planner = {
        "global_instruction": "find the target",
        "destination_query": "target region",
        "object_goal": "target object",
        "subtasks": [
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Reach the target region.",
                "target": "target region",
                "success_condition": "The target region is reached.",
            }
        ],
    }
    first = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            planner_output=planner,
        )
    )

    repeated = service.begin_navigation({"global_goal": "find the target"})

    assert repeated["planner_required"] is False
    assert repeated["semantic_route_plan"]["plan_id"] == first["semantic_route_plan"][
        "plan_id"
    ]
    assert repeated["slow_context"]["topology_context"]["agentic_plan"][
        "plan_id"
    ] == first["agentic_plan"]["plan_id"]


def test_advance_does_not_mutate_the_frozen_semantic_backbone(
    tmp_path, make_trace
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGenerationSkill(lambda: FakeProvider(calls, confidence=0.91), adapter))
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation()
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="test-fixed-backbone",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=0.0,
                y=0.0,
                row=0,
                column=0,
                clearance_m=1.0,
                navigation_point_xyz_m=[0.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=0.0,
                y=-2.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -2.0],
            ),
        ],
        edges=[BackboneEdge(source="backbone-000", target="backbone-001", distance_m=2.0)],
    )
    locations = service.navigation_memory.record_backbone_topology(topology)["node_locations"]
    route_count_before = service.navigation_memory.route_node_count

    result = service.advance_navigation(
        {
            "observation_id": started["observation"]["trace"]["observation_id"],
            "global_goal": "find the bed in the bedroom",
            "target_evidence": ["bedroom doorway"],
            "previous_assessment": {
                "execution_status": "not_started",
                "task_effect": "unknown",
                "evidence": "initial observation",
            },
            "route_hypothesis": {
                "statement": "The visible doorway is directly associated with the bedroom.",
                "confidence": 0.94,
            },
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "target": "the visible doorway",
                    "task_relevance": 0.95,
                    "information_gain": 0.8,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "target": "the visible doorway",
                "view": "front",
                "expected_evidence": "bedroom interior",
                "success_condition": "reach the doorway",
                "fallback": "inspect another route",
            },
            "semantic_map": semantic_map("corridor ending at the bedroom doorway"),
            "node_semantics": {
                "place_type": "corridor junction",
                "function": "connects the laboratory to adjacent rooms",
                "objects": ["doorway", "cabinet"],
                "landmarks": ["wooden doorway"],
                "spatial_character": "narrow transition with two continuations",
                "task_affordances": ["enter adjacent room"],
                "aliases": ["room transition"],
                "confidence": 0.94,
                "source": "model",
            },
            "max_control_steps": 24,
        }
    )

    assert result["ok"] is True
    assert result["topology_update"] is None
    graph = service.navigation_memory.route_graph()
    assert service.navigation_memory.route_node_count == route_count_before == 2
    assert not any(node["route_node_type"] == "semantic_landmark" for node in graph["nodes"])
    annotated = next(
        node for node in graph["nodes"] if node["location_id"] == locations["backbone-000"]
    )
    assert annotated["source"] == "map_backbone"
    assert annotated["semantic_confidence"] is None
    assert annotated["task_relevance"] is None
    assert annotated["visited"] is False
    assert annotated["topology_node_id"] == "backbone-000"
    assert annotated["node_semantics"] is None


def test_explore_route_keeps_visual_anchor_instead_of_forcing_nearby_node(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    started = service.begin_navigation()
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="test-exploration-backbone",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=0.0,
                y=-2.0,
                row=0,
                column=0,
                clearance_m=1.0,
                navigation_point_xyz_m=[0.0, 0.0, -2.0],
            )
        ],
        edges=[],
    )
    location_id = service.navigation_memory.record_backbone_topology(topology)["node_locations"][
        "backbone-000"
    ]
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-000": NodeSemantics(
                    place_type="corridor junction",
                    function="route exploration",
                    source="human",
                )
            },
        )
    )
    observation = service._observations[started["observation"]["trace"]["observation_id"]]
    nearby = service.navigation_memory.view(
        current_observation=observation
    ).nearby_backbone_nodes
    assert [item.location_id for item in nearby] == [location_id]
    assert nearby[0].visited is False

    request = AdvanceNavigationInput.model_validate(
        {
            "observation_id": started["observation"]["trace"]["observation_id"],
            "global_goal": "find the bedroom",
            "previous_assessment": {
                "execution_status": "not_started",
                "task_effect": "unknown",
                "evidence": "no direct bedroom anchor is visible",
            },
            "route_hypothesis": {
                "statement": "explore the nearest route decision point",
                "confidence": 0.7,
            },
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "intent": "explore_route",
                    "target": "open corridor floor",
                    "task_relevance": 0.5,
                    "information_gain": 0.9,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "intent": "explore_route",
                "target": "open corridor floor",
                "view": "front",
                "expected_evidence": "a new route branch",
                "success_condition": "reach a better observation point",
                "fallback": "return to the task start",
            },
            "semantic_map": semantic_map("corridor without direct goal evidence"),
            "max_control_steps": 24,
        }
    )

    action, policy_override = service._apply_navigation_policy(request, observation)

    assert action.action_type == "visible_anchor"
    assert action.target == "open corridor floor"
    assert policy_override is None
    assert service.navigation_memory.route_node_count == 1


def test_semantic_object_node_remains_context_without_overriding_visible_anchor(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="mixed-skill-priority",
        nodes=[
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=0.0,
                y=-2.0,
                row=0,
                column=0,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -2.0],
            ),
            BackboneNode(
                node_id="backbone-002",
                node_type="open_space_center",
                x=0.0,
                y=-5.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -5.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-001",
                target="backbone-002",
                distance_m=3.0,
            )
        ],
    )
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-001": NodeSemantics(
                    place_type="office area",
                    objects=["white cabinets"],
                    source="human",
                ),
                "backbone-002": NodeSemantics(
                    place_type="open hall",
                    objects=["refrigerator"],
                    landmarks=["tall white refrigerator"],
                    task_affordances=["navigate to refrigerator"],
                    source="human",
                ),
            },
        )
    )
    started = service.begin_navigation(
        {
            "global_goal": "go to office to find a white refrigerator",
            "destination_query": "office",
            "object_goal": "white refrigerator",
        }
    )
    service._update_semantic_route_after_execution(
        SelectedNavigationAction(
            action_type="known_location",
            target="office area",
            topology_node_id="backbone-001",
            expected_evidence="office views",
            success_condition="reach the office node",
            fallback="reassess",
        ),
        {"result": {"reached": True}},
    )
    current = service._observations[started["observation"]["trace"]["observation_id"]]
    request = AdvanceNavigationInput.model_validate(
        {
            "observation_id": current.trace.observation_id,
            "global_goal": "go to office to find a white refrigerator",
            "previous_assessment": {
                "execution_status": "completed",
                "task_effect": "no_progress",
                "evidence": "The office node has no visible white fridge.",
            },
            "route_hypothesis": {
                "statement": "Continue along the red carpet or use mapped refrigerator evidence.",
                "confidence": 0.8,
            },
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "intent": "explore_route",
                    "target": "open office space ahead",
                    "task_relevance": 0.7,
                    "information_gain": 0.7,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "intent": "explore_route",
                "target": "open office space ahead",
                "view": "front",
                "expected_evidence": "More office space.",
                "success_condition": "Reveal the next area.",
                "fallback": "Switch to backbone-002.",
            },
            "semantic_map": semantic_map("office without a visible white fridge"),
        }
    )

    action, policy_override = service._apply_navigation_policy(request, current)

    assert action.action_type == "visible_anchor"
    assert action.intent == "explore_route"
    assert action.target == "open office space ahead"
    assert policy_override is None
    assert service._semantic_route_plan is not None
    assert service._semantic_route_plan.phase == "local_anchor_search"
    assert service._semantic_route_plan.selected_topology_node_id == "backbone-001"


def test_task_level_semantic_ranker_selects_refrigerator_node_for_fridge_query(
    tmp_path,
    make_trace,
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(
        adapter,
        SkillRegistry(),
        semantic_node_provider_factory=lambda: FakeSemanticNodeProvider(calls),
    )
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="semantic-ranker",
        nodes=[
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=0.0,
                y=-2.0,
                row=0,
                column=0,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -2.0],
            ),
            BackboneNode(
                node_id="backbone-002",
                node_type="open_space_center",
                x=0.0,
                y=-5.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -5.0],
            ),
        ],
    )
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-001": NodeSemantics(
                    place_type="office area",
                    objects=["white cabinets"],
                    source="human",
                ),
                "backbone-002": NodeSemantics(
                    place_type="open hall",
                    objects=["refrigerator"],
                    landmarks=["tall white refrigerator"],
                    source="human",
                ),
            },
        )
    )

    started = service.begin_navigation(
        {
            "global_goal": "go to office to find a fridge",
            "destination_query": "office",
            "object_goal": "fridge",
        }
    )

    plan = started["semantic_route_plan"]
    assert plan["selected_topology_node_id"] == "backbone-002"
    assert plan["match_score"] == 0.96
    assert plan["matched_terms"] == ["fridge", "refrigerator"]
    assert plan["selection_reason"].startswith("Semantic task ranker selected")
    assert [
        (subtask["stage"], subtask["task_type"], subtask["status"])
        for subtask in plan["agentic_plan"]["subtasks"]
    ] == [
        ("approaching", "point_goal", "active"),
        ("reaching", "object_goal", "pending"),
    ]
    assert started["slow_context"]["topology_context"]["topo_memory"][
        "object_poi_layer"
    ]
    assert len(calls) == 1


def test_destination_reconciliation_preserves_intermediate_reaching_semantics():
    planner = AgenticPlannerOutput(
        global_instruction="go through the lobby, exit, turn left, and find the robot",
        destination_query="robot area after the left turn",
        object_goal="robot",
        subtasks=[
            {
                "stage": "reaching",
                "task_type": "point_goal",
                "instruction": "Reach the lobby exit.",
                "target": "lobby exit",
                "topology_node_id": "backbone-001",
                "success_condition": "The lobby exit is reached.",
            },
            {
                "stage": "interaction",
                "task_type": "object_goal",
                "instruction": "Exit, turn left, and go straight.",
                "target": "ordered route segment",
                "success_condition": "The ordered route segment is complete.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the robot.",
                "target": "robot",
                "success_condition": "The robot is visually confirmed.",
            },
        ],
    )
    destination = SemanticNodeSelection(
        selected_topology_node_id="backbone-002",
        confidence=0.95,
        matched_concepts=["robot", "left-side room"],
        selection_reason="The final robot belongs to the selected destination region.",
    )

    normalized = HierarchicalNavigationService._complete_planner_output(
        planner,
        destination_topology_node_id="backbone-002",
        destination_selection=destination,
    )

    reaching = [item for item in normalized.subtasks if item.stage == "reaching"]
    assert [item.stage for item in normalized.subtasks] == [
        "reaching",
        "interaction",
        "approaching",
        "reaching",
    ]
    assert reaching[0].target == "lobby exit"
    assert reaching[0].topology_node_id == "backbone-001"
    assert reaching[-1].target == "robot"
    assert reaching[-1].topology_node_id == "backbone-002"


def test_destination_point_goal_does_not_reorder_leading_route_instruction():
    planner = AgenticPlannerOutput(
        global_instruction="go out of the room, turn left, go straight, and find the robot",
        destination_query="area after leaving the room and turning left",
        object_goal="robot",
        subtasks=[
            {
                "stage": "interaction",
                "task_type": "object_goal",
                "instruction": "Exit the room, turn left, and go straight.",
                "target": "ordered route segment",
                "success_condition": "The ordered route segment is complete.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the robot.",
                "target": "robot",
                "success_condition": "The robot is visually confirmed.",
            },
        ],
    )

    normalized = HierarchicalNavigationService._complete_planner_output(
        planner,
        destination_topology_node_id="backbone-002",
    )
    plan = HierarchicalNavigationService._create_agentic_plan(
        normalized,
        selected_topology_node_id="backbone-002",
    )

    assert [subtask.stage for subtask in plan.subtasks] == [
        "interaction",
        "approaching",
        "reaching",
    ]
    assert plan.subtasks[0].topology_node_id is None
    assert plan.subtasks[1].topology_node_id == "backbone-002"
    assert plan.subtasks[2].topology_node_id == "backbone-002"


def test_runtime_does_not_override_main_agent_semantic_destination(
    tmp_path,
    make_trace,
):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(
        adapter,
        skills,
        semantic_node_provider_factory=lambda: FakeSemanticNodeProvider(calls),
    )
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="planner-semantic-reconciliation",
        nodes=[
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=0.0,
                y=-2.0,
                row=0,
                column=0,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -2.0],
            ),
            BackboneNode(
                node_id="backbone-002",
                node_type="open_space_center",
                x=0.0,
                y=-5.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -5.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-001",
                target="backbone-002",
                distance_m=3.0,
                path_world_xy_m=[(0.0, -2.0), (0.0, -5.0)],
            )
        ],
    )
    service._topology_model = topology
    service._topology_loaded_for_task = True
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-001": NodeSemantics(
                    place_type="office area",
                    objects=["white cabinets"],
                    source="human",
                ),
                "backbone-002": NodeSemantics(
                    place_type="open hall",
                    objects=["refrigerator"],
                    landmarks=["tall white refrigerator"],
                    source="human",
                ),
            },
        )
    )
    started = service.begin_navigation(
        {"global_goal": "go to office to find a white fridge"}
    )
    planner = {
        "global_instruction": "go to office to find a white fridge",
        "destination_query": "office",
        "object_goal": "white fridge",
        "node_selection": {
            "selected_topology_node_id": "backbone-001",
            "confidence": 0.9,
            "matched_concepts": ["office"],
            "selection_reason": "The place label matches office.",
        },
        "subtasks": [
            {
                "stage": "approaching",
                "task_type": "point_goal",
                "instruction": "Approach the office area.",
                "target": "office area",
                "topology_node_id": "backbone-001",
                "success_condition": "Runtime confirms arrival at backbone-001.",
            },
            {
                "stage": "reaching",
                "task_type": "object_goal",
                "instruction": "Find the white fridge.",
                "target": "white fridge",
                "success_condition": "The white fridge is visually confirmed.",
            },
        ],
    }

    result = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            planner_output=planner,
        )
    )

    route = result["semantic_route_plan"]
    assert route["selected_topology_node_id"] == "backbone-001"
    assert route["destination_topology_node_id"] == "backbone-001"
    assert route["destination_selection"]["confidence"] == 0.9
    assert [
        (subtask["stage"], subtask["topology_node_id"])
        for subtask in result["agentic_plan"]["subtasks"]
    ] == [
        ("approaching", "backbone-001"),
        ("reaching", "backbone-001"),
    ]
    reaching_subtask = result["agentic_plan"]["subtasks"][1]
    assert reaching_subtask["target"] == "white fridge"
    assert reaching_subtask["topology_node_id"] == "backbone-001"
    assert calls == []


def test_regional_route_constrains_anchor_view_then_opens_inside_target_region(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="direction-guidance",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=0.0,
                y=0.0,
                row=0,
                column=0,
                clearance_m=1.0,
                navigation_point_xyz_m=[0.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=-3.0,
                y=0.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[-3.0, 0.0, 0.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-000",
                target="backbone-001",
                distance_m=3.0,
                path_world_xy_m=[(0.0, 0.0), (-1.5, 0.0), (-3.0, 0.0)],
            )
        ],
    )
    service._topology_model = topology
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-000": NodeSemantics(place_type="office", source="human"),
                "backbone-001": NodeSemantics(
                    place_type="bedroom",
                    objects=["bed"],
                    source="human",
                ),
            },
        )
    )

    started = service.begin_navigation(
        {
            "global_goal": "go to bedroom to find the bed",
            "destination_query": "bedroom",
            "object_goal": "bed",
        }
    )

    guidance = started["semantic_route_plan"]["regional_guidance"]
    assert started["semantic_route_plan"]["phase"] == "approach_node"
    assert guidance["mode"] == "route_to_region"
    assert guidance["route_node_ids"] == ["backbone-000", "backbone-001"]
    assert guidance["route_progress_index"] == 0
    assert guidance["next_topology_node_id"] == "backbone-001"
    assert guidance["distance_to_next_node_m"] == 3.0
    assert guidance["distance_to_target_node_m"] == 3.0
    assert guidance["node_arrival_radius_m"] == 1.2
    assert guidance["target_region_reached"] is False
    assert guidance["relative_bearing_deg"] == 90.0
    assert guidance["allowed_anchor_views"] == ["left"]
    assert guidance["direction_constraint_active"] is True
    assert service._regional_anchor_depth_limit(5.0) == 3.0
    _, scope = service._anchor_region_context(
        {"world_point_xyz_m": [-1.0, 0.0, 0.0], "confidence": 0.95}
    )
    assert scope["active_semantic_node_id"] == "backbone-001"
    assert scope["regional_context_node_id"] == "backbone-001"
    assert scope["route_progress_index"] == 0
    assert scope["target_region_reached"] is False

    observation = service._observations[started["observation"]["trace"]["observation_id"]]
    at_goal = observation.model_copy(
        update={
            "pose": Pose2D(
                x=-3.0,
                y=0.0,
                yaw=0.0,
                source="test",
                frame="habitat_world_xz",
            )
        }
    )
    service._refresh_regional_guidance(at_goal)
    guidance = service._semantic_route_plan.regional_guidance
    assert guidance is not None
    assert guidance.mode == "search_within_region"
    assert guidance.route_progress_index == 1
    assert guidance.target_region_reached is True
    assert guidance.allowed_anchor_views == ["left", "front", "right"]
    assert guidance.direction_constraint_active is False
    assert service._semantic_route_plan.phase == "local_anchor_search"


def test_regional_anchor_chain_advances_connected_nodes_without_geometric_skip(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="connected-anchor-chain",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=0.0,
                y=0.0,
                row=0,
                column=0,
                clearance_m=1.0,
                navigation_point_xyz_m=[0.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-001",
                node_type="junction",
                x=0.0,
                y=4.0,
                row=0,
                column=1,
                clearance_m=1.0,
                navigation_point_xyz_m=[0.0, 0.0, 4.0],
            ),
            BackboneNode(
                node_id="backbone-002",
                node_type="open_space_center",
                x=1.4,
                y=0.0,
                row=1,
                column=1,
                clearance_m=1.0,
                navigation_point_xyz_m=[1.4, 0.0, 0.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-000",
                target="backbone-001",
                distance_m=4.0,
                path_world_xy_m=[(0.0, 0.0), (0.0, 4.0)],
            ),
            BackboneEdge(
                source="backbone-001",
                target="backbone-002",
                distance_m=4.24,
                path_world_xy_m=[(0.0, 4.0), (1.4, 0.0)],
            ),
        ],
    )
    service._topology_model = topology
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-000": NodeSemantics(place_type="office", source="human"),
                "backbone-001": NodeSemantics(place_type="corridor junction", source="human"),
                "backbone-002": NodeSemantics(
                    place_type="bedroom",
                    objects=["bed"],
                    source="human",
                ),
            },
        )
    )
    started = service.begin_navigation(
        {
            "global_goal": "go to bedroom to find the bed",
            "destination_query": "bedroom",
            "object_goal": "bed",
        }
    )
    observation = service._observations[started["observation"]["trace"]["observation_id"]]
    guidance = service._semantic_route_plan.regional_guidance
    assert guidance is not None
    assert guidance.route_node_ids == ["backbone-000", "backbone-001", "backbone-002"]
    assert guidance.route_progress_index == 0
    assert guidance.next_topology_node_id == "backbone-001"
    assert service._semantic_route_plan.phase == "approach_node"
    skip_request = AdvanceNavigationInput.model_validate(
        {
            "observation_id": observation.trace.observation_id,
            "global_goal": "go to bedroom to find the bed",
            "navigation_task_type": "point_goal",
            "cot": None,
            "previous_assessment": {
                "execution_status": "not_started",
                "task_effect": "unknown",
                "evidence": "The connected route has not started.",
            },
            "route_hypothesis": {
                "statement": "Follow the connected topology to the bedroom.",
                "confidence": 0.9,
            },
            "candidates": [
                {
                    "action_type": "known_location",
                    "target": "bedroom",
                    "task_relevance": 1.0,
                    "information_gain": 0.8,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "known_location",
                "target": "bedroom",
                "topology_node_id": "backbone-002",
                "expected_evidence": "Reach the bedroom region.",
                "success_condition": "Runtime confirms bedroom arrival.",
                "fallback": "Reassess the connected route.",
            },
            "semantic_map": semantic_map(),
        }
    )
    constrained_action, policy_override = service._apply_navigation_policy(
        skip_request, observation
    )
    assert constrained_action.topology_node_id == "backbone-001"
    assert policy_override["reason"] == "connected_topology_step_enforced"
    guidance = service._semantic_route_plan.regional_guidance
    service._semantic_route_plan = service._semantic_route_plan.model_copy(
        update={
            "regional_guidance": guidance.model_copy(
                update={
                    "allowed_anchor_views": ["front"],
                    "direction_constraint_active": True,
                    "recommended_viewpoint_action": None,
                }
            )
        }
    )
    anchor_payload = skip_request.model_dump(mode="json")
    anchor_payload["candidates"] = [
        {
            "action_type": "visible_anchor",
            "intent": "explore_route",
            "target": "open doorway",
            "task_relevance": 0.9,
            "information_gain": 0.8,
            "revisit_penalty": 0.0,
        }
    ]
    anchor_payload["action"] = {
        "action_type": "visible_anchor",
        "intent": "goal_pursuit",
        "target": "open doorway",
        "view": "front",
        "anchor_role": "affordance",
        "expected_evidence": "The connected route remains visible.",
        "success_condition": "The route advances toward the next node.",
        "fallback": "Use the next connected topology node.",
    }
    anchor_request = AdvanceNavigationInput.model_validate(anchor_payload)
    constrained_anchor, anchor_override = service._semantic_route_policy(
        anchor_request, observation
    )
    assert constrained_anchor.action_type == "known_location"
    assert constrained_anchor.topology_node_id == "backbone-001"
    assert anchor_override["constraint"] == "point_goal_topology_priority"
    target_payload = dict(anchor_payload)
    target_payload["action"] = {
        **anchor_payload["action"],
        "target": "directly visible bed",
        "anchor_role": "target",
    }
    target_payload["candidates"] = [
        {
            **anchor_payload["candidates"][0],
            "target": "directly visible bed",
            "intent": "goal_pursuit",
        }
    ]
    target_request = AdvanceNavigationInput.model_validate(target_payload)
    assert service._semantic_route_policy(target_request, observation) is None
    _, scope = service._anchor_region_context(
        {"world_point_xyz_m": [0.0, 0.0, 2.0], "confidence": 0.95}
    )
    assert scope["active_semantic_node_id"] == "backbone-002"
    assert scope["regional_context_node_id"] == "backbone-001"

    prematurely_at_target = observation.model_copy(
        update={
            "pose": Pose2D(
                x=1.4,
                y=0.0,
                yaw=0.0,
                source="test",
                frame="habitat_world_xz",
            )
        }
    )
    service._refresh_regional_guidance(prematurely_at_target)
    guidance = service._semantic_route_plan.regional_guidance
    assert guidance is not None
    assert guidance.distance_to_target_node_m == 0.0
    assert guidance.route_progress_index == 0
    assert guidance.next_topology_node_id == "backbone-001"
    assert guidance.target_region_reached is False
    assert guidance.mode == "route_to_region"

    at_connected_middle = observation.model_copy(
        update={
            "pose": Pose2D(
                x=0.0,
                y=4.0,
                yaw=0.0,
                source="test",
                frame="habitat_world_xz",
            )
        }
    )
    service._refresh_regional_guidance(at_connected_middle)
    guidance = service._semantic_route_plan.regional_guidance
    assert guidance is not None
    assert guidance.route_progress_index == 1
    assert guidance.current_topology_node_id == "backbone-001"
    assert guidance.next_topology_node_id == "backbone-002"
    assert guidance.target_region_reached is False

    service._refresh_regional_guidance(prematurely_at_target)
    guidance = service._semantic_route_plan.regional_guidance
    assert guidance is not None
    assert guidance.route_progress_index == 2
    assert guidance.target_region_reached is True
    assert guidance.mode == "search_within_region"
    assert service._semantic_route_plan.phase == "local_anchor_search"


def test_regional_route_turns_for_hidden_direction_but_does_not_block_visible_goal(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="hidden-direction-guidance",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=0.0,
                y=0.0,
                row=0,
                column=0,
                clearance_m=1.0,
                navigation_point_xyz_m=[0.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=0.0,
                y=3.0,
                row=0,
                column=1,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, 3.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-000",
                target="backbone-001",
                distance_m=3.0,
                path_world_xy_m=[(0.0, 0.0), (0.0, 1.5), (0.0, 3.0)],
            )
        ],
    )
    service._topology_model = topology
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-000": NodeSemantics(place_type="office", source="human"),
                "backbone-001": NodeSemantics(
                    place_type="bedroom",
                    objects=["bed"],
                    source="human",
                ),
            },
        )
    )
    started = service.begin_navigation(
        {
            "global_goal": "go to bedroom to find the bed",
            "destination_query": "bedroom",
            "object_goal": "bed",
        }
    )
    observation = service._observations[started["observation"]["trace"]["observation_id"]]
    base_request = {
        "observation_id": observation.trace.observation_id,
        "global_goal": "go to bedroom to find the bed",
        "previous_assessment": {
            "execution_status": "not_started",
            "task_effect": "unknown",
            "evidence": "The target region is behind the current camera coverage.",
        },
        "route_hypothesis": {
            "statement": "Follow the regional route before searching the bedroom.",
            "confidence": 0.9,
        },
        "semantic_map": semantic_map(),
    }
    exploratory_request = AdvanceNavigationInput.model_validate(
        {
            **base_request,
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "intent": "explore_route",
                    "target": "open passage",
                    "task_relevance": 0.7,
                    "information_gain": 0.8,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "intent": "explore_route",
                "target": "open passage",
                "view": "front",
                "expected_evidence": "The next regional route segment.",
                "success_condition": "Advance toward the bedroom region.",
                "fallback": "Turn to recover the route direction.",
            },
        }
    )

    action, policy_override = service._apply_navigation_policy(
        exploratory_request, observation
    )

    assert action.action_type == "viewpoint_adjustment"
    assert action.viewpoint_action == "turn_left"
    assert policy_override["reason"] == "regional_route_outside_current_views"

    visible_goal_request = AdvanceNavigationInput.model_validate(
        {
            **base_request,
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "intent": "goal_pursuit",
                    "target": "bed",
                    "task_relevance": 1.0,
                    "information_gain": 1.0,
                    "revisit_penalty": 0.0,
                }
            ],
                "action": {
                    "action_type": "visible_anchor",
                    "intent": "goal_pursuit",
                    "target": "bed",
                    "view": "front",
                    "anchor_role": "target",
                    "expected_evidence": "The bed remains directly visible.",
                "success_condition": "Ground the visible bed for task verification.",
                "fallback": "Reassess the current visual evidence.",
            },
        }
    )

    action, policy_override = service._apply_navigation_policy(visible_goal_request, observation)

    assert action == visible_goal_request.action
    assert policy_override is None


def test_unproductive_route_exploration_keeps_model_selected_action(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    started = service.begin_navigation()
    observation = service._observations[started["observation"]["trace"]["observation_id"]]
    semantics = SemanticMapUpdate.model_validate(semantic_map("task start"))
    service.navigation_memory.record_place(observation, semantics)
    adapter.execution_calls = 5
    intermediate = asyncio.run(
        adapter.observe(make_trace(observation_id="before-unproductive-branch"))
    )
    service.navigation_memory.record_place(
        intermediate,
        SemanticMapUpdate.model_validate(semantic_map("last useful route point")),
    )
    adapter.execution_calls = 10
    current = asyncio.run(adapter.observe(make_trace(observation_id="after-unproductive-branch")))
    request = AdvanceNavigationInput.model_validate(
        {
            "observation_id": current.trace.observation_id,
            "global_goal": "find the bedroom",
            "previous_assessment": {
                "execution_status": "completed",
                "task_effect": "no_progress",
                "evidence": "The explored corridor revealed no bedroom evidence.",
            },
            "route_hypothesis": {
                "statement": "Try another visible corridor.",
                "confidence": 0.4,
            },
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "intent": "explore_route",
                    "target": "far corridor floor",
                    "task_relevance": 0.2,
                    "information_gain": 0.5,
                    "revisit_penalty": 0.8,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "intent": "explore_route",
                "target": "far corridor floor",
                "view": "front",
                "expected_evidence": "A new room entrance.",
                "success_condition": "Reach the end of the corridor.",
                "fallback": "Return to the previous junction.",
            },
            "semantic_map": semantic_map("dead-end corridor"),
        }
    )

    action, policy_override = service._apply_navigation_policy(request, current)

    assert action == request.action
    assert policy_override is None


def test_known_location_zero_motion_is_only_location_arrival(tmp_path, make_trace):
    class AlreadyThereAdapter(FakeHierarchicalAdapter):
        async def plan_path(self, world_point):
            return {
                "requested_goal_xyz_m": world_point,
                "snapped_goal_xyz_m": world_point,
                "world_path_xyz_m": [world_point],
                "robot_path_forward_left_m": [[0.0, 0.0]],
                "geodesic_distance_m": 0.0,
                "waypoint_count": 1,
            }

        async def follow_navmesh_path_step(self, world_path, **kwargs):
            return {
                "reached": True,
                "remaining_distance_m": 0.0,
                "stalled": False,
                "pose_after": {"x": 0.0, "y": 0.0, "yaw": self.yaw},
                "control": None,
            }

    adapter = AlreadyThereAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    target_observation = asyncio.run(
        adapter.observe(make_trace(observation_id="already-there"))
    )
    service.navigation_memory.record_place(
        target_observation,
        SemanticMapUpdate.model_validate(semantic_map("current room")),
    )
    observed = service.observe()["observation"]

    result = service.navigate_to_known_location(
        {
            "location_id": "location-001",
            "observation_id": observed["trace"]["observation_id"],
            "semantic_map": semantic_map("current room"),
        }
    )

    assert result["result"]["reached"] is True
    assert result["result"]["control_steps"] == 0
    assert result["result"]["motion_performed"] is False
    assert result["result"]["completion_scope"] == "location_arrival_only"
    assert result["result"]["no_motion_reason"] == "already_at_target_location"


def test_unproductive_route_respects_explicit_viewpoint_adjustment(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    started = service.begin_navigation()
    observation = service._observations[started["observation"]["trace"]["observation_id"]]
    semantics = SemanticMapUpdate.model_validate(semantic_map("task start"))
    service.navigation_memory.record_place(observation, semantics)
    adapter.execution_calls = 10
    current = asyncio.run(adapter.observe(make_trace(observation_id="after-hidden-view")))
    request = AdvanceNavigationInput.model_validate(
        {
            "observation_id": current.trace.observation_id,
            "global_goal": "find the bedroom",
            "previous_assessment": {
                "execution_status": "completed",
                "task_effect": "no_progress",
                "evidence": "The target may be outside the current three-view coverage.",
            },
            "route_hypothesis": {
                "statement": "A 90-degree left turn can reveal the unobserved side.",
                "confidence": 0.8,
            },
            "candidates": [
                {
                    "action_type": "viewpoint_adjustment",
                    "intent": "explore_route",
                    "target": "unobserved left-side view",
                    "task_relevance": 0.8,
                    "information_gain": 0.9,
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {
                "action_type": "viewpoint_adjustment",
                "intent": "explore_route",
                "target": "unobserved left-side view",
                "viewpoint_action": "turn_left",
                "expected_evidence": "A previously unseen room side.",
                "success_condition": "Capture views after a 90-degree left turn.",
                "fallback": "Revisit the strongest historical evidence location.",
            },
            "semantic_map": semantic_map("target absent from current views"),
        }
    )

    action, policy_override = service._apply_navigation_policy(request, current)

    assert action == request.action
    assert policy_override is None


def test_low_confidence_anchor_adjusts_view_without_point_navigation(tmp_path, make_trace):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGenerationSkill(lambda: FakeProvider(calls, confidence=0.75), adapter))
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    observed = service.observe()["observation"]

    result = service.navigate_to_anchor(
        {
            "anchor": "unclear doorway",
            "view": "front",
            "observation_id": observed["trace"]["observation_id"],
            "semantic_map": semantic_map(),
            "low_confidence_recovery": "adjust_viewpoint",
            "recovery_viewpoint_action": "turn_right",
        }
    )

    assert result["status"] == "incomplete"
    assert result["phase"] == "review_low_confidence_recovery"
    assert result["anchor_rejected"] is True
    assert result["grounding_confidence"] == 0.75
    assert result["grounding_confidence_threshold"] == 0.8
    assert result["recovery"] == {
        "requested": "adjust_viewpoint",
        "executed": "adjust_viewpoint",
        "target": "turn_right",
    }
    assert adapter.execution_calls == 0
    assert adapter.viewpoint_actions == ["turn_right"] * 6
    assert result["next_observation_id"] == "hierarchical-obs-1"


def test_low_confidence_anchor_can_return_to_previous_route_location(tmp_path, make_trace):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGenerationSkill(lambda: FakeProvider(calls, confidence=0.4), adapter))
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    previous = asyncio.run(adapter.observe(make_trace(observation_id="previous"))).model_copy(
        update={
            "pose": Pose2D(x=0.0, y=-2.0, yaw=0.0, source="test", frame="habitat_world_xz"),
            "metadata": {
                "metric_position_xyz": [0.0, 0.0, -2.0],
                "map_id": "test-map",
            },
        }
    )
    current = asyncio.run(adapter.observe(make_trace(observation_id="current")))
    service.navigation_memory.record_place(
        previous,
        SemanticMapUpdate.model_validate(semantic_map("previous corridor")),
    )
    service.navigation_memory.record_place(
        current,
        SemanticMapUpdate.model_validate(semantic_map("uncertain branch")),
    )
    observed = service.observe()["observation"]

    result = service.navigate_to_anchor(
        {
            "anchor": "uncertain opening",
            "view": "front",
            "observation_id": observed["trace"]["observation_id"],
            "semantic_map": semantic_map("uncertain branch"),
            "low_confidence_recovery": "previous_location",
            "max_control_steps": 24,
        }
    )

    assert result["status"] == "incomplete"
    assert result["recovery"] == {
        "requested": "previous_location",
        "executed": "previous_location",
        "target": "location-001",
    }
    assert result["result"]["reached"] is True
    assert adapter.execution_calls == 10
    assert result["trajectory_memory"]


def test_viewpoint_adjustment_rotates_90_degrees_and_returns_fresh_three_views(
    tmp_path, make_trace
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    observed = service.observe()["observation"]

    result = service.adjust_viewpoint(
        {
            "observation_id": observed["trace"]["observation_id"],
            "action": "turn_left",
            "semantic_map": semantic_map(),
        }
    )

    assert adapter.viewpoint_actions == ["turn_left"] * 6
    assert result["result"]["action"] == "turn_left"
    assert result["result"]["control_steps"] == 6
    assert round(result["result"]["rotation_degrees"], 1) == 90.0
    assert result["next_observation_id"] == "hierarchical-obs-1"
    assert set(result["post_observation"]["rgb_refs"]) == {"left", "front", "right"}
    try:
        service.adjust_viewpoint(
            {
                "observation_id": observed["trace"]["observation_id"],
                "action": "turn_right",
                "semantic_map": semantic_map(),
            }
        )
    except ValueError as exc:
        assert "stale observation_id" in str(exc)
    else:
        raise AssertionError("stale viewpoint observation should fail")


def test_scene_memory_version_records_and_clears_task_memory(tmp_path, make_trace):
    calls = []
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(AnchorGroundingSkill(lambda: FakeProvider(calls)))
    skills.register(PixelToPointSkill(adapter))
    skills.register(PathPlanningSkill(adapter))
    skills.register(PathExecutionSkill(adapter))
    skills.register(AnchorGenerationSkill(lambda: FakeProvider(calls), adapter))
    skills.register(PointNavigationSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills, scene_memory_enabled=True)
    first_task_id = service.task_id
    scene = {
        "room_hint": "white kitchen hallway",
        "left": ["white cabinets"],
        "front": ["open hallway"],
        "right": ["robot equipment"],
        "distinctive_landmarks": ["washing machine"],
    }

    result = service.navigate_to_anchor(
        {
            "anchor": "open hallway",
            "view": "front",
            "semantic_map": semantic_map(),
            "scene_memory": scene,
            "max_control_steps": 12,
        }
    )

    assert result["scene_memory_update"]["scene_id"] == "scene-001"
    assert result["scene_memory_update"]["revisited"] is False
    assert len(result["scene_memory_index"]) == 1
    finished = service.finish_task(
        {
            "outcome": "done",
            "observation_id": result["next_observation_id"],
            "scene_memory": scene,
        }
    )

    assert finished["finished_task_id"] == first_task_id
    assert finished["memory_cleared"] is True
    assert finished["memory_summary"]["unique_scenes"] == 2
    assert finished["memory_summary"]["total_visits"] == 2
    assert service.task_id != first_task_id
    assert service.health()["scene_memory_count"] == 0


def test_finish_verifier_can_reject_global_completion_without_clearing_memory(
    tmp_path, make_trace
):
    class RejectingFinishProvider:
        async def complete(self, messages, *, response_schema=None):
            audit = _view_audit_response(messages, response_schema)
            if audit is not None:
                return audit
            return LLMResponse(
                model="fake-finish-verifier",
                content="{}",
                raw={},
                parsed_json={
                    "evidence_satisfies_task": False,
                    "terminal_failure_supported": False,
                    "observed_evidence": ["Current views show only a corridor."],
                },
            )

        async def close(self):
            return None

    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(
        adapter,
        SkillRegistry(),
        finish_verifier_provider_factory=RejectingFinishProvider,
    )
    observed = service.observe()["observation"]
    original_task_id = service.task_id

    result = service.finish_task(
        {
            "outcome": "done",
            "observation_id": observed["trace"]["observation_id"],
        }
    )

    assert result["status"] == "finish_rejected"
    assert result["memory_cleared"] is False
    assert result["finish_verification"]["recommended_action"] == "replan"
    assert service.task_id == original_task_id


def test_active_agentic_task_cannot_skip_finish_transition(tmp_path, make_trace):
    calls = []

    class AcceptingFinishProvider:
        async def complete(self, messages, *, response_schema=None):
            audit = _view_audit_response(messages, response_schema)
            if audit is not None:
                return audit
            calls.append(messages)
            return LLMResponse(
                model="fake-finish-verifier",
                content="{}",
                raw={},
                parsed_json={
                    "evidence_satisfies_task": True,
                    "terminal_failure_supported": False,
                    "observed_evidence": ["A target is visible."],
                },
            )

        async def close(self):
            return None

    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(
        adapter,
        SkillRegistry(),
        finish_verifier_provider_factory=AcceptingFinishProvider,
    )
    started = service.begin_navigation({"global_goal": "find the target"})
    original_task_id = service.task_id

    result = service.finish_task(
        {
            "outcome": "done",
            "observation_id": started["observation"]["trace"]["observation_id"],
        }
    )

    assert result["status"] == "finish_not_ready"
    assert result["memory_cleared"] is False
    assert result["finish_verification"]["status"] == "not_requested"
    assert result["next_call_contract"]["next_tool"] == "advance_navigation"
    assert calls == []
    assert service.task_id == original_task_id


def test_finish_verifier_rejects_authorization_without_evidence(tmp_path, make_trace):
    class EvidenceFreeFinishProvider:
        async def complete(self, messages, *, response_schema=None):
            audit = _view_audit_response(messages, response_schema)
            if audit is not None:
                return audit
            return LLMResponse(
                model="fake-finish-verifier",
                content="{}",
                raw={},
                parsed_json={
                    "evidence_satisfies_task": True,
                    "terminal_failure_supported": False,
                    "observed_evidence": [],
                },
            )

        async def close(self):
            return None

    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(
        adapter,
        SkillRegistry(),
        finish_verifier_provider_factory=EvidenceFreeFinishProvider,
    )
    observed = service.observe()["observation"]
    original_task_id = service.task_id

    result = service.finish_task(
        {"outcome": "done", "observation_id": observed["trace"]["observation_id"]}
    )

    assert result["status"] == "finish_rejected"
    assert result["finish_verification"]["observed_evidence"][0].startswith(
        "Finish Verifier protocol error:"
    )
    assert service.task_id == original_task_id


def test_protocol_error_returns_authoritative_next_call_contract(tmp_path, make_trace):
    service = HierarchicalNavigationService(
        FakeHierarchicalAdapter(tmp_path, make_trace), SkillRegistry()
    )
    started = service.begin_navigation({"global_goal": "find the target"})

    result = service.protocol_error_payload(
        ValueError("the first Slow decision must include planner_output")
    )

    assert result["ok"] is False
    assert result["next_call_contract"]["next_tool"] == "advance_navigation"
    assert result["next_call_contract"]["planner_output"] == "required"
    assert result["next_call_contract"]["observation_id"] == (
        started["observation"]["trace"]["observation_id"]
    )


def test_successful_finish_publishes_terminal_relay_state(tmp_path, make_trace):
    class AcceptingFinishProvider:
        async def complete(self, messages, *, response_schema=None):
            audit = _view_audit_response(messages, response_schema)
            if audit is not None:
                return audit
            return LLMResponse(
                model="fake-finish-verifier",
                content="{}",
                raw={},
                parsed_json={
                    "evidence_satisfies_task": True,
                    "terminal_failure_supported": False,
                    "observed_evidence": ["Current front view contains the target."],
                },
            )

        async def close(self):
            return None

    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(ViewpointAdjustmentSkill(adapter))
    service = HierarchicalNavigationService(
        adapter,
        skills,
        finish_verifier_provider_factory=AcceptingFinishProvider,
    )
    started = service.begin_navigation({"global_goal": "find the target"})
    first = service.advance_navigation(
        viewpoint_decision(
            started["observation"]["trace"]["observation_id"],
            task_type="object_goal",
            planner_output={
                "global_instruction": "find the target",
                "destination_query": "target region",
                "object_goal": "target object",
                "subtasks": [
                    {
                        "stage": "reaching",
                        "task_type": "object_goal",
                        "instruction": "Find the target object.",
                        "target": "target object",
                        "success_condition": "The target object is confirmed.",
                    }
                ],
            },
        )
    )
    ready = service.advance_navigation(
        viewpoint_decision(
            first["next_observation_id"],
            task_type="object_goal",
            reflection={
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The target object is visually confirmed.",
                "should_replan": False,
            },
        )
    )

    result = service.finish_task(
        {"outcome": "done", "observation_id": ready["next_observation_id"]}
    )
    relay = service.relay_state()["navigation_task"]

    assert result["ok"] is True
    assert relay["status"] == "done"
    assert relay["phase"] == "done"
    assert relay["instruction"] == "find the target"
    assert relay["finish_required"] is False


def test_agentos_relay_keeps_latest_visual_anchor_pipeline_fields(
    tmp_path,
    make_trace,
):
    service = HierarchicalNavigationService(
        FakeHierarchicalAdapter(tmp_path, make_trace),
        SkillRegistry(),
    )
    service._latest_pipeline = {
        "status": "completed",
        "anchor_annotation_ref": "/logs/anchor-bed.jpg",
        "generated_point": {
            "target_label": "bed",
            "target_view": "right",
            "target_confidence": 0.9,
        },
    }
    service._agent_task_state = {
        "status": "running",
        "task_id": "agent-task-test",
        "instruction": "go to bedroom to find a bed",
    }

    relay = service.relay_state()["navigation_task"]

    assert relay["status"] == "running"
    assert relay["instruction"] == "go to bedroom to find a bed"
    assert relay["anchor_annotation_ref"] == "/logs/anchor-bed.jpg"
    assert relay["generated_point"]["target_label"] == "bed"


def test_pixel_annotation_always_matches_the_executed_goal(
    tmp_path,
    make_trace,
    monkeypatch,
):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skills = SkillRegistry()
    skills.register(PixelToPointSkill(adapter))
    service = HierarchicalNavigationService(adapter, skills)
    observation = service.observe()["observation"]
    captured = []

    def annotate(image_ref, anchor, run_id, **kwargs):
        captured.append((image_ref, anchor, run_id, kwargs))
        return str(tmp_path / "executed-target.jpg")

    monkeypatch.setattr(AnchorGroundingSkill, "_annotate_anchor", annotate)
    request = HierarchicalNavigateInput.model_validate(
        {
            "slow_step_id": "slow-test",
            "anchor": "white refrigerator",
            "anchor_role": "target",
            "view": "front",
            "observation_id": observation["trace"]["observation_id"],
            "semantic_map": semantic_map(),
            "target_pixel": [320, 360],
            "target_bbox_px": [280, 200, 360, 360],
            "target_pixel_confidence": 0.95,
            "target_pixel_evidence": "The refrigerator is visible in front.",
            "n1_target_view": "front",
            "n1_target_pixel": [320, 360],
            "n1_target_label": "white refrigerator",
            "n1_target_bbox_px": [280, 200, 360, 360],
            "n1_target_confidence": 0.95,
            "n1_affordance_view": "right",
            "n1_affordance_pixel": [500, 400],
            "n1_affordance_label": "open passage",
            "n1_affordance_bbox_px": [430, 250, 570, 400],
            "n1_affordance_confidence": 0.91,
        }
    )

    result = service._project_slow_brain_target_pixel(
        request,
        service._observations[observation["trace"]["observation_id"]],
    )["result"]["outputs"]

    assert len(captured) == 1
    assert captured[0][1].anchor == "white refrigerator"
    assert captured[0][1].view == "front"
    assert captured[0][3]["primary_role"] == "target"
    assert "secondary_target" not in captured[0][3]
    assert "secondary_affordance" not in captured[0][3]
    assert result["annotation_role"] == "target"
    assert result["annotation_view"] == "front"


def test_compact_validation_error_does_not_echo_large_inputs():
    try:
        AdvanceNavigationInput.model_validate({"unexpected": "x" * 20_000})
    except ValueError as exc:
        summary = _compact_error_message(exc)
    else:
        raise AssertionError("invalid advance input must fail validation")

    assert len(summary) < 800
    assert "20000" not in summary
    assert "unexpected" in summary or "Field required" in summary


def test_pixel_to_point_rejects_stale_observation(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    skill = PixelToPointSkill(adapter)
    observation = asyncio.run(adapter.observe(make_trace(observation_id="obs-current")))

    from agentos.core.events import InMemoryEventBus
    from agentos.schemas.skill import SkillRequest
    from agentos.skills.base import CancellationToken, SkillContext

    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="project",
        inputs={
            "source_observation_id": "obs-stale",
            "view": "front",
            "x_px": 320,
            "y_px": 360,
        },
    )
    context = SkillContext(InMemoryEventBus(), CancellationToken())

    try:
        asyncio.run(skill.run(request, observation, context))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("stale projection input should fail")


def test_pixel_to_point_limits_far_depth_along_original_ray(tmp_path, make_trace):
    class FarDepthAdapter(FakeHierarchicalAdapter):
        async def pixel_to_world(self, **kwargs):
            return {
                "view": "front",
                "pixel": [320, 240],
                "depth_pixel": [320, 240],
                "depth_m": 12.0,
                "camera_point_right_up_forward_m": [0.0, 0.0, 12.0],
                "robot_point_forward_left_m": [12.0, 0.0],
                "world_point_xyz_m": [0.0, 0.0, -12.0],
                "navigation_candidate_xyz_m": [0.0, 0.0, -12.0],
            }

    adapter = FarDepthAdapter(tmp_path, make_trace)
    skill = PixelToPointSkill(adapter)
    observation = asyncio.run(adapter.observe(make_trace(observation_id="obs-current")))
    request = SkillRequest(
        trace=observation.trace,
        skill_name=skill.name,
        command="project",
        inputs={
            "source_observation_id": "obs-current",
            "view": "front",
            "x_px": 320,
            "y_px": 240,
            "max_depth_m": 5.0,
        },
    )

    result = asyncio.run(
        skill.run(request, observation, SkillContext(InMemoryEventBus(), CancellationToken()))
    )

    assert result.outputs["raw_depth_m"] == 12.0
    assert result.outputs["depth_m"] == 5.0
    assert result.outputs["depth_limited"] is True
    assert result.outputs["world_point_xyz_m"] == [0.0, 0.0, -5.0]
    assert result.outputs["navigation_candidate_xyz_m"] == [0.0, 0.0, -5.0]


def test_semantic_route_uses_node_as_region_without_clipping_anchor(tmp_path, make_trace):
    adapter = FakeHierarchicalAdapter(tmp_path, make_trace)
    service = HierarchicalNavigationService(adapter, SkillRegistry())
    topology = BackboneTopology(
        map_id="test-map",
        geometry_revision="test-v1",
        source="habitat_navmesh",
        frame="habitat_world_xz",
        cache_key="semantic-route-test",
        nodes=[
            BackboneNode(
                node_id="backbone-003",
                node_type="open_space_center",
                x=0.0,
                y=-2.0,
                row=0,
                column=0,
                clearance_m=1.2,
                navigation_point_xyz_m=[0.0, 0.0, -2.0],
            )
        ],
    )
    service._topology_model = topology
    service.navigation_memory.record_backbone_topology(topology)
    service.navigation_memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-map",
            geometry_revision="test-v1",
            nodes={
                "backbone-003": NodeSemantics(
                    place_type="bedroom",
                    function="sleeping and resting",
                    objects=["bed"],
                    aliases=["sleeping room"],
                    source="human",
                )
            },
        )
    )

    started = service.begin_navigation(
        {
            "global_goal": "go to bedroom to find the bed",
            "destination_query": "bedroom",
            "object_goal": "bed",
            "node_local_radius_m": 5.0,
            "anchor_max_depth_m": 5.0,
        }
    )
    plan = started["semantic_route_plan"]

    assert plan["phase"] == "approach_node"
    assert plan["execution_strategy"] == "anchor_chain"
    assert plan["selected_topology_node_id"] == "backbone-003"
    assert "selected_location_id" not in plan
    assert plan["object_goal"] == "bed"

    action = SelectedNavigationAction(
        action_type="known_location",
        target="bedroom",
        topology_node_id="backbone-003",
        expected_evidence="bedroom observations",
        success_condition="reach the bedroom node",
        fallback="reassess another fixed node",
    )
    service._update_semantic_route_after_execution(action, {"result": {"reached": True}})
    assert service._semantic_route_plan is not None
    assert service._semantic_route_plan.phase == "approach_node"

    point, scope = service._anchor_region_context(
        {"world_point_xyz_m": [12.0, 0.0, -2.0], "confidence": 0.95}
    )
    assert scope["node_scope_limited"] is False
    assert scope["active_semantic_node_id"] == "backbone-003"
    assert scope["regional_context_node_id"] == "backbone-003"
    assert point["world_point_xyz_m"] == [12.0, 0.0, -2.0]

    evidence = service.navigation_memory.record_trajectory_keyframes(
        [
            {
                "frame_index": 4,
                "observation_id": "obs-evidence",
                "timestamp": "2026-08-18T00:00:04+00:00",
                "pose": {
                    "x": 10.0,
                    "y": -2.0,
                    "yaw": 1.57,
                    "source": "simulator_truth",
                    "frame": "habitat_world_xz",
                },
                "map_id": "test-map",
                "pose_confidence": 1.0,
                "navigation_point_xyz_m": [10.0, 0.0, -2.0],
                "rgb_refs": {
                    "left": "/evidence/left.jpg",
                    "front": "/evidence/front.jpg",
                    "right": "/evidence/right.jpg",
                },
            }
        ],
        source_target="bed seen while approaching bedroom node",
    )
    evidence_location_id = evidence[0]["location_id"]
    service._update_semantic_route_after_execution(
        SelectedNavigationAction(
            action_type="known_location",
            intent="backtrack",
            target="bed evidence position",
            location_id=evidence_location_id,
            expected_evidence="restore the bed view",
            success_condition="reach the evidence position and restore yaw",
            fallback="return to the selected bedroom node",
        ),
        {"result": {"reached": True}},
    )

    assert service._semantic_route_plan is not None
    assert service._semantic_route_plan.selected_topology_node_id == "backbone-003"
    assert service._semantic_route_plan.evidence_location_id == evidence_location_id
    point, scope = service._anchor_region_context(
        {"world_point_xyz_m": [18.0, 0.0, -2.0], "confidence": 0.95}
    )
    assert scope["active_semantic_node_id"] == "backbone-003"
    assert scope["active_evidence_location_id"] == evidence_location_id
    assert scope["regional_context_node_id"] == "backbone-003"
    assert point["world_point_xyz_m"] == [18.0, 0.0, -2.0]
