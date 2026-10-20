from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

from PIL import Image
import yaml


def _load_tools_module():
    path = Path(__file__).parents[1] / ".hermes/plugins/cee-hierarchical/tools.py"
    spec = importlib.util.spec_from_file_location("cee_hierarchical_tools_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_image_context_module():
    path = Path(__file__).parents[1] / ".hermes/plugins/cee-hierarchical/image_context.py"
    spec = importlib.util.spec_from_file_location("cee_hierarchical_image_context_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_request_normalization_module():
    path = Path(__file__).parents[1] / ".hermes/plugins/cee-hierarchical/request_normalization.py"
    spec = importlib.util.spec_from_file_location("cee_hierarchical_normalization_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_schema_module():
    path = Path(__file__).parents[1] / ".hermes/plugins/cee-hierarchical/schemas.py"
    spec = importlib.util.spec_from_file_location("cee_hierarchical_schema_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plugin_module():
    plugin_dir = Path(__file__).parents[1] / ".hermes/plugins/cee-hierarchical"
    spec = importlib.util.spec_from_file_location(
        "cee_hierarchical_plugin_test",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_observe_returns_three_images_to_task_agent(monkeypatch, tmp_path):
    tools = _load_tools_module()
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"{view}.jpg"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(path)
        refs[view] = str(path)
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda _path, _payload: {
            "ok": True,
            "observation": {
                "trace": {"observation_id": "obs-1"},
                "rgb_refs": refs,
                "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            },
        },
    )

    result = tools.observe({})

    assert result["_multimodal"] is True
    assert sum(item["type"] == "image_url" for item in result["content"]) == 3
    assert "obs-1" in result["content"][-1]["text"]
    assert json.loads(result["text_summary"])["observation_id"] == "obs-1"


def test_semantic_map_search_returns_known_location_without_images(monkeypatch):
    tools = _load_tools_module()
    calls = []

    def request(path, payload):
        calls.append((path, payload))
        return {
            "ok": True,
            "query": "bedroom",
            "count": 1,
            "results": [
                {
                    "location_id": "location-007",
                    "topology_node_id": "backbone-005",
                    "score": 0.98,
                    "matched_terms": ["bedroom"],
                    "node_semantics": {"place_type": "bedroom", "source": "human"},
                }
            ],
        }

    monkeypatch.setattr(tools, "_request_json", request)

    result = json.loads(tools.search_semantic_map({"query": "bedroom"}))

    assert calls == [("/semantic-map/search", {"query": "bedroom"})]
    assert result["results"][0]["topology_node_id"] == "backbone-005"
    assert "location_id" not in result["results"][0]
    assert "topology_node_id" in result["instruction"]


def test_tool_http_error_preserves_runtime_next_call_contract(monkeypatch):
    tools = _load_tools_module()
    body = json.dumps(
        {
            "ok": False,
            "error": "self_reflection references an inactive subtask",
            "next_call_contract": {
                "next_tool": "advance_navigation",
                "observation_id": "hierarchical-obs-9",
                "self_reflection_subtask_id": "subtask-002",
            },
        }
    ).encode()

    def fail(_request, timeout):
        raise HTTPError(
            url="http://127.0.0.1:8766/decision/advance",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(body),
        )

    monkeypatch.setattr(tools, "urlopen", fail)

    result = json.loads(tools._request("/decision/advance", {}))

    assert result["error"].startswith("HTTP 400:")
    assert result["next_call_contract"]["self_reflection_subtask_id"] == (
        "subtask-002"
    )


def test_begin_navigation_forwards_semantic_task_decomposition(monkeypatch, tmp_path):
    tools = _load_tools_module()
    calls = []
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"begin-{view}.jpg"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(path)
        refs[view] = str(path)

    def request(path, payload):
        calls.append((path, payload))
        return {
            "ok": True,
            "observation": {
                "trace": {"observation_id": "obs-begin"},
                "rgb_refs": refs,
            },
            "semantic_route_plan": {
                "phase": "approach_node",
                "selected_topology_node_id": "backbone-005",
                "object_goal": "bed",
            },
            "memory_view": {
                "nearby_backbone_nodes": [
                    {
                        "location_id": "location-007",
                        "topology_node_id": "backbone-005",
                        "scene_summary": "bedroom",
                    }
                ],
                "current_place": {"place_id": "location-001", "scene_summary": "start"},
            },
            "route_graph": {
                "nodes": [
                    {"location_id": "location-001", "topology_node_id": None},
                    {
                        "location_id": "location-006",
                        "topology_node_id": "backbone-004",
                    },
                    {
                        "location_id": "location-007",
                        "topology_node_id": "backbone-005",
                    },
                ],
                "edges": [
                    {"source": "location-001", "target": "location-006"},
                    {
                        "source": "location-006",
                        "target": "location-007",
                        "path_world_xy_m": [[0.0, 0.0], [1.0, 1.0]],
                    },
                ],
            },
            "topology": {
                "loaded": True,
                "node_count": 8,
                "node_locations": {"backbone-005": "location-007"},
            },
            "next_call_contract": {
                "next_tool": "advance_navigation",
                "observation_id": "obs-begin",
                "planner_output": "required",
                "self_reflection": "null",
                "self_reflection_subtask_id": None,
                "navigation_task_type": None,
                "active_target_evidence": False,
            },
        }

    decomposition = {
        "global_goal": "go to bedroom to find the bed",
        "destination_query": "bedroom",
        "object_goal": "bed",
        "node_local_radius_m": 5.0,
        "anchor_max_depth_m": 5.0,
    }
    monkeypatch.setattr(tools, "_request_json", request)

    result = tools.begin_navigation(decomposition)

    assert calls == [("/decision/begin", decomposition)]
    summary = json.loads(result["text_summary"])
    assert summary["semantic_route_plan"]["selected_topology_node_id"] == "backbone-005"
    assert "selected_location_id" not in summary["semantic_route_plan"]
    assert summary["next_call_contract"]["planner_output"] == "required"
    assert summary["memory_view"]["nearby_backbone_nodes"][0] == {
        "topology_node_id": "backbone-005",
        "scene_summary": "bedroom",
    }
    assert summary["memory_view"]["current_place"]["location_id"] == "location-001"
    assert summary["route_graph"]["edges"] == [
        {"source": "location-001", "target": "backbone-004"},
        {"source": "backbone-004", "target": "backbone-005"},
    ]
    assert "node_locations" not in summary["topology"]
    assert "location-007" not in result["text_summary"]
    assert "allowed_anchor_views" in summary["instruction"]
    assert "target_region_reached=true" in summary["instruction"]
    assert "global_visual_anchor" not in summary
    assert "global_anchor" not in summary["memory_view"]


def test_advance_returns_bounded_memory_four_keyframes_and_fresh_views(monkeypatch, tmp_path):
    tools = _load_tools_module()
    annotation = tmp_path / "anchor.jpg"
    Image.new("RGB", (8, 6), color=(200, 20, 20)).save(annotation)
    keyframes = []
    for index in (0, 3, 7, 10):
        path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (8, 6), color=(index, 40, 50)).save(path)
        keyframes.append(
            {
                "frame_index": index,
                "pose": {"x": 0.0, "y": float(index), "yaw": 0.0},
                "location_id": f"location-{index + 2:03d}",
                "navigation_point_xyz_m": [0.0, 0.0, float(index)],
                "rgb_refs": {"front": str(path)},
            }
        )
    post_refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"post-{view}.jpg"
        Image.new("RGB", (8, 6), color=(20, 120, 40)).save(path)
        post_refs[view] = str(path)
    semantic_map_ref = tmp_path / "semantic-map.png"
    Image.new("RGB", (8, 12), color=(30, 60, 80)).save(semantic_map_ref)
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda _path, _payload: {
            "ok": True,
            "status": "completed",
            "anchor": "doorway",
            "view": "front",
            "decision": {
                "action_type": "visible_anchor",
                "target": "doorway",
                "view": "front",
                "expected_evidence": "room beyond doorway",
                "success_condition": "doorway is crossed",
                "fallback": "return to corridor",
            },
            "memory_delta": {"revision": 2, "created": True},
            "memory_view": {"revision": 2, "relevant_places": [], "total_places": 1},
            "anchor_annotation_ref": str(annotation),
            "next_observation_id": "obs-2",
            "post_observation": {
                "trace": {"observation_id": "obs-2"},
                "rgb_refs": post_refs,
                "topdown_ref": str(semantic_map_ref),
                "pose": {"x": 0.0, "y": 1.8, "yaw": 0.0},
            },
            "result": {
                "controller": "habitat_greedy_follower",
                "reached": True,
                "control_steps": 10,
                "distance_m": 1.8,
                "remaining_distance_m": 0.2,
                "collisions": 0,
                "stalled": False,
                "trajectory_keyframes": keyframes,
            },
        },
    )

    result = tools.advance_navigation({"action": {"action_type": "visible_anchor"}})

    assert result["_multimodal"] is True
    assert sum(item["type"] == "image_url" for item in result["content"]) == 7
    assert "habitat_greedy_follower" in result["content"][0]["text"]
    assert json.loads(result["text_summary"])["control_steps"] == 10
    assert json.loads(result["text_summary"])["next_observation_id"] == "obs-2"
    assert json.loads(result["text_summary"])["memory_view"]["total_places"] == 1
    assert json.loads(result["text_summary"])["executed_action"] == {
        "mode": "VISUAL_ANCHOR_NAV",
        "target": "doorway",
        "target_id": "doorway",
        "view": "front",
        "intent": None,
        "selection_source": "agent_selected",
        "override_reason": None,
    }
    assert not any("Qwen-selected anchor pixel" in item.get("text", "") for item in result["content"])
    assert all(
        item.get("image_url", {}).get("url", "").startswith("cee-local-image:")
        for item in result["content"]
        if item["type"] == "image_url"
    )
    assert len(json.dumps(result)) < 20_000
    assert "location-002" in result["content"][1]["text"]


def test_advance_ends_with_mandatory_next_tool_instruction(monkeypatch, tmp_path):
    tools = _load_tools_module()
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"mandatory-{view}.jpg"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(path)
        refs[view] = str(path)
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda _path, _payload: {
            "ok": True,
            "status": "completed",
            "decision": {"action_type": "viewpoint_adjustment"},
            "result": {},
            "next_call_contract": {
                "next_tool": "advance_navigation",
                "observation_id": "obs-2",
                "self_reflection": "required_complete_or_contradict_target_evidence",
            },
            "active_target_evidence": {"target": "white refrigerator"},
            "post_observation": {
                "trace": {"observation_id": "obs-2"},
                "rgb_refs": refs,
            },
        },
    )

    result = tools.advance_navigation({"observation_id": "obs-1"})

    final_text = result["content"][-1]["text"]
    assert "MANDATORY NEXT TOOL: advance_navigation" in final_text
    assert "Do not answer the user" in final_text


def test_executed_action_distinguishes_runtime_navigation_modes():
    tools = _load_tools_module()
    cases = [
        (
            {"action_type": "known_location", "topology_node_id": "backbone-002"},
            {"reason": "prefer_task_relevant_semantic_node"},
            ("NODE_NAV", "backbone-002", "runtime_override"),
        ),
        (
            {"action_type": "known_location", "location_id": "location-025"},
            {},
            ("L_REVISIT", "location-025", "agent_selected"),
        ),
        (
            {"action_type": "visible_anchor", "target": "wooden door", "view": "right"},
            {},
            ("VISUAL_ANCHOR_NAV", "wooden door", "agent_selected"),
        ),
        (
            {"action_type": "viewpoint_adjustment", "viewpoint_action": "turn_left"},
            {},
            ("VIEWPOINT_ADJUST", "turn_left", "agent_selected"),
        ),
    ]

    for decision, override, expected in cases:
        action = tools._executed_action(decision, override, {})
        assert (action["mode"], action["target_id"], action["selection_source"]) == expected

    recovery = tools._executed_action(
        {"action_type": "visible_anchor", "target": "unclear doorway"},
        {},
        {
            "anchor_rejected": True,
            "recovery": {
                "executed": "adjust_viewpoint",
                "target": "turn_right",
            },
        },
    )
    assert (recovery["mode"], recovery["target_id"], recovery["selection_source"]) == (
        "VIEWPOINT_ADJUST",
        "turn_right",
        "anchor_recovery",
    )


def test_runtime_binds_latest_observation_id_before_advance(monkeypatch, tmp_path):
    tools = _load_tools_module()
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"{view}.jpg"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(path)
        refs[view] = str(path)
    calls = []

    def request(path, payload):
        calls.append((path, dict(payload)))
        if path == "/decision/begin":
            return {
                "ok": True,
                "observation": {
                    "trace": {"observation_id": "obs-bound"},
                    "rgb_refs": refs,
                },
            }
        return {"ok": False, "error": "stop after payload capture"}

    monkeypatch.setattr(tools, "_request_json", request)
    tools.begin_navigation({})
    tools.advance_navigation({"observation_id": "stale", "action": {}})

    assert calls[-1] == (
        "/decision/advance",
        {"observation_id": "obs-bound", "action": {}},
    )


def test_adjust_viewpoint_returns_action_result_and_fresh_three_views(monkeypatch, tmp_path):
    tools = _load_tools_module()
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"adjusted-{view}.jpg"
        Image.new("RGB", (8, 6), color=(60, 70, 80)).save(path)
        refs[view] = str(path)
    topdown = tmp_path / "adjusted-semantic-map.png"
    Image.new("RGB", (8, 12), color=(30, 50, 70)).save(topdown)
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda path, payload: {
            "ok": True,
            "action": payload["action"],
            "next_observation_id": "obs-2",
            "result": {
                "collided": False,
                "distance_m": 0.0,
                "pose_before": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "pose_after": {"x": 0.0, "y": 0.0, "yaw": 0.26},
            },
            "post_observation": {
                "trace": {"observation_id": "obs-2"},
                "rgb_refs": refs,
                "topdown_ref": str(topdown),
                "pose": {"x": 0.0, "y": 0.0, "yaw": 0.26},
            },
        },
    )

    result = tools.adjust_viewpoint({"observation_id": "obs-1", "action": "turn_left"})

    assert result["_multimodal"] is True
    assert sum(item["type"] == "image_url" for item in result["content"]) == 4
    summary = json.loads(result["text_summary"])
    assert summary["action"] == "turn_left"
    assert summary["next_observation_id"] == "obs-2"


def test_localization_map_returns_semantic_map_and_fresh_three_views(monkeypatch, tmp_path):
    tools = _load_tools_module()
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"localized-{view}.jpg"
        Image.new("RGB", (8, 6), color=(90, 100, 110)).save(path)
        refs[view] = str(path)
    topdown = tmp_path / "topdown.png"
    Image.new("RGB", (8, 12), color=(40, 45, 40)).save(topdown)
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda _path, _payload: {
            "ok": True,
            "next_observation_id": "obs-map",
            "result": {
                "pose": {"x": 1.0, "y": 2.0, "yaw": 0.5},
                "map_frame": "habitat_world_xz",
                "legend": {"red": "current robot position"},
            },
            "exploration_memory": {
                "current_location_id": "location-003",
                "revisited": True,
                "total_explored_locations": 3,
                "recent_locations": [],
            },
            "observation": {
                "trace": {"observation_id": "obs-map"},
                "rgb_refs": refs,
                "topdown_ref": str(topdown),
                "pose": {"x": 1.0, "y": 2.0, "yaw": 0.5},
            },
        },
    )

    result = tools.inspect_localization_map(
        {
            "scene_summary": "corridor",
            "left": ["cabinet"],
            "front": ["doorway"],
            "right": [],
        }
    )

    assert result["_multimodal"] is True
    assert sum(item["type"] == "image_url" for item in result["content"]) == 4
    summary = json.loads(result["text_summary"])
    assert summary["exploration_memory"]["revisited"] is True
    assert summary["next_observation_id"] == "obs-map"


def test_known_location_navigation_returns_map_keyframes_and_fresh_views(monkeypatch, tmp_path):
    tools = _load_tools_module()
    topdown = tmp_path / "known-topdown.png"
    Image.new("RGB", (8, 12), color=(30, 40, 30)).save(topdown)
    keyframes = []
    for index in (0, 3, 7, 10):
        path = tmp_path / f"known-frame-{index}.jpg"
        Image.new("RGB", (8, 6), color=(index, 50, 60)).save(path)
        keyframes.append({"frame_index": index, "pose": {"x": 0.0}, "rgb_refs": {"front": str(path)}})
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"known-post-{view}.jpg"
        Image.new("RGB", (8, 6), color=(100, 110, 120)).save(path)
        refs[view] = str(path)
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda _path, _payload: {
            "ok": True,
            "status": "completed",
            "location_id": "location-002",
            "target_location": {"scene_summary": "main doorway", "semantic_labels": ["doorway"]},
            "next_observation_id": "obs-known",
            "result": {
                "controller": "habitat_greedy_follower",
                "reached": True,
                "control_steps": 10,
                "distance_m": 2.0,
                "remaining_distance_m": 0.1,
                "collisions": 0,
                "stalled": False,
                "trajectory_keyframes": keyframes,
            },
            "post_observation": {
                "trace": {"observation_id": "obs-known"},
                "rgb_refs": refs,
                "topdown_ref": str(topdown),
            },
            "exploration_memory": {"current_location_id": "location-002", "revisited": True},
        },
    )

    result = tools.navigate_known_location(
        {"location_id": "location-002", "observation_id": "obs-current"}
    )

    assert result["_multimodal"] is True
    assert sum(item["type"] == "image_url" for item in result["content"]) == 8
    summary = json.loads(result["text_summary"])
    assert summary["location_id"] == "location-002"
    assert summary["controller"] == "habitat_greedy_follower"
    assert summary["next_observation_id"] == "obs-known"


def test_hierarchical_hermes_declares_vision_and_only_exposes_task_boundary_tools():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/hierarchical_hermes.yaml").read_text())
    plugin = yaml.safe_load((root / ".hermes/plugins/cee-hierarchical/plugin.yaml").read_text())

    assert config["model"]["supports_vision"] is True
    assert config["model"]["max_tokens"] == 1536
    assert config["compression"]["enabled"] is False
    assert plugin["provides_tools"] == [
        "begin_navigation_task",
        "recall_navigation_memory",
        "advance_navigation",
        "finish_navigation_task",
    ]
    assert _load_schema_module().ADVANCE_NAVIGATION["parameters"]["required"] == [
        "action"
    ]
    properties = _load_schema_module().ADVANCE_NAVIGATION["parameters"]["properties"]
    assert _load_schema_module().ADVANCE_NAVIGATION["parameters"][
        "additionalProperties"
    ] is False
    assert properties["action"]["type"] == "object"
    assert properties["planner_output"]["type"] == ["object", "null"]
    assert "planner_output" in properties
    assert "previous_assessment" not in properties
    assert "route_hypothesis" not in properties
    assert "candidates" not in properties
    assert _load_schema_module().BEGIN_NAVIGATION_TASK["parameters"]["required"] == [
        "global_goal"
    ]
    assert len(json.dumps(_load_schema_module().ADVANCE_NAVIGATION)) < 5000


def test_hierarchical_plugin_does_not_register_internal_semantic_search():
    plugin = _load_plugin_module()

    class RegistrationContext:
        def __init__(self):
            self.tools = []

        def register_middleware(self, *_args, **_kwargs):
            return None

        def register_tool(self, *, name, **_kwargs):
            self.tools.append(name)

    context = RegistrationContext()
    plugin.register(context)

    assert context.tools == [
        "begin_navigation_task",
        "recall_navigation_memory",
        "advance_navigation",
        "finish_navigation_task",
    ]


def test_advance_context_keeps_road_but_bounds_detailed_topology_semantics():
    tools = _load_tools_module()
    function_layer = [
        {"topology_node_id": f"backbone-00{index}", "function": f"function-{index}"}
        for index in range(1, 4)
    ]
    object_layer = [
        {"topology_node_id": f"backbone-00{index}", "objects": [f"object-{index}"]}
        for index in range(1, 4)
    ]
    slow_context = {
        "current_observation_id": "obs-2",
        "historical_front_observation_ids": ["obs-1"],
        "task_specification": "find the refrigerator",
        "previous_decision": {"slow_step_id": "duplicate"},
        "topology_context": {
            "topo_memory": {
                "map_id": "test-map",
                "revision": 4,
                "current_topology_node_id": "backbone-001",
                "block_layer": [
                    {"topology_node_id": f"backbone-00{index}"}
                    for index in range(1, 4)
                ],
                "road_layer": [
                    {"source": "backbone-001", "target": "backbone-002"},
                    {"source": "backbone-002", "target": "backbone-003"},
                ],
                "function_layer": function_layer,
                "object_poi_layer": object_layer,
            },
            "agentic_plan": {"plan_id": "duplicate-plan"},
            "active_subtask": {
                "subtask_id": "subtask-001",
                "topology_node_id": "backbone-002",
            },
        },
    }
    route_plan = {
        "selected_topology_node_id": "backbone-002",
        "destination_topology_node_id": "backbone-003",
        "regional_guidance": {"route_node_ids": ["backbone-001", "backbone-002"]},
    }

    compact = tools._agent_slow_context(slow_context, route_plan)
    topo = compact["topology_context"]["topo_memory"]

    assert "previous_decision" not in compact
    assert "agentic_plan" not in compact["topology_context"]
    assert len(topo["block_layer"]) == 3
    assert len(topo["road_layer"]) == 2
    assert [item["topology_node_id"] for item in topo["function_layer"]] == [
        "backbone-001",
        "backbone-002",
        "backbone-003",
    ]
    assert [item["topology_node_id"] for item in topo["object_poi_layer"]] == [
        "backbone-001",
        "backbone-002",
        "backbone-003",
    ]


def test_hierarchical_prompts_enforce_structured_task_reasoning():
    root = Path(__file__).parents[1]
    for name in ("hierarchical_SOUL.md", "hierarchical_memory_SOUL.md"):
        prompt = (root / "configs" / name).read_text()
        assert "route hypothesis" in prompt
        assert "anchor chain" in prompt
        assert "allowed_anchor_views" in prompt
        assert "MemoryView" in prompt
        assert "recall_navigation_memory" in prompt
        assert "advance_navigation" in prompt


def test_visual_memory_recall_returns_at_most_four_historical_images(monkeypatch, tmp_path):
    tools = _load_tools_module()
    memories = []
    for index in range(4):
        path = tmp_path / f"memory-{index}.jpg"
        Image.new("RGB", (8, 6), color=(index, 60, 80)).save(path)
        memories.append(
            {
                "kind": "trajectory_keyframe",
                "subtask_id": "navigation-subtask-1",
                "location_id": f"location-{index + 1:03d}",
                "source_target": "bedroom doorway",
                "frame_index": index,
                "view": "front",
                "pose": {"x": float(index), "y": 0.0, "yaw": 0.0},
                "image_ref": str(path),
            }
        )
    monkeypatch.setattr(
        tools,
        "_request_json",
        lambda path, payload: {
            "ok": True,
            "scope": payload["scope"],
            "query": payload.get("query", ""),
            "location_id": payload.get("location_id"),
            "archive_count": 19,
            "memories": memories,
        },
    )

    result = tools.recall_navigation_memory({"scope": "task_overview", "limit": 4})

    assert result["_multimodal"] is True
    assert sum(item["type"] == "image_url" for item in result["content"]) == 4
    assert json.loads(result["text_summary"])["archive_count"] == 19


def test_image_window_keeps_latest_navigation_run_and_structured_history():
    image_context = _load_image_context_module()
    old_summary = '{"status":"completed","anchor":"doorway"}'
    latest_summary = '{"status":"completed","anchor":"sofa"}'
    request = {
        "messages": [
            {"role": "user", "content": "find the bedroom"},
            {
                "role": "assistant",
                "content": "first decision",
                "tool_calls": [
                    {"id": "call-old", "function": {"name": "navigate_to_visible_anchor"}}
                ],
            },
            {
                "role": "tool",
                "name": "navigate_to_visible_anchor",
                "tool_name": "navigate_to_visible_anchor",
                "tool_call_id": "call-old",
                "api_content": "stale-sidecar",
                "content": [
                    {"type": "text", "text": old_summary},
                    {"type": "text", "text": "old trajectory keyframe"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,OLD"}},
                ],
            },
            {
                "role": "assistant",
                "content": "continuing",
                "tool_calls": [
                    {"id": "call-latest", "function": {"name": "navigate_to_visible_anchor"}}
                ],
            },
            {
                "role": "tool",
                "name": "navigate_to_visible_anchor",
                "tool_name": "navigate_to_visible_anchor",
                "tool_call_id": "call-latest",
                "content": [
                    {"type": "text", "text": latest_summary},
                    *[
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,NEW{index}"}}
                        for index in range(8)
                    ],
                ],
            },
        ]
    }

    result = image_context.prune_historical_navigation_images(request=request)

    assert result is not None
    messages = result["request"]["messages"]
    assert len(messages) == 3
    assert messages[1]["content"] == "continuing"
    assert sum(part["type"] == "image_url" for part in messages[2]["content"]) == 8
    assert request["messages"][2]["content"][2]["type"] == "image_url"


def test_image_window_keeps_latest_movement_keyframes_with_newer_viewpoint_packet():
    image_context = _load_image_context_module()
    movement_content = [{"type": "text", "text": '{"status":"completed"}'}]
    movement_content.extend(
        [
            item
            for index in range(4)
            for item in (
                {"type": "text", "text": f"Habitat trajectory keyframe {index}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,K{index}"}},
            )
        ]
    )
    movement_content.extend(
        [
            {"type": "text", "text": "post-execution semantic map and current views"},
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,P{index}"}}
                for index in range(4)
            ],
        ]
    )
    viewpoint_content = [
        {"type": "text", "text": '{"action":"turn_left"}'},
        *[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,V{index}"}}
            for index in range(4)
        ],
    ]
    request = {
        "messages": [
            {
                "role": "tool",
                "name": "navigate_to_visible_anchor",
                "content": movement_content,
            },
            {"role": "assistant", "content": "adjusting view"},
            {"role": "tool", "name": "adjust_viewpoint", "content": viewpoint_content},
        ]
    }

    result = image_context.prune_historical_navigation_images(request=request)

    assert result is not None
    messages = result["request"]["messages"]
    assert sum(part["type"] == "image_url" for part in messages[0]["content"]) == 4
    assert sum(part["type"] == "image_url" for part in messages[2]["content"]) == 4


def test_image_window_drops_old_memory_views_but_keeps_memory_delta():
    image_context = _load_image_context_module()
    old = {
        "status": "completed",
        "decision": {"action_type": "visible_anchor", "target": "doorway"},
        "memory_delta": {"revision": 3, "created": True},
        "memory_view": {"revision": 3, "relevant_places": [{"place_id": "location-001"}]},
    }
    request = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "old", "function": {"name": "advance_navigation"}}],
            },
            {
                "role": "tool",
                "name": "advance_navigation",
                "tool_call_id": "old",
                "content": [
                    {"type": "text", "text": json.dumps(old)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,OLD"}},
                ],
            },
            {
                "role": "assistant",
                "content": "continue",
                "tool_calls": [{"id": "new", "function": {"name": "advance_navigation"}}],
            },
            {
                "role": "tool",
                "name": "advance_navigation",
                "tool_call_id": "new",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "completed",
                                "decision": {"action_type": "visible_anchor", "target": "hallway"},
                                "memory_view": {"revision": 4, "relevant_places": []},
                            }
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NEW"}},
                ],
            },
        ]
    }

    result = image_context.prune_historical_navigation_images(request=request)

    messages = result["request"]["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == "continue"
    assert "memory_view" in json.loads(messages[1]["content"][0]["text"])


def test_image_window_hydrates_only_latest_packet_and_stays_bounded_at_40_rounds(tmp_path):
    image_context = _load_image_context_module()
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(image)

    def make_request(rounds):
        messages = [{"role": "user", "content": "find the bedroom"}]
        for index in range(rounds):
            call_id = f"call-{index}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"decision {index}",
                        "tool_calls": [
                            {"id": call_id, "function": {"name": "advance_navigation"}}
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "advance_navigation",
                        "tool_call_id": call_id,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "status": "completed",
                                        "decision": {
                                            "action_type": "visible_anchor",
                                            "target": f"doorway-{index}",
                                        },
                                        "memory_view": {"revision": index, "relevant_places": []},
                                    }
                                ),
                            },
                            *[
                                item
                                for frame_index in range(8)
                                for item in (
                                    {"type": "text", "text": f"trajectory keyframe {frame_index}"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"cee-local-image:{image}"},
                                    },
                                )
                            ],
                        ],
                    },
                ]
            )
        return {"messages": messages}

    five = image_context.prune_historical_navigation_images(request=make_request(5))["request"]
    forty = image_context.prune_historical_navigation_images(request=make_request(40))["request"]

    assert len(five["messages"]) == len(forty["messages"]) == 3
    assert json.dumps(forty).count("data:image/jpeg;base64,") == 8
    assert "cee-local-image:" not in json.dumps(forty)
    assert len(json.dumps(forty)) - len(json.dumps(five)) < 100


def test_image_window_discards_task_start_views_after_latest_packet_at_40_rounds(tmp_path):
    image_context = _load_image_context_module()
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(image)
    messages = [
        {"role": "user", "content": "go through the wooden door and find the bedroom"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "begin", "function": {"name": "begin_navigation_task"}}],
        },
        {
            "role": "tool",
            "name": "begin_navigation_task",
            "tool_call_id": "begin",
            "content": [
                {"type": "text", "text": '{"status":"started"}'},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"cee-local-image:{image}"},
                    }
                    for _ in range(3)
                ],
            ],
        },
    ]
    for index in range(40):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"decision {index}",
                    "tool_calls": [{"id": call_id, "function": {"name": "advance_navigation"}}],
                },
                {
                    "role": "tool",
                    "name": "advance_navigation",
                    "tool_call_id": call_id,
                    "content": [
                        {"type": "text", "text": '{"status":"completed"}'},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"cee-local-image:{image}"},
                            }
                            for _ in range(8)
                        ],
                    ],
                },
            ]
        )

    bounded = image_context.prune_historical_navigation_images(request={"messages": messages})[
        "request"
    ]

    assert len(bounded["messages"]) == 3
    assert bounded["messages"][2]["name"] == "advance_navigation"
    assert all(message.get("name") != "begin_navigation_task" for message in bounded["messages"])
    assert json.dumps(bounded).count("data:image/jpeg;base64,") == 8
    assert "cee-local-image:" not in json.dumps(bounded)


def test_recalled_images_are_bounded_and_evicted_after_next_navigation_packet():
    image_context = _load_image_context_module()

    def images(prefix, count):
        return [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prefix}{index}"}}
            for index in range(count)
        ]

    movement = [{"type": "text", "text": '{"decision":{"action_type":"visible_anchor"}}'}]
    for index in range(4):
        movement.extend(
            [
                {"type": "text", "text": f"Latest trajectory keyframe {index}"},
                images("K", 4)[index],
            ]
        )
    movement.extend(images("CURRENT", 4))
    messages = [
        {
            "role": "tool",
            "name": "begin_navigation_task",
            "content": [{"type": "text", "text": '{"status":"started"}'}, *images("G", 3)],
        },
        {"role": "tool", "name": "advance_navigation", "content": movement},
        {
            "role": "tool",
            "name": "recall_navigation_memory",
            "content": [{"type": "text", "text": '{"scope":"task_overview"}'}, *images("R", 4)],
        },
    ]

    recalled = image_context.prune_historical_navigation_images(request={"messages": messages})[
        "request"
    ]
    assert json.dumps(recalled).count("data:image/jpeg;base64,") == 8
    assert recalled["messages"][-1]["name"] == "recall_navigation_memory"

    messages.append(
        {
            "role": "tool",
            "name": "advance_navigation",
            "content": [
                {"type": "text", "text": '{"decision":{"action_type":"visible_anchor"}}'},
                *images("NEXT", 8),
            ],
        }
    )
    advanced = image_context.prune_historical_navigation_images(request={"messages": messages})[
        "request"
    ]
    assert json.dumps(advanced).count("data:image/jpeg;base64,") == 8
    assert all(message.get("name") != "recall_navigation_memory" for message in advanced["messages"])


def test_navigation_request_normalizer_repairs_aliases_and_bounds_fields():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "observation_id": "obs-4",
            "global_goal": "find the bedroom",
            "previous_assessment": {
                "execution_status": "success",
                "task_effect": "useful_progress",
                "evidence": "crossed the doorway",
            },
            "route_hypothesis": "the hallway leads toward the bedroom",
            "candidates": [
                {
                    "action_type": "visible_anchor",
                    "target_location": "far doorway",
                    "task_relevance": 2,
                }
            ],
            "action": {
                "action_type": "visible_anchor",
                "target_location": "sofa",
                "view": "front",
            },
            "semantic_map": {
                "scene_summary": "x" * 300,
                "left": ["cabinet"],
                "front": ["sofa"],
                "right": [],
            },
        },
    )["args"]

    assert result["previous_assessment"]["execution_status"] == "completed"
    assert result["previous_assessment"]["task_effect"] == "progress"
    assert result["action"]["target"] == "sofa"
    assert result["action"]["low_confidence_recovery"] == "adjust_viewpoint"
    assert result["candidates"][-1]["target"] == "sofa"
    assert len(result["semantic_map"]["scene_summary"]) == 160

    finish = normalization.normalize_navigation_request(
        tool_name="finish_navigation_task",
        args={"global_goal": "ignored", "observation_id": "obs-5"},
    )["args"]
    assert finish == {"outcome": "done", "observation_id": "obs-5"}


def test_navigation_request_normalizer_preserves_abot_task_cot_and_reflection():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "navigation_task_type": "object_goal",
            "cot": {
                "target_localization_and_key_context": (
                    "The white refrigerator is visible in the front-right room."
                ),
                "path_planning": "Use the connected kitchen road, then approach locally.",
            },
            "self_reflection": {
                "subtask_id": "subtask-001",
                "completed": True,
                "outcome": "completed",
                "feedback": "The kitchen region was reached.",
                "should_replan": False,
            },
            "action": {
                "action_type": "visible_anchor",
                "target": "white refrigerator",
                "view": "front",
                "anchor_role": "target",
            },
        },
    )["args"]

    assert result["navigation_task_type"] == "object_goal"
    assert result["cot"] == {
        "target_localization_and_key_context": (
            "The white refrigerator is visible in the front-right room."
        ),
        "path_planning": "Use the connected kitchen road, then approach locally.",
    }
    assert result["self_reflection"]["subtask_id"] == "subtask-001"
    assert result["action"]["anchor_role"] == "target"
    assert result["semantic_map"]["scene_summary"].startswith(
        "The white refrigerator is visible"
    )
    assert result["semantic_map"]["front"] == ["white refrigerator"]

    point_goal = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "navigation_task_type": "point_goal",
            "cot": None,
            "self_reflection": None,
            "action": {
                "action_type": "known_location",
                "target": "backbone-002",
                "topology_node_id": "backbone-002",
            },
        },
    )["args"]
    assert point_goal["cot"] is None

    planned = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "navigation_task_type": "point_goal",
            "cot": None,
            "self_reflection": None,
            "planner_output": {
                "global_instruction": "go to the bedroom and find the bed",
                "destination_query": "bedroom",
                "object_goal": "bed",
                "node_selection": {
                    "selected_topology_node_id": "backbone-003",
                    "confidence": 0.94,
                    "matched_concepts": ["bedroom", "bed"],
                    "selection_reason": "The node jointly matches the place and object.",
                },
                "subtasks": [
                    {
                        "stage": "approaching",
                        "task_type": "point_goal",
                        "instruction": "Approach the bedroom region.",
                        "target": "bedroom",
                        "topology_node_id": "backbone-003",
                        "success_condition": "Reach the bedroom region.",
                    }
                ],
            },
            "action": {
                "action_type": "known_location",
                "target": "backbone-003",
                "topology_node_id": "backbone-003",
            },
        },
    )["args"]
    assert planned["planner_output"]["subtasks"][0]["topology_node_id"] == (
        "backbone-003"
    )
    assert planned["planner_output"]["node_selection"][
        "selected_topology_node_id"
    ] == "backbone-003"

    legacy_planned = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "navigation_task_type": "object_goal",
            "cot": None,
            "planner_output": {
                "task_spec": "go to office to find a white fridge",
                "active_target": {"target_description": "white fridge"},
                "target_region": "backbone-001",
                "subtasks": [
                    {
                        "type": "approaching",
                        "description": "Navigate to the office region.",
                        "node_id": "backbone-001",
                        "node_selection": {
                            "selected_topology_node_id": "backbone-001",
                            "confidence": 0.8,
                            "reason": "The model selected this regional hypothesis.",
                        },
                    },
                    {
                        "type": "reaching",
                        "description": "Locate the white fridge.",
                        "target_object": "white fridge",
                    },
                ],
            },
            "action": {
                "type": "visible_anchor",
                "target": "open area ahead",
                "view": "front",
                "anchor_role": "affordance",
            },
        },
    )["args"]
    legacy_plan = legacy_planned["planner_output"]
    assert legacy_plan["global_instruction"] == "go to office to find a white fridge"
    assert legacy_plan["object_goal"] == "white fridge"
    assert legacy_plan["subtasks"][0]["stage"] == "approaching"
    assert legacy_plan["subtasks"][0]["task_type"] == "point_goal"
    assert legacy_plan["subtasks"][0]["topology_node_id"] == "backbone-001"
    assert legacy_plan["subtasks"][1]["task_type"] == "object_goal"
    assert legacy_plan["node_selection"]["selected_topology_node_id"] == (
        "backbone-001"
    )

    nested_plan = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "navigation_task_type": "object_goal",
            "planner_output": {
                "active_target_evidence": {
                    "target_description": "tall white refrigerator",
                    "topology_node": "backbone-002",
                    "confidence": 0.75,
                    "evidence_type": "topo_memory_inference",
                    "landmarks": ["white refrigerator", "wooden shelves"],
                },
                "plan": {
                    "subtasks": [
                        {
                            "type": "approaching",
                            "task_type": "point_goal",
                            "instruction": "go to office to find a white fridge",
                            "description": "Navigate to the open hall.",
                            "node_id": "backbone-002",
                        },
                        {
                            "type": "reaching",
                            "task_type": "object_goal",
                            "target_object": "white fridge",
                            "target_localization": {
                                "topology_node_id": "backbone-002",
                                "description": "tall white refrigerator",
                            },
                        },
                    ]
                },
            },
            "action": {
                "type": "visible_anchor",
                "target": "open route ahead",
                "view": "front",
            },
        },
    )["args"]["planner_output"]

    assert nested_plan["global_instruction"] == (
        "go to office to find a white fridge"
    )
    assert nested_plan["object_goal"] == "white fridge"
    assert nested_plan["subtasks"][1]["topology_node_id"] == "backbone-002"
    assert nested_plan["node_selection"]["selected_topology_node_id"] == (
        "backbone-002"
    )

    wrapped = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "navigation_task_type": "point_goal",
            "agentic_plan": {
                "planner_output": {
                    "global_instruction": "go to office to find a white fridge",
                    "destination_query": "office area with white fridge",
                    "object_goal": "white fridge",
                    "subtasks": [
                        {
                            "stage": "approaching",
                            "task_type": "point_goal",
                            "instruction": "Navigate to the refrigerator area.",
                            "target": "refrigerator area",
                            "topology_node_id": "backbone-002",
                            "success_condition": "Reach backbone-002.",
                        },
                        {
                            "stage": "reaching",
                            "task_type": "object_goal",
                            "instruction": "Find the white fridge.",
                            "target": "white fridge",
                            "success_condition": "The white fridge is visible.",
                        },
                    ],
                }
            },
            "action": {
                "mode": "known_location",
                "known_node": "backbone-002",
            },
        },
    )["args"]

    assert wrapped["planner_output"]["object_goal"] == "white fridge"
    assert wrapped["action"]["action_type"] == "known_location"
    assert wrapped["action"]["topology_node_id"] == "backbone-002"


def test_navigation_request_normalizer_preserves_natural_reasoning_aliases():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "assessment": "current_place_unknown",
            "route_hypothesis": "Leave the kitchen and find the sofa near the bedroom.",
            "candidates": [
                {
                    "anchor": "open hallway visible ahead",
                    "direction": "forward",
                    "expected_evidence": "a sofa or bedroom doorway",
                    "revisit_penalty": 0.0,
                }
            ],
            "action": {"type": "visible_anchor", "action_semantics": "move_straight"},
            "expected_success_condition": "reach the larger room",
            "fallback": "search the side passage",
        },
    )["args"]

    assert result["action"] == {
        "action_type": "visible_anchor",
        "intent": "goal_pursuit",
        "target": "open hallway visible ahead",
        "view": "front",
        "low_confidence_recovery": "adjust_viewpoint",
        "expected_evidence": "a sofa or bedroom doorway",
        "success_condition": "reach the larger room",
        "fallback": "search the side passage",
    }
    assert result["global_goal"] == "Leave the kitchen and find the sofa near the bedroom."
    assert result["previous_assessment"]["execution_status"] == "not_started"


def test_navigation_request_normalizer_keeps_instruction_before_final_reaching():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "planner_output": {
                "global_instruction": "go out, turn left, and find the robot",
                "destination_query": "area after the left turn",
                "object_goal": "robot",
                "subtasks": [
                    {
                        "stage": "interaction",
                        "task_type": "instruction_following",
                        "instruction": "Go out of the room, turn left, and go straight.",
                        "target": "ordered route segment",
                        "success_condition": "The route segment is complete.",
                    }
                ],
            },
            "action": {
                "action_type": "viewpoint_adjustment",
                "target": "turn_left",
                "viewpoint_action": "turn_left",
            },
        },
    )["args"]

    assert [
        subtask["stage"] for subtask in result["planner_output"]["subtasks"]
    ] == ["interaction", "reaching"]
    assert result["planner_output"]["subtasks"][-1]["target"] == "robot"


def test_navigation_request_normalizer_promotes_legacy_selected_anchor_semantics():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "action": {
                "confidence": 0.85,
                "intent": "goal_pursuit",
                "selected_anchor": {
                    "type": "visible_anchor",
                    "description": "Bed with tan cover visible in the front view",
                    "semantic_hint": "bed - tan/brown bed cover",
                    "x_px": 580,
                    "y_px": 350,
                    "depth_m": 3.5,
                },
            },
            "assessment": "completed",
            "route_hypothesis": "The bed is visible in the bedroom.",
            "expected_evidence": "Close-up evidence of the bed.",
            "success_condition": "Approach the bed within 1.5m.",
            "fallback": "Revisit the previous bed observation.",
        },
    )["args"]

    assert result["action"]["action_type"] == "visible_anchor"
    assert result["action"]["target"] == "bed - tan/brown bed cover"
    assert result["action"]["view"] == "front"
    assert result["action"]["expected_evidence"] == "Close-up evidence of the bed."
    assert result["action"]["success_condition"] == "Approach the bed within 1.5m."
    assert result["previous_assessment"]["execution_status"] == "not_started"
    assert "visible traversable anchor" not in str(result)


def test_navigation_request_normalizer_preserves_exploration_topology():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "action": {
                "action_type": "visible_anchor",
                "intent": "explore_route",
                "target": "open corridor floor",
                "view": "front",
            },
            "semantic_map": {
                "scene_summary": "junction with two open corridors",
                "left": ["left corridor"],
                "front": ["front corridor"],
                "right": [],
                "route_node_type": "junction",
                "traversable_directions": ["left", "front", "blocked"],
            },
        },
    )["args"]

    assert result["action"]["intent"] == "explore_route"
    assert result["candidates"][0]["intent"] == "explore_route"
    assert result["semantic_map"]["route_node_type"] == "junction"
    assert result["semantic_map"]["traversable_directions"] == ["left", "front"]


def test_navigation_request_normalizer_extracts_known_location_id_from_target():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "action": {
                "action_type": "known_location",
                "intent": "explore_route",
                "target": "the beige sofa visible in location-025 at z=18.46",
            },
            "candidates": [
                {
                    "action_type": "known_location",
                    "target": "the beige sofa at location-025",
                }
            ],
        },
    )["args"]

    assert result["action"]["target"] == "the beige sofa visible in location-025 at z=18.46"
    assert result["action"]["location_id"] == "location-025"


def test_navigation_request_normalizer_prefers_stable_topology_node_id():
    normalization = _load_request_normalization_module()
    result = normalization.normalize_navigation_request(
        tool_name="advance_navigation",
        args={
            "action": {
                "action_type": "known_location",
                "intent": "goal_pursuit",
                "target": "bedroom node backbone-005",
                "topology_node_id": "backbone-005",
            }
        },
    )["args"]

    assert result["action"]["topology_node_id"] == "backbone-005"
    assert "location_id" not in result["action"]


def test_finish_task_clears_memory(monkeypatch):
    tools = _load_tools_module()
    monkeypatch.setattr(
        tools,
        "_request",
        lambda path, payload: json.dumps(
            {
                "ok": True,
                "path": path,
                "outcome": payload["outcome"],
                "memory_cleared": True,
            }
        ),
    )

    result = json.loads(tools.finish_task({"outcome": "done"}))

    assert result["path"] == "/task/finish"
    assert result["memory_cleared"] is True
