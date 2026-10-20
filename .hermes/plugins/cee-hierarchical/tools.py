from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_LOCAL_IMAGE_PREFIX = "cee-local-image:"
_LATEST_OBSERVATION_ID: str | None = None


def _request(path: str, payload: dict | None = None, *, timeout_s: float = 150.0) -> str:
    base_url = os.environ.get("CEE_HIERARCHICAL_TOOL_URL", "http://127.0.0.1:8766").rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        parsed_error = None
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict):
                parsed_error = dict(parsed)
            if isinstance(parsed, dict) and parsed.get("error"):
                detail = str(parsed["error"])
        except json.JSONDecodeError:
            pass
        response = parsed_error or {"ok": False}
        response["ok"] = False
        response["error"] = f"HTTP {exc.code}: {detail[:760]}"
        return json.dumps(response, ensure_ascii=False)
    except (URLError, TimeoutError, OSError) as exc:
        return json.dumps({"ok": False, "error": f"Hierarchical service unavailable: {exc}"})


def status(_args: dict, **_kwargs) -> str:
    return _request("/health")


def capabilities(_args: dict, **_kwargs) -> str:
    return _request("/capabilities")


def observe(_args: dict, **_kwargs) -> dict | str:
    return _observation_multimodal(_request_json("/observe", {}))


def begin_navigation(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/decision/begin", args)
    _remember_observation(result)
    response = _observation_multimodal(result)
    if not isinstance(response, dict):
        return response
    observation = result.get("observation") or {}
    compact = {
        "ok": True,
        "observation_id": (observation.get("trace") or {}).get("observation_id"),
        "pose": observation.get("pose"),
        "agentic_plan": (result.get("semantic_route_plan") or {}).get("agentic_plan"),
        "slow_context": result.get("slow_context"),
        "planner_required": result.get("planner_required", False),
        "next_call_contract": result.get("next_call_contract"),
        "instruction": (
            "Follow next_call_contract exactly. If planner_required=true, the first advance_navigation call must include an ABot-N0 planner_output "
            "built from this task, current tri-view, Episodic Memory, and topology_context.topo_memory. Use typed "
            "Approaching/Point-Goal, Reaching/Object-or-POI-Goal, and Interaction/Instruction-or-Person roles in "
            "the order required by the instruction; roles may interleave or repeat. Select only connected topology "
            "nodes present in Road layer; after planning, "
            "regional_guidance.allowed_anchor_views constrains each connected visual transition, and local "
            "object search begins only when target_region_reached=true. Do not call begin_navigation_task again "
            "for the same goal; continue from the returned active Agentic Plan."
        ),
    }
    if compact["slow_context"] is None:
        compact.update(
            memory_view=_agent_memory_view(result.get("memory_view")),
            route_graph=_agent_route_graph(result.get("route_graph")),
            topology=_agent_topology(result.get("topology")),
            semantic_route_plan=result.get("semantic_route_plan"),
        )
    response["content"].insert(0, {"type": "text", "text": json.dumps(compact, ensure_ascii=False)})
    response["text_summary"] = json.dumps(compact, ensure_ascii=False)
    return response


def recall_navigation_memory(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/memory/visual-recall", args)
    if not result.get("ok"):
        return json.dumps(result)
    memories = (result.get("memories") or [])[:4]
    compact = {
        "ok": True,
        "scope": result.get("scope"),
        "query": result.get("query"),
        "location_id": result.get("location_id"),
        "count": len(memories),
        "archive_count": result.get("archive_count"),
        "instruction": (
            "These are bounded historical memories, not current observations. A location-filtered recall returns "
            "the stored LEFT/FRONT/RIGHT evidence for that exact L position when available. Compare it with the "
            "latest views, then continue using the current observation_id."
        ),
    }
    content = [{"type": "text", "text": json.dumps(compact, ensure_ascii=False)}]
    for memory in memories:
        label = (
            f"Recalled {memory.get('kind')} from subtask={memory.get('subtask_id')}; "
            f"location_id={memory.get('location_id')}; target={memory.get('source_target')}; "
            f"frame_index={memory.get('frame_index')}; view={memory.get('view')}; "
            f"pose={memory.get('pose')}"
        )
        content.extend(_image_content(label, str(memory.get("image_ref") or "")))
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": json.dumps(compact, ensure_ascii=False),
    }


def search_semantic_map(args: dict, **_kwargs) -> str:
    result = _request_json("/semantic-map/search", args)
    if not result.get("ok"):
        return json.dumps(result, ensure_ascii=False)
    results = [
        {
            "topology_node_id": item.get("topology_node_id"),
            "score": item.get("score"),
            "matched_terms": item.get("matched_terms") or [],
            "node_semantics": item.get("node_semantics"),
        }
        for item in (result.get("results") or [])
    ]
    compact = {
        "ok": True,
        "query": result.get("query"),
        "count": result.get("count"),
        "results": results,
        "instruction": (
            "Use topology_node_id as regional semantic context. A node may justify an explicit region switch but does "
            "not override a useful visible anchor. An empty result means the place requires visual exploration."
        ),
    }
    return json.dumps(compact, ensure_ascii=False)


def advance_navigation(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/decision/advance", _bind_latest_observation(args))
    _remember_observation(result)
    if result.get("status") == "failed" or result.get("error"):
        return json.dumps(result)
    execution = result.get("result") or {}
    decision = result.get("decision") or {}
    policy_override = result.get("policy_override") or {}
    semantic_route_plan = result.get("semantic_route_plan") or {}
    compact = {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "finish_required": result.get("finish_required", False),
        "decision": decision,
        "slow_output": result.get("slow_output"),
        "fast_execution": result.get("fast_execution"),
        "slow_context": _agent_slow_context(
            result.get("slow_context"),
            semantic_route_plan,
        ),
        "self_reflection": result.get("self_reflection"),
        "protocol_normalization": result.get("protocol_normalization"),
        "next_call_contract": result.get("next_call_contract"),
        "active_target_evidence": result.get("active_target_evidence"),
        "executed_action": _executed_action(decision, policy_override, result),
        "controller": execution.get("controller"),
        "reached": execution.get("reached"),
        "execution_chunks": execution.get("execution_chunks"),
        "termination_reason": execution.get("termination_reason"),
        "control_steps": execution.get("control_steps"),
        "distance_m": execution.get("distance_m"),
        "remaining_distance_m": execution.get("remaining_distance_m"),
        "motion_performed": execution.get("motion_performed"),
        "completion_scope": execution.get("completion_scope"),
        "no_motion_reason": execution.get("no_motion_reason"),
        "collisions": execution.get("collisions"),
        "stalled": execution.get("stalled"),
        "grounding_confidence": result.get("grounding_confidence"),
        "grounding_confidence_threshold": result.get("grounding_confidence_threshold"),
        "anchor_visible": result.get("anchor_visible"),
        "grounded_view": (result.get("generated_point") or {}).get("view") or result.get("view"),
        "anchor_source_observation_id": result.get("anchor_source_observation_id"),
        "anchor_rejected": result.get("anchor_rejected", False),
        "raw_anchor_depth_m": result.get("raw_anchor_depth_m"),
        "anchor_depth_limited": result.get("anchor_depth_limited", False),
        "anchor_max_depth_m": result.get("anchor_max_depth_m"),
        "active_semantic_node_id": result.get("active_semantic_node_id"),
        "regional_context_node_id": result.get("regional_context_node_id"),
        "active_evidence_location_id": result.get("active_evidence_location_id"),
        "anchor_distance_from_node_m": result.get("anchor_distance_from_node_m"),
        "node_scope_limited": result.get("node_scope_limited", False),
        "recovery": result.get("recovery"),
        "next_observation_id": result.get("next_observation_id"),
        "memory_delta": _agent_memory_delta(result.get("memory_delta")),
        "topology_update": result.get("topology_update"),
        "agentic_plan": result.get("agentic_plan"),
        "regional_state": _agent_regional_state(semantic_route_plan),
        "trajectory_memory": _agent_trajectory_memory(
            result.get("trajectory_memory")
        ),
    }
    if compact["slow_context"] is None:
        compact["memory_view"] = _agent_memory_view(result.get("memory_view"))
    content = [{"type": "text", "text": json.dumps(compact, ensure_ascii=False)}]
    post_observation = result.get("post_observation") or {}
    if decision.get("action_type") in {"visible_anchor", "known_location"}:
        for frame in (execution.get("trajectory_keyframes") or [])[:4]:
            front_ref = (frame.get("rgb_refs") or {}).get("front")
            if front_ref:
                content.extend(
                    _image_content(
                        f"Latest trajectory keyframe {frame.get('frame_index')}; "
                        f"location_id={frame.get('location_id')}; pose={frame.get('pose')}; "
                        f"map_id={frame.get('map_id')}; "
                        f"navigation_point_xyz_m={frame.get('navigation_point_xyz_m')}",
                        str(front_ref),
                    )
                )
    post_refs = post_observation.get("rgb_refs") or {}
    for view in ("left", "front", "right"):
        ref = post_refs.get(view)
        if not ref:
            return json.dumps({"ok": False, "error": f"post-execution {view} RGB image is unavailable"})
        content.extend(_image_content(f"Current {view.upper()} camera view.", str(ref)))
    next_contract = compact.get("next_call_contract") or {}
    next_tool = next_contract.get("next_tool") or "advance_navigation"
    content.append(
        {
            "type": "text",
            "text": (
                f"MANDATORY NEXT TOOL: {next_tool}. Follow next_call_contract exactly. "
                "Do not answer the user or claim task completion before "
                "finish_navigation_task returns an authorized result."
            ),
        }
    )
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": json.dumps(compact, ensure_ascii=False),
    }


def _executed_action(decision: dict, policy_override: dict, result: dict) -> dict:
    if result.get("finish_required"):
        return {
            "mode": "FINISH_VERIFY",
            "target": None,
            "target_id": None,
            "view": None,
            "intent": "verify_global_goal",
            "selection_source": "runtime_transition",
            "override_reason": "final_subtask_reflected_complete",
        }
    recovery = result.get("recovery") or {}
    if result.get("anchor_rejected") and recovery.get("executed") == "adjust_viewpoint":
        return {
            "mode": "VIEWPOINT_ADJUST",
            "target": recovery.get("target"),
            "target_id": recovery.get("target"),
            "view": None,
            "intent": "recovery",
            "selection_source": "anchor_recovery",
            "override_reason": "anchor_rejected",
        }
    if result.get("anchor_rejected") and recovery.get("executed") == "previous_location":
        return {
            "mode": "L_REVISIT",
            "target": recovery.get("target"),
            "target_id": recovery.get("target"),
            "view": None,
            "intent": "recovery",
            "selection_source": "anchor_recovery",
            "override_reason": "anchor_rejected",
        }
    action_type = decision.get("action_type")
    if action_type == "known_location":
        if decision.get("topology_node_id"):
            mode = "NODE_NAV"
            target_id = decision.get("topology_node_id")
        else:
            mode = "L_REVISIT"
            target_id = decision.get("location_id")
    elif action_type == "visible_anchor":
        mode = "VISUAL_ANCHOR_NAV"
        target_id = decision.get("target")
    elif action_type == "viewpoint_adjustment":
        mode = "VIEWPOINT_ADJUST"
        target_id = decision.get("viewpoint_action")
    else:
        mode = "UNKNOWN"
        target_id = decision.get("target")
    executed = {
        "mode": mode,
        "target": decision.get("target"),
        "target_id": target_id,
        "view": decision.get("view"),
        "intent": decision.get("intent"),
        "selection_source": "runtime_override" if policy_override else "agent_selected",
        "override_reason": policy_override.get("reason"),
    }
    if decision.get("anchor_role") is not None:
        executed["anchor_role"] = decision["anchor_role"]
    return executed


def _agent_memory_view(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    compact = dict(raw)
    compact.pop("global_anchor", None)
    for key in ("current_place", "backtrack_location"):
        compact[key] = _agent_place(raw.get(key))
    for key in ("relevant_places", "nearby_backbone_nodes"):
        values = raw.get(key)
        if isinstance(values, list):
            compact[key] = [_agent_place(value) for value in values]
    return compact


def _agent_memory_delta(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    compact = dict(raw)
    compact["current_place"] = _agent_place(raw.get("current_place"))
    return compact


def _agent_slow_context(raw: object, semantic_route_plan: object) -> object:
    if not isinstance(raw, dict):
        return raw
    compact = {
        key: raw.get(key)
        for key in (
            "current_observation_id",
            "historical_front_observation_ids",
            "task_specification",
            "active_target_evidence",
        )
    }
    topology_context = raw.get("topology_context")
    if not isinstance(topology_context, dict):
        compact["topology_context"] = topology_context
        return compact
    relevant_ids = _regional_topology_ids(semantic_route_plan, topology_context)
    topo_memory = topology_context.get("topo_memory")
    if isinstance(topo_memory, dict):
        topo_memory = dict(topo_memory)
        if relevant_ids:
            for layer in ("function_layer", "object_poi_layer"):
                topo_memory[layer] = [
                    item
                    for item in topo_memory.get(layer) or []
                    if isinstance(item, dict)
                    and item.get("topology_node_id") in relevant_ids
                ]
    compact["topology_context"] = {
        "topo_memory": topo_memory,
        "active_subtask": topology_context.get("active_subtask"),
    }
    return compact


def _regional_topology_ids(
    semantic_route_plan: object,
    topology_context: dict,
) -> set[str]:
    relevant: set[str] = set()
    topo_memory = topology_context.get("topo_memory")
    if isinstance(topo_memory, dict) and topo_memory.get("current_topology_node_id"):
        relevant.add(str(topo_memory["current_topology_node_id"]))
    active = topology_context.get("active_subtask")
    if isinstance(active, dict) and active.get("topology_node_id"):
        relevant.add(str(active["topology_node_id"]))
    if isinstance(semantic_route_plan, dict):
        selected = semantic_route_plan.get("selected_topology_node_id")
        if selected:
            relevant.add(str(selected))
        destination = semantic_route_plan.get("destination_topology_node_id")
        if destination:
            relevant.add(str(destination))
        guidance = semantic_route_plan.get("regional_guidance")
        if isinstance(guidance, dict):
            relevant.update(str(item) for item in guidance.get("route_node_ids") or [])
    return relevant


def _agent_regional_state(raw: object) -> object:
    if not isinstance(raw, dict):
        return None
    return {
        "phase": raw.get("phase"),
        "selected_topology_node_id": raw.get("selected_topology_node_id"),
        "destination_topology_node_id": raw.get("destination_topology_node_id"),
        "destination_selection": raw.get("destination_selection"),
        "object_goal": raw.get("object_goal"),
        "regional_guidance": raw.get("regional_guidance"),
    }


def _agent_trajectory_memory(raw: object) -> list[object]:
    if not isinstance(raw, list):
        return []
    compact = []
    for item in raw[:4]:
        if not isinstance(item, dict):
            compact.append(item)
            continue
        compact.append(
            {
                key: item.get(key)
                for key in (
                    "frame_index",
                    "observation_id",
                    "location_id",
                    "pose",
                    "map_id",
                    "navigation_point_xyz_m",
                )
                if item.get(key) is not None
            }
        )
    return compact


def _agent_place(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    compact = dict(raw)
    topology_node_id = compact.get("topology_node_id")
    if topology_node_id:
        compact.pop("location_id", None)
        compact.pop("place_id", None)
    else:
        place_id = compact.pop("place_id", None)
        if place_id and not compact.get("location_id"):
            compact["location_id"] = place_id
    return compact


def _agent_route_graph(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    compact = dict(raw)
    node_ids: dict[str, str] = {}
    nodes: list[object] = []
    for raw_node in raw.get("nodes") or []:
        if not isinstance(raw_node, dict):
            nodes.append(raw_node)
            continue
        node = dict(raw_node)
        location_id = node.get("location_id")
        topology_node_id = node.get("topology_node_id")
        external_id = topology_node_id or location_id
        if location_id and external_id:
            node_ids[str(location_id)] = str(external_id)
        if topology_node_id:
            node.pop("location_id", None)
        nodes.append(node)
    edges: list[object] = []
    for raw_edge in raw.get("edges") or []:
        if not isinstance(raw_edge, dict):
            edges.append(raw_edge)
            continue
        edge = dict(raw_edge)
        edge.pop("path_world_xy_m", None)
        for endpoint in ("source", "target"):
            value = edge.get(endpoint)
            if value in node_ids:
                edge[endpoint] = node_ids[str(value)]
        edges.append(edge)
    compact["nodes"] = nodes
    compact["edges"] = edges
    return compact


def _agent_topology(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    compact = dict(raw)
    compact.pop("node_locations", None)
    return compact


def adjust_viewpoint(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/skills/viewpoint-adjustment", args)
    if not result.get("ok"):
        return json.dumps(result)
    response = _observation_multimodal(
        {"ok": True, "observation": result.get("post_observation") or {}}
    )
    if not isinstance(response, dict):
        return response
    compact = {
        "ok": True,
        "action": result.get("action"),
        "collided": (result.get("result") or {}).get("collided"),
        "distance_m": (result.get("result") or {}).get("distance_m"),
        "pose_before": (result.get("result") or {}).get("pose_before"),
        "pose_after": (result.get("result") or {}).get("pose_after"),
        "next_observation_id": result.get("next_observation_id"),
        "instruction": "Use these fresh views directly; select a visible anchor or adjust viewpoint once more.",
    }
    response["content"].insert(0, {"type": "text", "text": json.dumps(compact, ensure_ascii=False)})
    topdown_ref = (result.get("post_observation") or {}).get("topdown_ref")
    if topdown_ref:
        response["content"][1:1] = _image_content(
            "Current directional semantic map after viewpoint adjustment.",
            str(topdown_ref),
        )
    response["text_summary"] = json.dumps(compact, ensure_ascii=False)
    return response


def inspect_localization_map(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/skills/localization-map", args)
    if not result.get("ok"):
        return json.dumps(result)
    observation = result.get("observation") or {}
    response = _observation_multimodal({"ok": True, "observation": observation})
    if not isinstance(response, dict):
        return response
    localization = result.get("result") or {}
    compact = {
        "ok": True,
        "pose": localization.get("pose"),
        "map_frame": localization.get("map_frame"),
        "map_legend": localization.get("legend"),
        "exploration_memory": result.get("exploration_memory"),
        "next_observation_id": result.get("next_observation_id"),
        "instruction": (
            "Use node-level place semantics and known locations to avoid repeated exploration. "
            "Camera direction is observation evidence, not the semantic identity of a place."
        ),
    }
    topdown_ref = observation.get("topdown_ref")
    if not topdown_ref:
        return json.dumps({"ok": False, "error": "top-down map is unavailable"})
    response["content"].insert(0, {"type": "text", "text": json.dumps(compact, ensure_ascii=False)})
    response["content"][1:1] = _image_content(
        "Current semantic topology: fixed locations with orientation-invariant place knowledge.",
        str(topdown_ref),
    )
    response["text_summary"] = json.dumps(compact, ensure_ascii=False)
    return response


def navigate_known_location(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/skills/known-location-navigate", args)
    if result.get("status") == "failed" or result.get("error"):
        return json.dumps(result)
    execution = result.get("result") or {}
    compact = {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "navigation_target_id": result.get("navigation_target_id"),
        "navigation_target_kind": result.get("navigation_target_kind"),
        "topology_node_id": result.get("topology_node_id"),
        "location_id": result.get("location_id"),
        "target_location": result.get("target_location"),
        "controller": execution.get("controller"),
        "reached": execution.get("reached"),
        "control_steps": execution.get("control_steps"),
        "distance_m": execution.get("distance_m"),
        "remaining_distance_m": execution.get("remaining_distance_m"),
        "motion_performed": execution.get("motion_performed"),
        "completion_scope": execution.get("completion_scope"),
        "no_motion_reason": execution.get("no_motion_reason"),
        "collisions": execution.get("collisions"),
        "stalled": execution.get("stalled"),
        "next_observation_id": result.get("next_observation_id"),
        "exploration_memory": result.get("exploration_memory"),
    }
    content = [{"type": "text", "text": json.dumps(compact, ensure_ascii=False)}]
    post_observation = result.get("post_observation") or {}
    topdown_ref = post_observation.get("topdown_ref")
    if topdown_ref:
        content.extend(
            _image_content(
                "Post-execution top-down map after navigating to the known location.",
                str(topdown_ref),
            )
        )
    for frame in (execution.get("trajectory_keyframes") or [])[:4]:
        front_ref = (frame.get("rgb_refs") or {}).get("front")
        if front_ref:
            content.extend(
                _image_content(
                    f"Known-location trajectory keyframe {frame.get('frame_index')}; pose={frame.get('pose')}",
                    str(front_ref),
                )
            )
    post_refs = post_observation.get("rgb_refs") or {}
    for view in ("left", "front", "right"):
        ref = post_refs.get(view)
        if not ref:
            return json.dumps({"ok": False, "error": f"post-execution {view} RGB image is unavailable"})
        content.extend(_image_content(f"Post-execution current {view.upper()} camera view.", str(ref)))
    content.append(
        {
            "type": "text",
            "text": json.dumps(
                {
                    "next_observation_id": result.get("next_observation_id"),
                    "instruction": "Review these fresh views and choose either a visible anchor or another known location.",
                },
                ensure_ascii=False,
            ),
        }
    )
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": json.dumps(compact, ensure_ascii=False),
    }


def ground_anchor(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/skills/anchor-grounding", args)
    if not result.get("ok"):
        return json.dumps(result)
    outputs = (result.get("result") or {}).get("outputs") or {}
    annotated_ref = outputs.get("annotated_ref")
    content = [{"type": "text", "text": json.dumps(outputs, ensure_ascii=False)}]
    if annotated_ref:
        content.extend(_image_content("Qwen anchor pixel annotation.", str(annotated_ref)))
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": json.dumps(outputs, ensure_ascii=False),
    }


def project_pixel(args: dict, **_kwargs) -> str:
    return _request("/skills/pixel-to-point", args)


def plan_path(args: dict, **_kwargs) -> str:
    return _request("/skills/path-planning", args)


def execute_path(args: dict, **_kwargs) -> str:
    return _request("/skills/path-execution", args)


def navigate_anchor(args: dict, **_kwargs) -> dict | str:
    result = _request_json("/skills/hierarchical-navigate", args)
    if result.get("status") == "failed" or result.get("error"):
        return json.dumps(result)
    execution = result.get("result") or {}
    compact = {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "anchor": result.get("anchor"),
        "view": result.get("view"),
        "generated_point": result.get("generated_point"),
        "raw_anchor_depth_m": result.get("raw_anchor_depth_m"),
        "anchor_depth_limited": result.get("anchor_depth_limited", False),
        "node_scope_radius_m": result.get("node_scope_radius_m"),
        "node_scope_limited": result.get("node_scope_limited", False),
        "anchor_visible": result.get("anchor_visible"),
        "anchor_source_observation_id": result.get("anchor_source_observation_id"),
        "controller": execution.get("controller"),
        "reached": execution.get("reached"),
        "control_steps": execution.get("control_steps"),
        "distance_m": execution.get("distance_m"),
        "remaining_distance_m": execution.get("remaining_distance_m"),
        "collisions": execution.get("collisions"),
        "stalled": execution.get("stalled"),
        "next_observation_id": result.get("next_observation_id"),
        "scene_memory_update": result.get("scene_memory_update"),
        "scene_memory_index": result.get("scene_memory_index") or [],
        "scene_memory_warning": result.get("scene_memory_warning"),
        "exploration_memory": result.get("exploration_memory"),
    }
    content = [{"type": "text", "text": json.dumps(compact, ensure_ascii=False)}]
    post_observation = result.get("post_observation") or {}
    topdown_ref = post_observation.get("topdown_ref")
    if topdown_ref:
        content.extend(
            _image_content(
                "Post-execution directional semantic map with known locations and view-oriented labels.",
                str(topdown_ref),
            )
        )
    for frame in (execution.get("trajectory_keyframes") or [])[:4]:
        front_ref = (frame.get("rgb_refs") or {}).get("front")
        if front_ref:
            content.extend(
                _image_content(
                    f"Habitat native-step trajectory keyframe {frame.get('frame_index')}; pose={frame.get('pose')}",
                    str(front_ref),
                )
            )
    post_refs = post_observation.get("rgb_refs") or {}
    for view in ("left", "front", "right"):
        ref = post_refs.get(view)
        if not ref:
            return json.dumps({"ok": False, "error": f"post-execution {view} RGB image is unavailable"})
        content.extend(_image_content(f"Post-execution current {view.upper()} camera view.", str(ref)))
    content.append(
        {
            "type": "text",
            "text": json.dumps(
                {
                    "next_observation_id": result.get("next_observation_id"),
                    "post_execution_pose": post_observation.get("pose"),
                    "instruction": "Review these fresh views directly; do not call hierarchical_observe again.",
                },
                ensure_ascii=False,
            ),
        }
    )
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": json.dumps(compact, ensure_ascii=False),
    }


def finish_task(args: dict, **_kwargs) -> str:
    return _request("/task/finish", _bind_latest_observation(args))


def _request_json(path: str, payload: dict | None = None) -> dict:
    raw = _request(path, payload)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": raw}
    return result if isinstance(result, dict) else {"ok": False, "error": "invalid service response"}


def _bind_latest_observation(args: dict) -> dict:
    payload = dict(args)
    if _LATEST_OBSERVATION_ID:
        payload["observation_id"] = _LATEST_OBSERVATION_ID
    return payload


def _remember_observation(result: dict) -> None:
    global _LATEST_OBSERVATION_ID

    observation = result.get("observation") or result.get("post_observation") or {}
    observation_id = result.get("next_observation_id") or (observation.get("trace") or {}).get(
        "observation_id"
    )
    if observation_id:
        _LATEST_OBSERVATION_ID = str(observation_id)


def _observation_multimodal(result: dict) -> dict | str:
    if not result.get("ok"):
        return json.dumps(result)
    observation = result.get("observation") or {}
    refs = observation.get("rgb_refs") or {}
    content = []
    for view in ("left", "front", "right"):
        ref = refs.get(view)
        if not ref:
            return json.dumps({"ok": False, "error": f"{view} RGB image is unavailable"})
        content.extend(_image_content(f"Current robot {view.upper()} camera view.", str(ref)))
    content.append(
        {
            "type": "text",
            "text": json.dumps(
                {
                    "observation_id": (observation.get("trace") or {}).get("observation_id"),
                    "pose": observation.get("pose"),
                    "view_order": ["left", "front", "right"],
                },
                ensure_ascii=False,
            ),
        }
    )
    summary = {
        "ok": True,
        "observation_id": (observation.get("trace") or {}).get("observation_id"),
        "pose": observation.get("pose"),
        "view_order": ["left", "front", "right"],
        "instruction": "Select one anchor that is actually visible in these images; do not guess an observation ID.",
    }
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": json.dumps(summary, ensure_ascii=False),
    }


def _image_content(label: str, ref: str) -> list[dict]:
    path = Path(ref).expanduser().resolve()
    if not path.is_file():
        return [{"type": "text", "text": f"{label} Image unavailable: {ref}"}]
    return [
        {"type": "text", "text": label},
        {
            "type": "image_url",
            "image_url": {"url": f"{_LOCAL_IMAGE_PREFIX}{path}"},
        },
    ]
