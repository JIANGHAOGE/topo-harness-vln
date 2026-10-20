from __future__ import annotations

import re
from typing import Any


_ACTIONS = frozenset({"visible_anchor", "known_location", "viewpoint_adjustment"})
_ANCHOR_ROLES = frozenset({"affordance", "target"})
_VIEWS = frozenset({"left", "front", "right"})
_VIEWPOINT_ACTIONS = frozenset(
    {"turn_left", "turn_right", "move_forward", "move_backward"}
)
_INTENTS = frozenset({"goal_pursuit", "explore_route", "backtrack"})
_ROUTE_NODE_TYPES = frozenset({"none", "open_space_center", "junction"})
_LOCATION_ID_PATTERN = re.compile(r"\blocation-[0-9]{3,}\b", flags=re.IGNORECASE)
_TOPOLOGY_NODE_ID_PATTERN = re.compile(r"\bbackbone-[0-9]{3,}\b", flags=re.IGNORECASE)


def normalize_navigation_request(
    *, tool_name: str, args: dict[str, Any], **_kwargs: Any
) -> dict[str, Any] | None:
    if tool_name == "advance_navigation":
        normalized = _normalize_advance(args)
    elif tool_name == "finish_navigation_task":
        normalized = _normalize_finish(args)
    else:
        return None
    return {
        "args": normalized,
        "source": "cee-hierarchical-request-normalizer",
        "reason": "normalized bounded navigation protocol fields",
    }


def _normalize_advance(raw: dict[str, Any]) -> dict[str, Any]:
    cot_raw = raw.get("cot") or raw.get("decision_trace")
    if not isinstance(cot_raw, dict):
        cot_raw = {}
    raw_candidates = [
        item
        for item in (
            raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
        )[:3]
        if isinstance(item, dict)
    ]
    supplied_action = raw.get("action")
    action_raw = dict(supplied_action) if isinstance(supplied_action, dict) else {}
    nested_action = action_raw.get("action")
    if isinstance(nested_action, dict):
        action_raw = {**nested_action, **{key: value for key, value in action_raw.items() if key != "action"}}
    selected_anchor = action_raw.get("selected_anchor")
    if isinstance(selected_anchor, dict):
        action_raw.setdefault(
            "action_type",
            selected_anchor.get("action_type") or selected_anchor.get("type"),
        )
        if not any(action_raw.get(key) for key in ("target", "target_location", "anchor")):
            action_raw["target"] = (
                selected_anchor.get("semantic_hint")
                or selected_anchor.get("description")
            )
        action_raw.setdefault(
            "view",
            selected_anchor.get("view") or selected_anchor.get("direction"),
        )
    for key in (
        "action_type",
        "target",
        "target_location",
        "anchor",
        "view",
        "topology_node_id",
        "location_id",
        "viewpoint_action",
    ):
        if key not in action_raw and raw.get(key) is not None:
            action_raw[key] = raw[key]
    if not isinstance(supplied_action, dict) and supplied_action is not None:
        action_raw["action"] = supplied_action
    if raw_candidates and not any(
        action_raw.get(key)
        for key in (
            "target",
            "target_location",
            "anchor",
            "topology_node_id",
            "location_id",
            "viewpoint_action",
        )
    ):
        candidate = raw_candidates[0]
        action_raw["target"] = _target(candidate, _action_type(candidate, candidate))
        action_raw.setdefault("view", candidate.get("view") or candidate.get("direction"))

    action_type = _action_type(action_raw, raw)
    target = _target(action_raw, action_type)
    candidate_evidence = raw_candidates[0].get("expected_evidence") if raw_candidates else None
    action = {
        "action_type": action_type,
        "intent": _intent(action_raw.get("intent") or raw.get("intent"), action_type),
        "target": target,
        "expected_evidence": _text(
            action_raw.get("expected_evidence")
            or raw.get("expected_evidence")
            or candidate_evidence,
            240,
            "observable progress toward the global goal",
        ),
        "success_condition": _text(
            action_raw.get("success_condition")
            or raw.get("success_condition")
            or raw.get("expected_success_condition"),
            240,
            "the selected navigation action completes",
        ),
        "fallback": _text(
            action_raw.get("fallback") or raw.get("fallback"),
            240,
            "reassess the fresh observation",
        ),
    }
    if action_type == "visible_anchor":
        action["view"] = _view(action_raw.get("view") or raw.get("view"))
        anchor_role = action_raw.get("anchor_role") or raw.get("anchor_role")
        if anchor_role in _ANCHOR_ROLES:
            action["anchor_role"] = str(anchor_role)
        recovery = action_raw.get("low_confidence_recovery") or raw.get(
            "low_confidence_recovery"
        )
        action["low_confidence_recovery"] = (
            recovery
            if recovery in {"adjust_viewpoint", "previous_location"}
            else "adjust_viewpoint"
        )
        recovery_action = action_raw.get("recovery_viewpoint_action") or raw.get(
            "recovery_viewpoint_action"
        )
        if recovery_action in _VIEWPOINT_ACTIONS:
            action["recovery_viewpoint_action"] = recovery_action
    elif action_type == "known_location":
        topology_node_id = _topology_node_id(
            action_raw.get("topology_node_id"),
            action_raw.get("known_node"),
            raw.get("topology_node_id"),
            target,
            *(candidate.get("target") for candidate in raw_candidates),
        )
        if topology_node_id is not None:
            action["topology_node_id"] = topology_node_id
        else:
            action["location_id"] = _location_id(
                action_raw.get("location_id"),
                action_raw.get("known_location_id"),
                raw.get("location_id"),
            target,
            *(candidate.get("target") for candidate in raw_candidates),
        )
    else:
        viewpoint_action = action_raw.get("viewpoint_action") or action_raw.get("action") or target
        action["viewpoint_action"] = (
            viewpoint_action if viewpoint_action in _VIEWPOINT_ACTIONS else "turn_left"
        )

    candidates = [_normalize_candidate(item) for item in raw_candidates]
    selected_candidate = {
        "action_type": action_type,
        "intent": action["intent"],
        "target": target,
        "task_relevance": 1.0,
        "information_gain": 0.5,
        "revisit_penalty": 0.0,
    }
    if not any(
        item["action_type"] == action_type and item["target"].casefold() == target.casefold()
        for item in candidates
    ):
        candidates = (candidates[:2] + [selected_candidate]) if candidates else [selected_candidate]

    assessment = raw.get("previous_assessment")
    if not isinstance(assessment, dict) and raw.get("assessment") is not None:
        assessment = {
            "execution_status": "not_started",
            "task_effect": "unknown",
            "evidence": raw.get("assessment"),
        }
    if not isinstance(assessment, dict):
        assessment = {}
    route = raw.get("route_hypothesis")
    if isinstance(route, str):
        route = {"statement": route}
    if not isinstance(route, dict):
        route = {}

    navigation_task_type = _navigation_task_type(raw.get("navigation_task_type"), cot_raw)
    cot = _normalize_cot(navigation_task_type, cot_raw, assessment, route, action_type, target)

    semantic_map = raw.get("semantic_map")
    if not isinstance(semantic_map, dict):
        semantic_map = {}
    semantic_views = {
        "left": _labels(semantic_map.get("left")),
        "front": _labels(semantic_map.get("front")),
        "right": _labels(semantic_map.get("right")),
    }
    if action_type == "visible_anchor" and not semantic_views[action["view"]]:
        semantic_views[action["view"]] = [target]
    inferred_scene_summary = (
        cot_raw.get("target_localization_and_key_context")
        or cot_raw.get("ongoing_sub_instruction")
        or f"Current observation selected {target}."
    )
    normalized: dict[str, Any] = {
        "observation_id": _text(raw.get("observation_id"), 160, "latest"),
        "global_goal": _text(
            raw.get("global_goal") or route.get("statement"),
            240,
            "Continue the active navigation task.",
        ),
        "target_evidence": [
            _text(item, 80, "")
            for item in (raw.get("target_evidence") if isinstance(raw.get("target_evidence"), list) else [])[:8]
            if _text(item, 80, "")
        ],
        "navigation_task_type": navigation_task_type,
        "cot": cot,
        "previous_assessment": {
            "execution_status": _execution_status(assessment.get("execution_status")),
            "task_effect": _task_effect(assessment.get("task_effect")),
            "evidence": _text(assessment.get("evidence"), 240, "No prior execution evidence."),
        },
        "route_hypothesis": {
            "statement": _text(route.get("statement"), 240, "Follow the best available evidence."),
            "confidence": _score(route.get("confidence"), 0.5),
        },
        "candidates": candidates,
        "action": action,
        "semantic_map": {
            "scene_summary": _text(
                semantic_map.get("scene_summary"), 160, inferred_scene_summary
            ),
            **semantic_views,
            "route_node_type": (
                semantic_map.get("route_node_type")
                if semantic_map.get("route_node_type") in _ROUTE_NODE_TYPES
                else "none"
            ),
            "traversable_directions": [
                str(direction)
                for direction in (
                    semantic_map.get("traversable_directions")
                    if isinstance(semantic_map.get("traversable_directions"), list)
                    else []
                )[:3]
                if direction in _VIEWS
            ],
        },
        "max_control_steps": _integer(raw.get("max_control_steps"), 64, 1, 1000),
        "goal_tolerance_m": _number(raw.get("goal_tolerance_m"), 0.3, 0.01, 2.0),
        "anchor_max_depth_m": _number(raw.get("anchor_max_depth_m"), 5.0, 0.1, 20.0),
    }
    reflection = raw.get("self_reflection")
    if isinstance(reflection, dict) and re.fullmatch(
        r"subtask-[0-9]{3,}", str(reflection.get("subtask_id") or "")
    ):
        normalized["self_reflection"] = {
            "subtask_id": str(reflection["subtask_id"]),
            "completed": bool(reflection.get("completed")),
            "outcome": (
                reflection.get("outcome")
                if reflection.get("outcome") in {"completed", "partial", "failed"}
                else "partial"
            ),
            "feedback": _text(
                reflection.get("feedback"), 240, "Reassess the active subtask."
            ),
            "should_replan": bool(reflection.get("should_replan")),
        }
    planner_source = raw.get("planner_output")
    agentic_plan_alias = raw.get("agentic_plan")
    if not isinstance(planner_source, dict) and isinstance(agentic_plan_alias, dict):
        planner_source = agentic_plan_alias.get("planner_output")
    planner_output = _normalize_planner_output(planner_source)
    if planner_output is not None:
        normalized["planner_output"] = planner_output
    rejected = _text(raw.get("rejected_branch"), 160, "")
    if rejected:
        normalized["rejected_branch"] = rejected
    return normalized


def _normalize_planner_output(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    nested_plan = value.get("plan")
    if not isinstance(nested_plan, dict):
        nested_plan = {}
    raw_subtasks = value.get("subtasks")
    if not isinstance(raw_subtasks, list):
        raw_subtasks = nested_plan.get("subtasks")
    if not isinstance(raw_subtasks, list):
        raw_subtasks = []
    subtasks = []
    raw_selections: list[dict[str, Any]] = []
    for raw_subtask in raw_subtasks[:8]:
        if not isinstance(raw_subtask, dict):
            continue
        stage = raw_subtask.get("stage") or raw_subtask.get("type")
        task_type = raw_subtask.get("task_type") or {
            "approaching": "point_goal",
            "reaching": "object_goal",
            "interaction": "instruction_following",
        }.get(stage)
        if stage not in {"approaching", "reaching", "interaction"}:
            continue
        if task_type not in {
            "point_goal",
            "instruction_following",
            "object_goal",
            "poi_goal",
            "person_following",
        }:
            continue
        target_localization = raw_subtask.get("target_localization")
        if not isinstance(target_localization, dict):
            target_localization = {}
        subtask = {
            "stage": stage,
            "task_type": task_type,
            "instruction": _text(
                raw_subtask.get("instruction") or raw_subtask.get("description"),
                240,
                "Execute the active subtask.",
            ),
            "target": _text(
                raw_subtask.get("target")
                or raw_subtask.get("target_object")
                or raw_subtask.get("node_id")
                or target_localization.get("description")
                or raw_subtask.get("description"),
                120,
                "navigation goal",
            ),
            "success_condition": _text(
                raw_subtask.get("success_condition")
                or raw_subtask.get("completion_condition"),
                240,
                "Current evidence confirms this subtask is complete.",
            ),
        }
        regional_guidance = raw_subtask.get("regional_guidance")
        if not isinstance(regional_guidance, dict):
            regional_guidance = {}
        topology_node_id = _topology_node_id(
            raw_subtask.get("topology_node_id"),
            raw_subtask.get("node_id"),
            target_localization.get("topology_node_id"),
            regional_guidance.get("next_topology_node_id"),
        )
        if topology_node_id is not None:
            subtask["topology_node_id"] = topology_node_id
        for selection in (
            raw_subtask.get("node_selection"),
            regional_guidance.get("node_selection"),
        ):
            if isinstance(selection, dict):
                raw_selections.append(selection)
        subtasks.append(subtask)
    if not subtasks:
        return None
    global_instruction = _text(
        value.get("global_instruction")
        or nested_plan.get("global_instruction")
        or value.get("task_spec")
        or value.get("global_goal")
        or next(
            (
                item.get("instruction")
                for item in raw_subtasks
                if isinstance(item, dict) and item.get("instruction")
            ),
            None,
        ),
        240,
        "Continue the navigation task.",
    )
    destination_query = _text(
        value.get("destination_query")
        or value.get("target_place")
        or value.get("target_region")
        or global_instruction,
        120,
        "target region",
    )
    raw_active_target = value.get("active_target") or value.get(
        "active_target_evidence"
    )
    active_target = raw_active_target
    if isinstance(raw_active_target, dict):
        active_target = raw_active_target.get(
            "target_description"
        ) or raw_active_target.get(
            "target"
        )
    reaching_target = next(
        (
            subtask["target"]
            for subtask in subtasks
            if subtask["stage"] == "reaching"
        ),
        None,
    )
    object_goal = _text(
        value.get("object_goal") or reaching_target or active_target,
        120,
        "target object",
    )
    raw_selection = value.get("node_selection")
    if not isinstance(raw_selection, dict) and raw_selections:
        raw_selection = raw_selections[0]
    if not isinstance(raw_selection, dict) and isinstance(raw_active_target, dict):
        active_target_node_id = _topology_node_id(
            raw_active_target.get("selected_topology_node_id"),
            raw_active_target.get("topology_node_id"),
            raw_active_target.get("topology_node"),
        )
        if active_target_node_id is not None:
            raw_selection = {
                "selected_topology_node_id": active_target_node_id,
                "confidence": raw_active_target.get("confidence"),
                "matched_concepts": (
                    raw_active_target.get("landmarks")
                    if isinstance(raw_active_target.get("landmarks"), list)
                    else []
                ),
                "selection_reason": (
                    raw_active_target.get("selection_reason")
                    or raw_active_target.get("evidence_type")
                    or "The model associated this target evidence with the selected node."
                ),
            }
    node_selection = None
    if isinstance(raw_selection, dict):
        selected_topology_node_id = _topology_node_id(
            raw_selection.get("selected_topology_node_id")
        )
        if selected_topology_node_id is not None:
            node_selection = {
                "selected_topology_node_id": selected_topology_node_id,
                "confidence": _number(raw_selection.get("confidence"), 0.0, 0.0, 1.0),
                "matched_concepts": [
                    _text(item, 64, "")
                    for item in (
                        raw_selection.get("matched_concepts")
                        if isinstance(raw_selection.get("matched_concepts"), list)
                        else []
                    )[:12]
                    if _text(item, 64, "")
                ],
                "selection_reason": _text(
                    raw_selection.get("selection_reason") or raw_selection.get("reason"),
                    240,
                    "The selected node best satisfies the scene-task relation.",
                ),
            }
    if not any(subtask["stage"] == "reaching" for subtask in subtasks):
        topology_node_id = next(
            (
                subtask.get("topology_node_id")
                for subtask in reversed(subtasks)
                if subtask.get("topology_node_id")
            ),
            None,
        )
        reaching = {
            "stage": "reaching",
            "task_type": "object_goal",
            "instruction": f"Locate and verify {object_goal} in {destination_query}.",
            "target": object_goal,
            "success_condition": (
                f"Current visual evidence confirms {object_goal} in {destination_query}."
            ),
        }
        if topology_node_id is not None:
            reaching["topology_node_id"] = topology_node_id
        subtasks.append(reaching)
    normalized = {
        "global_instruction": global_instruction,
        "destination_query": destination_query,
        "object_goal": object_goal,
        "subtasks": subtasks,
        "node_local_radius_m": _number(
            value.get("node_local_radius_m"), 5.0, 0.1, 20.0
        ),
        "anchor_max_depth_m": _number(
            value.get("anchor_max_depth_m"), 5.0, 0.1, 20.0
        ),
    }
    if node_selection is not None:
        normalized["node_selection"] = node_selection
    return normalized


def _normalize_finish(raw: dict[str, Any]) -> dict[str, Any]:
    outcome = raw.get("outcome")
    normalized: dict[str, Any] = {
        "outcome": outcome if outcome in {"done", "failed", "cancelled"} else "done"
    }
    observation_id = raw.get("observation_id")
    if isinstance(observation_id, str) and observation_id:
        normalized["observation_id"] = observation_id
    scene_memory = raw.get("scene_memory")
    if isinstance(scene_memory, dict):
        normalized["scene_memory"] = {
            "room_hint": _text(scene_memory.get("room_hint"), 160, ""),
            "left": _free_labels(scene_memory.get("left"), 6),
            "front": _free_labels(scene_memory.get("front"), 6),
            "right": _free_labels(scene_memory.get("right"), 6),
            "distinctive_landmarks": _free_labels(
                scene_memory.get("distinctive_landmarks"), 8
            ),
        }
    return normalized


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    action_type = _action_type(raw, raw)
    return {
        "action_type": action_type,
        "intent": _intent(raw.get("intent"), action_type),
        "target": _target(raw, action_type),
        "task_relevance": _score(raw.get("task_relevance"), 0.5),
        "information_gain": _score(raw.get("information_gain"), 0.5),
        "revisit_penalty": _score(raw.get("revisit_penalty"), 0.0),
    }


def _intent(value: Any, action_type: str) -> str:
    if value in _INTENTS:
        return str(value)
    if action_type == "known_location":
        return "backtrack"
    return "goal_pursuit"


def _location_id(*values: Any) -> str:
    for value in values:
        if not isinstance(value, str):
            continue
        match = _LOCATION_ID_PATTERN.search(value)
        if match is not None:
            return match.group(0).casefold()
    return "location-000"


def _topology_node_id(*values: Any) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        match = _TOPOLOGY_NODE_ID_PATTERN.search(value)
        if match is not None:
            return match.group(0).casefold()
    return None


def _action_type(primary: dict[str, Any], fallback: dict[str, Any]) -> str:
    value = (
        primary.get("action_type")
        or primary.get("type")
        or primary.get("mode")
        or fallback.get("action_type")
        or fallback.get("type")
        or fallback.get("mode")
    )
    if value in _ACTIONS:
        return str(value)
    if (
        primary.get("topology_node_id")
        or primary.get("known_node")
        or primary.get("location_id")
        or primary.get("known_location_id")
        or primary.get("target_location")
    ):
        return "known_location"
    primitive_action = primary.get("action")
    if primary.get("viewpoint_action") or (
        isinstance(primitive_action, str) and primitive_action in _VIEWPOINT_ACTIONS
    ):
        return "viewpoint_adjustment"
    return "visible_anchor"


def _target(raw: dict[str, Any], action_type: str) -> str:
    aliases = (
        raw.get("target"),
        raw.get("target_location"),
        raw.get("anchor"),
        raw.get("topology_node_id"),
        raw.get("known_node"),
        raw.get("location_id"),
        raw.get("known_location_id"),
        raw.get("viewpoint_action"),
        raw.get("action"),
    )
    defaults = {
        "visible_anchor": "visible traversable anchor",
        "known_location": "location-000",
        "viewpoint_adjustment": "turn_left",
    }
    return _text(next((value for value in aliases if value), None), 120, defaults[action_type])


def _execution_status(value: Any) -> str:
    aliases = {
        "success": "completed",
        "done": "completed",
        "in_progress": "partial",
        "blocked": "failed",
    }
    normalized = aliases.get(str(value), value)
    return normalized if normalized in {"not_started", "completed", "partial", "failed"} else "partial"


def _task_effect(value: Any) -> str:
    aliases = {
        "useful_progress": "progress",
        "positive_progress": "progress",
        "advanced": "progress",
        "stagnation": "no_progress",
        "no_change": "no_progress",
        "rejected": "hypothesis_rejected",
        "collision": "blocked",
        "stuck": "blocked",
    }
    normalized = aliases.get(str(value), value)
    allowed = {"progress", "no_progress", "hypothesis_rejected", "blocked", "unknown"}
    return normalized if normalized in allowed else "unknown"


def _view(value: Any) -> str:
    aliases = {"forward": "front", "straight": "front"}
    normalized = aliases.get(str(value), value)
    return str(normalized) if normalized in _VIEWS else "front"


def _labels(value: Any) -> list[str]:
    return _free_labels(value, 4, 48)


def _free_labels(value: Any, limit: int, width: int = 80) -> list[str]:
    if not isinstance(value, list):
        return []
    return [label for item in value[:limit] if (label := _text(item, width, ""))]


def _text(value: Any, limit: int, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return (text or default)[:limit]


def _score(value: Any, default: float) -> float:
    return _number(value, default, 0.0, 1.0)


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _navigation_task_type(value: Any, cot: dict[str, Any]) -> str:
    allowed = {
        "point_goal",
        "instruction_following",
        "object_goal",
        "poi_goal",
        "person_following",
    }
    if value in allowed:
        return str(value)
    if "target_localization_and_key_context" in cot or "path_planning" in cot:
        return "object_goal"
    return "instruction_following"


def _normalize_cot(
    task_type: str,
    raw: dict[str, Any],
    assessment: dict[str, Any],
    route: dict[str, Any],
    action_type: str,
    target: str,
) -> dict[str, str] | None:
    if task_type == "instruction_following":
        return {
            "finished_sub_instruction": _text(
                raw.get("finished_sub_instruction"),
                160,
                _text(assessment.get("evidence"), 160, "No previous sub-instruction."),
            ),
            "ongoing_sub_instruction": _text(
                raw.get("ongoing_sub_instruction"),
                160,
                f"Execute {action_type}: {target}",
            ),
        }
    if task_type == "object_goal":
        return {
            "target_localization_and_key_context": _text(
                raw.get("target_localization_and_key_context"),
                240,
                "The target is not yet confirmed in the current tri-view.",
            ),
            "path_planning": _text(
                raw.get("path_planning"),
                240,
                _text(route.get("statement"), 240, "Follow the strongest current evidence."),
            ),
        }
    return None
