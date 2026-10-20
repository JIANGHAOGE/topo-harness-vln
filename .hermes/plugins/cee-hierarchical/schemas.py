STATUS = {
    "name": "hierarchical_status",
    "description": "Check the independent hierarchical Habitat navigation service.",
    "parameters": {"type": "object", "properties": {}},
}

CAPABILITIES = {
    "name": "hierarchical_capabilities",
    "description": "List the structured skills in the hierarchical navigation version.",
    "parameters": {"type": "object", "properties": {}},
}

OBSERVE = {
    "name": "hierarchical_observe",
    "description": "Capture synchronized left/front/right RGB and depth observations.",
    "parameters": {"type": "object", "properties": {}},
}

SEMANTIC_MAP_PROPERTIES = {
    "scene_summary": {
        "type": "string",
        "minLength": 1,
        "maxLength": 160,
        "description": "Current observation evidence; this is not the canonical map-node identity.",
    },
    "left": {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 48},
        "maxItems": 4,
    },
    "front": {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 48},
        "maxItems": 4,
    },
    "right": {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 48},
        "maxItems": 4,
    },
    "route_node_type": {
        "type": "string",
        "enum": ["none", "open_space_center", "junction"],
        "default": "none",
        "description": (
            "Mark open_space_center only when the robot is roughly central in a broad area; mark junction only "
            "when multiple distinct traversable continuations are visible."
        ),
    },
    "traversable_directions": {
        "type": "array",
        "items": {"type": "string", "enum": ["left", "front", "right"]},
        "maxItems": 3,
        "description": "Robot-relative directions with clearly visible traversable floor or an open threshold.",
    },
}

SEMANTIC_MAP = {
    "type": "object",
    "properties": SEMANTIC_MAP_PROPERTIES,
    "required": ["scene_summary", "left", "front", "right"],
}

ADJUST_VIEWPOINT = {
    "name": "adjust_viewpoint",
    "description": (
        "Execute one 90-degree turn or one bounded translation to reveal a more useful view, then return a fresh "
        "synchronized left/front/right observation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string", "minLength": 1},
            "action": {
                "type": "string",
                "enum": ["turn_left", "turn_right", "move_forward", "move_backward"],
            },
            "rotation_degrees": {
                "type": "number",
                "minimum": 15,
                "maximum": 180,
                "default": 90,
                "description": "Turn magnitude; ignored for translations.",
            },
            "semantic_map": SEMANTIC_MAP,
        },
        "required": ["observation_id", "action", "semantic_map"],
    },
}

INSPECT_LOCALIZATION_MAP = {
    "name": "inspect_localization_map",
    "description": (
        "Associate concise current-scene semantics with Habitat pose and top-down navmesh localization, detect revisits, "
        "and return the map plus fresh left/front/right views."
    ),
    "parameters": {
        "type": "object",
        "properties": SEMANTIC_MAP_PROPERTIES,
        "required": ["scene_summary", "left", "front", "right"],
    },
}

NAVIGATE_KNOWN_LOCATION = {
    "name": "navigate_to_known_location",
    "description": (
        "Navigate to a stable backbone node or a task-scoped historical observation. Use topology_node_id for map "
        "nodes and location_id only for trajectory-keyframe recall or backtracking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topology_node_id": {"type": "string", "pattern": "^backbone-[0-9]{3,}$"},
            "location_id": {"type": "string", "pattern": "^location-[0-9]{3,}$"},
            "observation_id": {"type": "string", "minLength": 1},
            "semantic_map": SEMANTIC_MAP,
            "max_control_steps": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 64},
            "goal_tolerance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 2, "default": 0.3},
        },
        "required": ["observation_id", "semantic_map"],
        "oneOf": [
            {"required": ["topology_node_id"]},
            {"required": ["location_id"]},
        ],
    },
}

SCENE_MEMORY = {
    "type": "object",
    "properties": {
        "room_hint": {"type": "string"},
        "left": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "front": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "right": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "distinctive_landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["room_hint", "left", "front", "right", "distinctive_landmarks"],
}

GROUND_ANCHOR = {
    "name": "ground_visible_anchor",
    "description": (
        "Call Qwen vision once over the synchronized left/front/right views. The supplied view is a planner hint; "
        "Qwen returns the camera where the anchor is actually visible and marks the detection bbox bottom center."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string"},
            "anchor": {"type": "string"},
            "view": {"type": "string", "enum": ["left", "front", "right"]},
        },
        "required": ["observation_id", "anchor", "view"],
    },
}

PROJECT_PIXEL = {
    "name": "project_anchor_pixel",
    "description": "Use matching Habitat depth to convert an anchor pixel to camera, robot, and world coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string"},
            "view": {"type": "string", "enum": ["left", "front", "right"]},
            "x_px": {"type": "integer", "minimum": 0, "maximum": 639},
            "y_px": {"type": "integer", "minimum": 0, "maximum": 479},
        },
        "required": ["observation_id", "view", "x_px", "y_px"],
    },
}

PLAN_PATH = {
    "name": "plan_anchor_path",
    "description": "Snap a world point to Habitat navmesh and calculate a shortest path.",
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string"},
            "world_point_xyz_m": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["observation_id", "world_point_xyz_m"],
    },
}

EXECUTE_PATH = {
    "name": "execute_anchor_path",
    "description": "Execute a Habitat world-coordinate path with Pure Pursuit and no per-frame LLM calls.",
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string"},
            "world_path_xyz_m": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "minItems": 1,
            },
            "max_control_steps": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 64},
            "goal_tolerance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 2, "default": 0.3},
            "control_period_s": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.1},
        },
        "required": ["observation_id", "world_path_xyz_m"],
    },
}

NAVIGATE_ANCHOR = {
    "name": "navigate_to_visible_anchor",
    "description": (
        "Search all three synchronized views for one currently visible anchor, generate a point in the matched "
        "camera, then use the shared point-navigation skill to plan and execute it. Returns the latest semantic "
        "map, up to four trajectory keyframes, and fresh views."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "anchor": {"type": "string", "minLength": 1},
            "view": {"type": "string", "enum": ["left", "front", "right"]},
            "observation_id": {"type": ["string", "null"]},
            "semantic_map": SEMANTIC_MAP,
            "scene_memory": SCENE_MEMORY,
            "max_control_steps": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 64},
            "goal_tolerance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 2, "default": 0.3},
            "anchor_max_depth_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 20,
                "default": 5.0,
            },
        },
        "required": ["anchor", "view", "semantic_map"],
    },
}

BEGIN_NAVIGATION_TASK = {
    "name": "begin_navigation_task",
    "description": (
        "Initialize one navigation task and return the current synchronized tri-view plus ABot-N0 four-layer "
        "Topo-Memory. Call exactly once. Do not decompose the task before this result: the first "
        "advance_navigation call must build planner_output from instruction, current observation, Episodic "
        "Memory, and Topo-Memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "global_goal": {"type": "string", "minLength": 1, "maxLength": 240},
        },
        "required": ["global_goal"],
        "additionalProperties": False,
    },
}

RECALL_NAVIGATION_MEMORY = {
    "name": "recall_navigation_memory",
    "description": (
        "Recall at most four task-scoped historical keyframe images without moving the robot. Use only when the "
        "current views, latest subtask keyframes, and structured memory are insufficient. task_overview samples "
        "the complete task trajectory; latest_subtask replays the most recent navigation run; task supports "
        "location_id or semantic query filtering. Each trajectory location_id is an exact L evidence position; "
        "filtering by it returns its stored LEFT/FRONT/RIGHT views when available, covering objects that disappeared "
        "before the destination B node was reached."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["task_overview", "latest_subtask", "task"],
                "default": "task_overview",
            },
            "location_id": {
                "type": ["string", "null"],
                "pattern": "^location-[0-9]{3,}$",
            },
            "query": {"type": "string", "maxLength": 160, "default": ""},
            "limit": {"type": "integer", "minimum": 1, "maximum": 4, "default": 4},
        },
    },
}

SEARCH_SEMANTIC_MAP = {
    "name": "search_semantic_map",
    "description": (
        "Search orientation-invariant map-node knowledge before exploring. Returns stable topology IDs and current "
        "semantics matching place type, function, objects, landmarks, aliases, or task affordances."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 160},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
    },
}

INSTRUCTION_FOLLOWING_COT = {
    "type": "object",
    "properties": {
        "finished_sub_instruction": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
        },
        "ongoing_sub_instruction": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
        },
    },
    "required": ["finished_sub_instruction", "ongoing_sub_instruction"],
    "additionalProperties": False,
}

OBJECT_GOAL_COT = {
    "type": "object",
    "properties": {
        "target_localization_and_key_context": {
            "type": "string",
            "minLength": 1,
            "maxLength": 240,
        },
        "path_planning": {
            "type": "string",
            "minLength": 1,
            "maxLength": 240,
        },
    },
    "required": ["target_localization_and_key_context", "path_planning"],
    "additionalProperties": False,
}

SEMANTIC_NODE_SELECTION = {
    "type": "object",
    "description": (
        "Model-grounded destination choice over fixed B nodes. Judge the requested place/function and object/POI "
        "as one relation; map place labels are evidence, not infallible literal constraints."
    ),
    "properties": {
        "selected_topology_node_id": {
            "type": "string",
            "pattern": "^backbone-[0-9]{3,}$",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "matched_concepts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 64},
            "maxItems": 12,
        },
        "selection_reason": {"type": "string", "minLength": 1, "maxLength": 240},
    },
    "required": [
        "selected_topology_node_id",
        "confidence",
        "matched_concepts",
        "selection_reason",
    ],
    "additionalProperties": False,
}

AGENTIC_PLANNER_OUTPUT = {
    "type": "object",
    "description": (
        "ABot-N0 Planner output. Required on the first macro decision and again only when Self-Reflector "
        "sets should_replan=true. Select topology IDs only from the supplied four-layer Topo-Memory."
    ),
    "properties": {
        "global_instruction": {"type": "string", "minLength": 1, "maxLength": 240},
        "destination_query": {"type": "string", "minLength": 1, "maxLength": 120},
        "object_goal": {"type": "string", "minLength": 1, "maxLength": 120},
        "node_selection": SEMANTIC_NODE_SELECTION,
        "subtasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "contains": {
                "type": "object",
                "properties": {"stage": {"const": "reaching"}},
                "required": ["stage"],
            },
            "items": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "enum": ["approaching", "reaching", "interaction"],
                    },
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "point_goal",
                            "instruction_following",
                            "object_goal",
                            "poi_goal",
                            "person_following",
                        ],
                    },
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 240},
                    "target": {"type": "string", "minLength": 1, "maxLength": 120},
                    "topology_node_id": {
                        "type": ["string", "null"],
                        "pattern": "^backbone-[0-9]{3,}$",
                    },
                    "success_condition": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 240,
                    },
                },
                "required": [
                    "stage",
                    "task_type",
                    "instruction",
                    "target",
                    "success_condition",
                ],
                "allOf": [
                    {
                        "if": {"properties": {"stage": {"const": "approaching"}}},
                        "then": {"properties": {"task_type": {"const": "point_goal"}}},
                    },
                    {
                        "if": {"properties": {"stage": {"const": "reaching"}}},
                        "then": {
                            "properties": {
                                "task_type": {"enum": ["object_goal", "poi_goal"]}
                            }
                        },
                    },
                    {
                        "if": {"properties": {"stage": {"const": "interaction"}}},
                        "then": {
                            "properties": {
                                "task_type": {
                                    "enum": [
                                        "instruction_following",
                                        "person_following",
                                    ]
                                }
                            }
                        },
                    },
                ],
                "additionalProperties": False,
            },
        },
        "node_local_radius_m": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 20,
            "default": 5.0,
        },
        "anchor_max_depth_m": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 20,
            "default": 5.0,
        },
    },
    "required": [
        "global_instruction",
        "destination_query",
        "object_goal",
        "node_selection",
        "subtasks",
    ],
    "additionalProperties": False,
}

ADVANCE_NAVIGATION = {
    "name": "advance_navigation",
    "description": (
        "Submit one ABot-N0 Actor decision and the task-specific ABot-N1 CoT at a macro boundary. "
        "The runtime deterministically routes it to visible-anchor, "
        "known-location, or viewpoint-adjustment execution and returns fresh views plus bounded memory. "
        "The result exposes the final Runtime action as executed_action.mode and marks whether it was selected by "
        "the Agent or replaced by Runtime policy. "
        "Prefer the visible anchor that most directly advances the global task. After each macro-step, reassess the "
        "latest trajectory and current three views, then select the next anchor in the chain. Use a historical "
        "evidence location or 90-degree viewpoint adjustment when it is more informative, and use a backbone node "
        "only for an explicit region switch or recovery. "
        "A mapped semantic node is not considered visited until this task reaches it. Runtime topology and node "
        "semantics are frozen for the Hermes lifetime: only prebuilt backbone nodes are route nodes. The task start "
        "is a historical L-memory position, and execution never creates or annotates a backbone node."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string", "minLength": 1},
            "planner_output": {
                "type": ["object", "null"],
                "description": (
                    "Canonical ABot-N0 Planner output. Required on the first macro decision and after replan. "
                    "Structural aliases are normalized before strict Runtime validation."
                ),
            },
            "agentic_plan": {
                "type": ["object", "null"],
                "description": "Compatibility wrapper; nested planner_output is normalized before Runtime.",
            },
            "active_subtask": {},
            "navigation_task_type": {
                "type": ["string", "null"],
                "description": "Use the active typed subtask from agentic_plan.",
            },
            "cot": {
                "description": (
                    "Exact ABot-N1 task-specific CoT. Instruction following uses done/current; object goal uses "
                    "localization/path. Point, POI, and person-following use null."
                ),
                "type": ["object", "null"],
            },
            "self_reflection": {
                "description": (
                    "ABot-N0 Self-Reflector result for the previously active macro-subtask; null before the first action."
                ),
                "type": ["object", "null"],
            },
            "action": {
                "type": "object",
                "minProperties": 1,
                "description": (
                    "One macro action. Canonical modes are visible_anchor, known_location, and "
                    "viewpoint_adjustment. Structural aliases are normalized before strict Runtime validation."
                ),
            },
            "semantic_map": {
                "type": ["object", "null"],
                "description": "Bounded current-scene labels normalized before Runtime validation.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

FINISH_TASK = {
    "name": "finish_navigation_task",
    "description": (
        "Request global completion or terminal failure from the independent AgentOS Finish Verifier. Local Skill "
        "success, A* arrival, or Slow STOP cannot authorize completion. Rejected requests preserve task memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": ["done", "failed", "cancelled"]},
            "observation_id": {"type": ["string", "null"]},
            "scene_memory": SCENE_MEMORY,
        },
        "required": [],
    },
}
