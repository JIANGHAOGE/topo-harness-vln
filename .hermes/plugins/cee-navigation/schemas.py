STATUS = {
    "name": "cee_navigation_status",
    "description": "Check whether the CEE-AgentOS Habitat navigation service is ready.",
    "parameters": {"type": "object", "properties": {}},
}

START_AGENT_TASK = {
    "name": "cee_start_agent_task",
    "description": (
        "Submit one global embodied task to the server-side AgentOS Harness. Hermes does not plan or execute "
        "skills; it receives only compressed boundary events and the final verified result. The canonical "
        "argument is instruction, for example {\"instruction\": \"go to office to find a fridge\"}. "
        "Legacy task, task_description, and global_goal names are accepted as compatibility aliases."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "minLength": 1,
                "description": "Canonical complete embodied-navigation instruction from the user.",
            },
            "task": {
                "type": "string",
                "minLength": 1,
                "description": "Deprecated alias for instruction.",
            },
            "task_description": {
                "type": "string",
                "minLength": 1,
                "description": "Deprecated alias for instruction.",
            },
            "global_goal": {
                "type": "string",
                "minLength": 1,
                "description": "Deprecated alias for instruction.",
            },
            "success_conditions": {"type": "array", "items": {"type": "string"}},
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 1000},
            "timeout_s": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 7200,
                "default": 1800,
            },
            "max_plan_revisions": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 20,
            },
            "wait_for_completion": {
                "type": "boolean",
                "default": True,
                "description": "Wait internally and return a compact Agent reasoning timeline plus final result.",
            },
            "wait_timeout_s": {
                "type": "number",
                "minimum": 30,
                "maximum": 3600,
                "default": 900,
            },
        },
        "anyOf": [
            {"required": ["instruction"]},
            {"required": ["task"]},
            {"required": ["task_description"]},
            {"required": ["global_goal"]},
        ],
    },
}

NAVIGATION_TASK_STATUS = {
    "name": "cee_navigation_task_status",
    "description": "Read the AgentOS plan, active Skill Runner session, verifier result, and final state.",
    "parameters": {"type": "object", "properties": {}},
}

CANCEL_AGENT_TASK = {
    "name": "cee_cancel_agent_task",
    "description": "Request cancellation of the active AgentOS task without resetting the environment.",
    "parameters": {"type": "object", "properties": {}},
}
