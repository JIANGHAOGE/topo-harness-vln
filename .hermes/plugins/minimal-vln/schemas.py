OBSERVE = {
    "name": "observe",
    "description": "Look through the robot's left, front, and right RGB cameras without moving.",
    "parameters": {"type": "object", "properties": {}},
}

STEP = {
    "name": "step",
    "description": (
        "Execute exactly one Habitat action. 0=permanent STOP, 1=forward 0.25m, "
        "2=left 15 degrees, 3=right 15 degrees. The actions list must contain one item."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 3},
            },
        },
        "required": ["actions"],
    },
}
