from __future__ import annotations

from agentos.memory.scene import TaskSceneMemory
from agentos.schemas.hierarchical import SceneSemanticDescription
from agentos.schemas.observation import Observation, Pose2D


def _observation(make_trace, observation_id: str, x: float, z: float) -> Observation:
    return Observation(
        trace=make_trace(observation_id=observation_id),
        pose=Pose2D(x=x, y=z, yaw=0.0, source="test", frame="habitat_world_xz"),
    )


def test_scene_memory_detects_revisit_and_repeated_anchor(make_trace):
    memory = TaskSceneMemory("task-1")
    scene = SceneSemanticDescription(
        room_hint="laboratory hallway",
        left=["white cabinets"],
        front=["open doorway"],
        right=["washing machine"],
        distinctive_landmarks=["red barrier"],
    )

    first = memory.record_visit(
        _observation(make_trace, "obs-1", 1.0, 2.0),
        scene,
        selected_anchor="open doorway [front]",
    )
    second = memory.record_visit(
        _observation(make_trace, "obs-2", 1.2, 2.1),
        scene,
        selected_anchor="open doorway [front]",
    )

    assert first["revisited"] is False
    assert second["revisited"] is True
    assert second["anchor_repeated"] is True
    assert memory.summary()["unique_scenes"] == 1
    assert memory.summary()["total_visits"] == 2


def test_scene_memory_keeps_semantically_distinct_nearby_places_separate(make_trace):
    memory = TaskSceneMemory("task-1")
    kitchen = SceneSemanticDescription(room_hint="kitchen", front=["white cabinets"])
    bedroom = SceneSemanticDescription(room_hint="bedroom", front=["bed and nightstand"])

    memory.record_visit(
        _observation(make_trace, "obs-1", 0.0, 0.0),
        kitchen,
        selected_anchor="hallway [front]",
    )
    result = memory.record_visit(
        _observation(make_trace, "obs-2", 0.8, 0.0),
        bedroom,
        selected_anchor="bed [front]",
    )

    assert result["revisited"] is False
    assert memory.summary()["unique_scenes"] == 2
