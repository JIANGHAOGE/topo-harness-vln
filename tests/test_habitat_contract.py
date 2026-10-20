from __future__ import annotations

import asyncio
from pathlib import Path

from agentos.embodiments.habitat_gs import HabitatGSAdapter, HabitatGSConfig
from agentos.planning.static import StaticPlanner
from agentos.schemas.observation import Observation
from agentos.schemas.task import TaskRequest


def test_habitat_action_aliases_are_explicit(tmp_path: Path):
    adapter = HabitatGSAdapter(
        HabitatGSConfig(dataset_root=tmp_path, scene_id="test", output_dir=tmp_path / "frames")
    )
    assert adapter.ACTION_ALIASES["forward"] == "move_forward"
    assert adapter.ACTION_ALIASES["back"] == "move_backward"
    assert adapter.ACTION_ALIASES["left"] == "turn_left"
    assert adapter.SENSOR_ORDER == ("left", "front", "right")
    assert adapter.SENSOR_YAW_DEG == {"left": 90.0, "front": 0.0, "right": -90.0}


def test_static_planner_passes_explicit_skill_inputs(make_trace):
    trace = make_trace()
    task = TaskRequest(
        trace=trace,
        instruction="Smoke route",
        metadata={"skill_inputs": {"actions": ["forward", "stop"]}},
    )
    observation = Observation(trace=trace)
    plan = asyncio.run(StaticPlanner().plan(task, observation))
    assert plan.steps[0].inputs["actions"] == ["forward", "stop"]
