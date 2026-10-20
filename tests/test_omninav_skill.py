from __future__ import annotations

import asyncio
import json

import httpx

from agentos.core.events import InMemoryEventBus
from agentos.schemas.enums import SkillStatus
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.skill import SkillRequest
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.navigation import OmniNavSkill


class FakeEmbodiment:
    def __init__(self) -> None:
        self.received = []

    async def apply_skill_result(self, result):
        self.received.append(result)
        if result.outputs["omninav_stop"]:
            return {
                "action": "hold",
                "reason": "model_stop_or_empty_trajectory",
                "collided": False,
                "distance_m": 0.0,
                "pose_before": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "pose_after": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            }
        return {
            "action": "move_forward",
            "reason": "first_usable_local_waypoint",
            "collided": False,
            "distance_m": 0.2,
            "pose_before": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "pose_after": {"x": 0.2, "y": 0.0, "yaw": 0.0},
        }


def _run_skill(skill, request, observation):
    context = SkillContext(event_bus=InMemoryEventBus(), cancellation=CancellationToken())
    return asyncio.run(skill.run(request, observation, context))


def _observation(make_trace, tmp_path, *, include_right: bool = True):
    refs = {}
    for view in ("left", "front", "right"):
        if view == "right" and not include_right:
            continue
        path = tmp_path / f"rgb_{view}.jpg"
        path.write_bytes(f"jpeg-{view}".encode())
        refs[view] = str(path)
    depth = tmp_path / "depth_front.png"
    depth.write_bytes(b"png-depth")
    return Observation(
        trace=make_trace(),
        rgb_refs=refs,
        depth_refs={"front": str(depth)},
        pose=Pose2D(x=1.0, y=2.0, yaw=0.25, source="test", frame="habitat_world_xz"),
        metadata={"embodiment": "habitat-gs"},
    )


def _request(observation):
    return SkillRequest(
        trace=observation.trace.for_skill(skill_run_id="omninav-test"),
        skill_name="NavigationSkill",
        command="Walk through the doorway.",
        inputs={"instruction": "Walk through the doorway.", "reset": True, "frame_index": 7},
    )


def test_omninav_skill_sends_official_three_view_fields_and_executes_once(make_trace, tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "model_trajectory": [[0.4, 0.05], [0.8, 0.1]],
                "omninav_arrive_pred": 0,
                "model_infer_sec": 0.21,
                "model_goal_r": 0.4,
                "model_goal_theta": 0.12,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embodiment = FakeEmbodiment()
    observation = _observation(make_trace, tmp_path)
    skill = OmniNavSkill("http://omninav.test/eval", embodiment=embodiment, http_client=client)

    result = _run_skill(skill, _request(observation), observation)
    asyncio.run(client.aclose())

    body = captured["body"]
    assert b'name="image"' in body
    assert b'name="image_left"' in body
    assert b'name="image_right"' in body
    assert b'name="depth"' in body
    assert b'"rgb_view_order": ["left", "front", "right"]' in body
    assert result.status is SkillStatus.SUCCESS
    assert result.outputs["trajectory"] == [[0.4, 0.05], [0.8, 0.1]]
    assert result.outputs["trajectory_coord_frame"] == "[forward,left] meters"
    assert result.outputs["execution"]["action"] == "move_forward"
    assert result.outputs["execution_report"]["distance_m"] == 0.2
    assert len(embodiment.received) == 1


def test_omninav_stop_is_local_evidence_and_holds_embodiment(make_trace, tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"trajectory": [], "omninav_arrive_pred": 1})
        )
    )
    embodiment = FakeEmbodiment()
    observation = _observation(make_trace, tmp_path)
    result = _run_skill(
        OmniNavSkill("http://omninav.test/eval", embodiment=embodiment, http_client=client),
        _request(observation),
        observation,
    )
    asyncio.run(client.aclose())

    assert result.status is SkillStatus.SUCCESS
    assert result.outputs["omninav_stop"] is True
    assert result.outputs["execution"]["action"] == "hold"
    assert result.stop_reason == "omninav_local_stop"
    assert result.outputs["execution_report"]["stop_source"] == "omninav_local_stop"


def test_omninav_skill_rejects_missing_view_without_request(make_trace, tmp_path):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"trajectory": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    observation = _observation(make_trace, tmp_path, include_right=False)
    result = _run_skill(OmniNavSkill("http://omninav.test/eval", http_client=client), _request(observation), observation)
    asyncio.run(client.aclose())

    assert result.status is SkillStatus.FAILURE
    assert "missing required RGB view" in (result.error or "")
    assert calls == 0


def test_omninav_skill_rejects_non_finite_trajectory(make_trace, tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=json.dumps({"trajectory": [[float("nan"), 0.0]]}))
        )
    )
    observation = _observation(make_trace, tmp_path)
    result = _run_skill(OmniNavSkill("http://omninav.test/eval", http_client=client), _request(observation), observation)
    asyncio.run(client.aclose())

    assert result.status is SkillStatus.FAILURE
    assert "not finite" in (result.error or "")
