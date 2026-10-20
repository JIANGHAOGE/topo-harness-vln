from __future__ import annotations

import asyncio
import re
import time

import httpx
import pytest

from agentos.api.habitat_tool_server import HabitatToolService
from agentos.memory import SQLiteGraphMemory
from agentos.schemas.memory import MemoryNodeType, MemoryQuery
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation, Pose2D
from agentos.skills.navigation import OmniNavSkill
from agentos.skills.registry import SkillRegistry


class FakeObservation:
    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return {
            "trace": {"observation_id": "obs-0"},
            "rgb_refs": {"front": "front.jpg"},
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        }


class FakeAdapter:
    name = "fake-habitat"

    def __init__(self) -> None:
        self.sim = object()
        self.actions = []
        self.resets = 0

    async def observe(self, _trace):
        return FakeObservation()

    async def observe_live(self, _trace):
        return FakeObservation()

    def step(self, action: str) -> dict:
        self.actions.append(action)
        return {"action": action, "distance_m": 0.2 if action == "forward" else 0.0}

    def close(self) -> None:
        self.sim = None

    def start(self) -> None:
        self.sim = object()
        self.resets += 1


def test_service_observes_and_executes_only_bounded_actions():
    adapter = FakeAdapter()
    service = HabitatToolService(adapter, max_actions=2)

    observation = service.dispatch("POST", "/observe", {})
    result = service.dispatch("POST", "/execute_actions", {"actions": ["left", "forward"]})

    assert observation["observation"]["rgb_refs"]["front"] == "front.jpg"
    assert result["executed_actions"] == ["left", "forward"]
    assert adapter.actions == ["left", "forward"]
    assert service.dispatch("GET", "/relay/state", {})["observation"]["trace"]["observation_id"] == "obs-0"
    assert service.dispatch("GET", "/relay/live", {})["observation"]["rgb_refs"]["front"] == "front.jpg"


def test_service_rejects_raw_or_oversized_control_requests():
    service = HabitatToolService(FakeAdapter(), max_actions=2)

    with pytest.raises(ValueError, match="unsupported"):
        service.execute_actions({"actions": ["vx=1.0"]})
    with pytest.raises(ValueError, match="at most 2"):
        service.execute_actions({"actions": ["forward", "forward", "forward"]})


def test_service_never_silently_falls_back_when_agentos_is_not_configured():
    service = HabitatToolService(FakeAdapter(), skills=SkillRegistry())

    with pytest.raises(ValueError, match="AgentOS is not configured"):
        service.start_agent_task({"instruction": "Find the bedroom."})


def test_service_reset_restarts_at_configured_origin():
    adapter = FakeAdapter()
    service = HabitatToolService(adapter)

    result = service.dispatch("POST", "/reset", {})

    assert result["ok"] is True
    assert adapter.resets == 1
    assert result["observation"]["pose"]["x"] == 0.0


def test_service_exposes_capabilities_and_source_grounded_memory(tmp_path):
    memory = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    service = HabitatToolService(FakeAdapter(), skills=SkillRegistry(), memory=memory)

    service.observe()
    remembered = service.remember(
        {
            "source_observation_id": "obs-0",
            "node_type": "place",
            "title": "open hallway",
            "content": "An open hallway is visible ahead.",
            "confidence": 0.8,
        }
    )
    found = service.search_memory({"text": "hallway", "node_types": ["place"], "limit": 3})

    assert service.capabilities()["skills"] == []
    assert remembered["node"]["source_ref"] == "observation:obs-0"
    assert found["hits"][0]["node"]["title"] == "open hallway"
    memory.close()


def test_observation_ids_are_scoped_to_episode_after_reset(tmp_path):
    memory = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    service = HabitatToolService(FakeAdapter(), memory=memory)

    service.observe()
    first_episode = service.episode_id
    service.reset()
    second_episode = service.episode_id

    first = memory.search(MemoryQuery(episode_id=first_episode, node_types=[MemoryNodeType.SOURCE]))
    second = memory.search(MemoryQuery(episode_id=second_episode, node_types=[MemoryNodeType.SOURCE]))
    assert first[0].node.node_id != second[0].node.node_id
    memory.close()


def test_service_runs_omninav_only_against_cached_observation(tmp_path):
    class OmniAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.rgb_refs = {}
            for view in ("left", "front", "right"):
                path = tmp_path / f"{view}.jpg"
                path.write_bytes(view.encode())
                self.rgb_refs[view] = str(path)

        async def observe(self, trace):
            return Observation(
                trace=trace,
                rgb_refs=self.rgb_refs,
                pose=Pose2D(x=0.0, y=0.0, yaw=0.0, source="test", frame="test"),
                metadata={"embodiment": "habitat-gs"},
            )

        async def apply_skill_result(self, _result):
            return {
                "action": "move_forward",
                "collided": False,
                "distance_m": 0.2,
                "pose_before": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "pose_after": {"x": 0.2, "y": 0.0, "yaw": 0.0},
            }

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"trajectory": [[0.5, 0.0]], "arrive_pred": 0})
        )
    )
    adapter = OmniAdapter()
    registry = SkillRegistry()
    registry.register(OmniNavSkill("http://omninav.test/eval", embodiment=adapter, http_client=client))
    service = HabitatToolService(adapter, skills=registry)

    observed = service.dispatch("POST", "/observe", {})
    observation_id = observed["observation"]["trace"]["observation_id"]
    result = service.dispatch(
        "POST",
        "/skills/omninav",
        {"instruction": "Go through the doorway.", "observation_id": observation_id},
    )
    asyncio.run(client.aclose())

    assert result["ok"] is True
    assert service.relay_state()["omninav"]["instruction"] == "Go through the doorway."
    assert result["result"]["outputs"]["execution"]["action"] == "move_forward"
    with pytest.raises(ValueError, match="unknown or expired"):
        service.run_omninav({"instruction": "Go", "observation_id": "obs-stale"})


def test_agent_supervised_omninav_runs_atomic_subtasks_and_only_agent_stop_finishes(tmp_path, monkeypatch):
    class OmniAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.rgb_refs = {}
            for view in ("left", "front", "right"):
                path = tmp_path / f"task-{view}.jpg"
                path.write_bytes(view.encode())
                self.rgb_refs[view] = str(path)

        async def observe(self, trace):
            return Observation(
                trace=trace,
                rgb_refs=self.rgb_refs,
                pose=Pose2D(x=0.0, y=0.0, yaw=0.0, source="test", frame="test"),
                metadata={"embodiment": "habitat-gs"},
            )

        async def apply_skill_result(self, result):
            stopped = bool(result.outputs.get("omninav_stop"))
            return {
                "action": "hold" if stopped else "pure_pursuit",
                "collided": False,
                "distance_m": 0.0 if stopped else 0.2,
                "pose_before": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "pose_after": {"x": 0.0 if stopped else 0.2, "y": 0.0, "yaw": 0.0},
            }

    requests = 0
    request_bodies = []

    def respond(request):
        nonlocal requests
        requests += 1
        request_bodies.append(request.content)
        if requests == 1:
            return httpx.Response(200, json={"trajectory": "temporarily-invalid", "arrive_pred": 0})
        return httpx.Response(200, json={"trajectory": [[0.5, 0.0]], "arrive_pred": int(requests >= 3)})

    class FakeProvider:
        def __init__(self) -> None:
            self.post_calls = 0
            self.finish_calls = 0
            self.keyframe_counts = []
            self.closed = False

        async def complete(self, messages, *, response_schema=None):
            title = (response_schema or {}).get("title")
            if title == "SlowBrainSingleViewAudit":
                system = str(messages[0].content)
                view = next(
                    item
                    for item in ("left", "front", "right")
                    if f"robot-{item.upper()}" in system
                )
                parsed = {
                    "view": view,
                    "visible_objects": ["hallway", "robot"],
                    "navigable_semantic_anchors": ["visible hallway entrance"],
                    "target_relevant_evidence": ["A robot may be visible."],
                    "scene_summary": f"Current {view} camera scene.",
                }
            elif title == "OmniTaskPlan":
                parsed = {
                    "decision": "run_omninav",
                    "task_summary": "Find the robot.",
                    "task_success_conditions": ["The robot is visibly reached."],
                    "subtasks": ["Navigate towards the visible hallway entrance"],
                    "initial_visible_anchor": "the visible hallway entrance",
                    "initial_anchor_view": "front",
                    "initial_subtask": "Navigate towards the visible hallway entrance",
                    "initial_subtask_success_conditions": ["The hallway is reached."],
                    "initial_scene_description": "A hallway is visible ahead.",
                    "rationale_summary": "Use the hallway as the first local target.",
                }
            elif title == "OmniFinishVerification":
                self.finish_calls += 1
                should_finish = self.finish_calls == 2
                parsed = {
                    "verdict": "finish" if should_finish else "continue",
                    "evidence_view": "front" if should_finish else "none",
                    "evidence_summary": (
                        "The requested robot is unambiguously visible in front."
                        if should_finish
                        else "The current evidence is insufficient for global completion."
                    ),
                    "next_visible_anchor": (
                        "the visible robot" if should_finish else "the doorway beyond the ambiguous object"
                    ),
                    "next_anchor_view": "front",
                    "next_subtask": (
                        "Navigate towards the visible robot"
                        if should_finish
                        else "Navigate towards the doorway beyond the ambiguous object"
                    ),
                    "next_subtask_success_conditions": [
                        "The requested target is unambiguously visible in the current views."
                    ],
                }
            else:
                self.post_calls += 1
                text = "\n".join(
                    str(item.get("text", ""))
                    for item in messages[-1].content
                    if item.get("type") == "text"
                )
                keyframes = [
                    (int(frame_index), observation_id)
                    for frame_index, observation_id in re.findall(
                        r"OMNINAV_TRAJECTORY_KEYFRAME_\d+ frame_index=(\d+) observation_id=([^\s]+)",
                        text,
                    )
                ]
                self.keyframe_counts.append(len(keyframes))
                should_stop = self.post_calls >= 2
                parsed = {
                    "previous_subtask_status": "success",
                    "evidence_summary": "The local target was reached.",
                    "current_scene_description": "The robot is visible ahead.",
                    "omni_action_description": "OmniNav traversed the local route and stopped.",
                    "trajectory_descriptions": [
                        {
                            "frame_index": frame_index,
                            "observation_id": observation_id,
                            "description": f"Trajectory view at frame {frame_index}.",
                        }
                        for frame_index, observation_id in keyframes
                    ],
                    "decision": "stop" if should_stop else "next_subtask",
                    "next_visible_anchor": "the visible robot",
                    "next_anchor_view": "front",
                    "next_subtask": "Navigate towards the visible robot",
                    "next_subtask_success_conditions": (
                        ["The global task is complete."] if should_stop else ["The robot is reached."]
                    ),
                    "rationale_summary": "Global task complete." if should_stop else "Continue to the robot.",
                }
            return LLMResponse(model="fake-agent", content="{}", raw={}, parsed_json=parsed)

        async def close(self):
            self.closed = True

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    adapter = OmniAdapter()
    registry = SkillRegistry()
    registry.register(OmniNavSkill("http://omninav.test/eval", embodiment=adapter, http_client=client))
    memory = SQLiteGraphMemory(tmp_path / "task-memory.sqlite3")
    provider = FakeProvider()
    monkeypatch.setattr(
        "agentos.api.habitat_tool_server.build_provider_from_config",
        lambda _config: provider,
    )
    service = HabitatToolService(adapter, skills=registry, memory=memory, agent_config={"llm": {}})

    started = service.start_omninav_task(
        {
            "instruction": "Find the robot.",
            "max_steps": 10,
            "max_omninav_steps_per_skill": 5,
        }
    )
    deadline = time.monotonic() + 2.0
    status = started
    while status["status"] in {"starting", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
        status = service.navigation_task_status()
    asyncio.run(client.aclose())

    assert status["status"] == "completed"
    assert status["reason"] == "finish_verified"
    assert status["step"] == 4
    assert status["distance_m"] == pytest.approx(0.2)
    assert requests == 5
    assert b'"reset": true' in request_bodies[0]
    assert b'"reset": false' in request_bodies[2]
    assert b'"reset": true' in request_bodies[-1]
    assert status["transient_error"] is None
    assert status["retry"] is None
    assert status["agent_calls"] == 6
    assert status["omninav_stop"] is True
    assert status["omninav_stop_scope"] == "local_evidence_only"
    assert len(status["trajectory_memory"]) == 4
    assert provider.post_calls == 3
    assert provider.finish_calls == 2
    assert all(count <= 4 for count in provider.keyframe_counts)
    assert provider.closed is True
    memories = memory.search(
        MemoryQuery(
            task_id=status["task_id"],
            node_types=[MemoryNodeType.SEMANTIC_EVENT],
            limit=10,
        )
    )
    assert len(memories) == 4
    assert adapter.resets == 0
    memory.close()


def test_omninav_accepts_null_trajectory_on_model_stop():
    outputs = OmniNavSkill._normalize_response(
        {"trajectory": None, "model_trajectory": None, "episode_done": True, "omninav_arrive_pred": 0}
    )

    assert outputs["trajectory"] == []
    assert outputs["omninav_stop"] is True
