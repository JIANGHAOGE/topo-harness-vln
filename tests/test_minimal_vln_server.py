from __future__ import annotations

from agentos.api.minimal_vln_server import MinimalVLNService
from agentos.minimal_vln_prompt import build_interactive_prompt, build_prompt


class FakeObservation:
    def __init__(self) -> None:
        self.rgb_refs = {
            "left": "/logs/minimal_vln/left.jpg",
            "front": "/logs/minimal_vln/front.jpg",
            "right": "/logs/minimal_vln/right.jpg",
        }

    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return {
            "trace": {"observation_id": "obs-0"},
            "rgb_refs": self.rgb_refs,
            "topdown_ref": "/logs/minimal_vln/topdown.png",
            "pose": {"x": 1.0, "y": 2.0, "yaw": 0.5},
        }


class FakeAdapter:
    def __init__(self) -> None:
        self.sim = object()
        self.actions: list[str] = []
        self.resets = 0

    async def observe(self, _trace):
        return FakeObservation()

    async def observe_live(self, _trace):
        return FakeObservation()

    def step(self, action: str) -> dict:
        self.actions.append(action)
        return {"action": action}

    def close(self) -> None:
        self.sim = None

    def start(self) -> None:
        self.sim = object()
        self.resets += 1


def test_model_contract_exposes_three_rgb_images_and_bounded_action_summary():
    adapter = FakeAdapter()
    service = MinimalVLNService(adapter, action_budget=5)

    observation = service.dispatch("POST", "/observe", {})
    action = service.dispatch("POST", "/step", {"actions": [2]})

    assert observation == {
        "ok": True,
        "view_order": ["left", "front", "right"],
        "image_paths": {
            "left": "/logs/minimal_vln/left.jpg",
            "front": "/logs/minimal_vln/front.jpg",
            "right": "/logs/minimal_vln/right.jpg",
        },
    }
    assert action == {"executed": 1, "remaining_budget": 4, "terminal": False}
    assert adapter.actions == ["left"]
    assert "pose" not in observation
    assert "collided" not in action

    camera = service.health()["camera"]
    assert camera["width"] == 640
    assert camera["height"] == 480
    assert camera["hfov_deg"] == 120.0
    assert camera["vfov_deg"] == 90.0
    assert camera["yaw_deg"] == {"left": 90.0, "front": 0.0, "right": -90.0}


def test_stop_is_permanent_until_explicit_episode_reset():
    adapter = FakeAdapter()
    service = MinimalVLNService(adapter)

    stopped = service.step({"actions": [0]})

    assert stopped["executed"] == 1
    assert stopped["terminal"] is True
    assert adapter.actions == ["stop"]
    try:
        service.observe()
    except ValueError as exc:
        assert "episode is terminal" in str(exc)
    else:
        raise AssertionError("terminal episode accepted another observation")
    service.reset()
    assert service.observe()["ok"] is True
    assert adapter.resets == 1


def test_step_rejects_unknown_or_multi_action_lists():
    service = MinimalVLNService(FakeAdapter())

    for payload, message in [
        ({"actions": [4]}, "unsupported"),
        ({"actions": [True]}, "integer"),
        ({"actions": [1, 1]}, "exactly one"),
    ]:
        try:
            service.step(payload)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"invalid payload was accepted: {payload}")


def test_action_budget_terminates_episode():
    service = MinimalVLNService(FakeAdapter(), action_budget=2)

    first = service.step({"actions": [1]})
    result = service.step({"actions": [1]})

    assert first == {"executed": 1, "remaining_budget": 1, "terminal": False}
    assert result == {"executed": 1, "remaining_budget": 0, "terminal": True}


def test_prompt_preserves_paper_action_and_budget_contract():
    prompt = build_prompt("Walk through the doorway.", budget=500)

    assert '"Walk through the doorway."' in prompt
    assert "0 = STOP" in prompt
    assert "1 = move forward 0.25 m" in prompt
    assert "2 = turn left 15 degrees" in prompt
    assert "3 = turn right 15 degrees" in prompt
    assert "budget of 500 movement actions" in prompt
    assert "within 3 meters" in prompt
    assert "execute exactly one primitive action" in prompt
    assert "Every response must contain exactly one tool call" in prompt
    assert "Do not write narration" in prompt


def test_interactive_prompt_waits_for_instruction_and_keeps_minimal_tools():
    prompt = build_interactive_prompt(budget=500)

    assert "Wait for the user to provide a navigation instruction" in prompt
    assert "Do not call a tool before receiving one" in prompt
    assert "observe()" in prompt
    assert "step(actions)" in prompt
    assert "exactly three RGB images in LEFT, FRONT, RIGHT order" in prompt
