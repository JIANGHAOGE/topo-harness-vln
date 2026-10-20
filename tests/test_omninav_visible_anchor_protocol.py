from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from agentos.planning import OmniFinishVerification, OmniNavSupervisor, OmniPostDecision, OmniTaskPlan
from agentos.schemas.llm import LLMResponse


def test_initial_omninav_command_requires_visible_anchor_form():
    with pytest.raises(ValidationError, match="Navigate towards"):
        OmniTaskPlan(
            decision="run_omninav",
            task_summary="Find a bedroom.",
            task_success_conditions=["A bed is clearly visible."],
            subtasks=["Search for the bedroom"],
            initial_visible_anchor="the open doorway",
            initial_anchor_view="front",
            initial_subtask="Search for the bedroom",
            initial_subtask_success_conditions=["The doorway is reached."],
            initial_scene_description="An open doorway is visible ahead.",
            rationale_summary="The doorway is a currently visible waypoint.",
        )


def test_post_agent_continue_requires_camera_grounded_anchor():
    with pytest.raises(ValidationError, match="next_visible_anchor"):
        OmniPostDecision(
            previous_subtask_status="success",
            evidence_summary="The prior doorway was reached.",
            current_scene_description="A corridor is visible ahead.",
            omni_action_description="OmniNav approached the doorway.",
            trajectory_descriptions=[
                {"frame_index": 1, "observation_id": "obs-1", "description": "At the doorway."}
            ],
            decision="next_subtask",
            next_subtask="Navigate towards the corridor opening",
            next_subtask_success_conditions=["The corridor opening is reached."],
            rationale_summary="Continue through visible waypoints.",
        )


def test_finish_rejection_returns_visible_anchor_navigation_command():
    result = OmniFinishVerification(
        verdict="continue",
        evidence_view="none",
        evidence_summary="No bed is currently visible.",
        next_visible_anchor="the open doorway beside the cabinet",
        next_anchor_view="right",
        next_subtask="Navigate towards the open doorway beside the cabinet",
        next_subtask_success_conditions=["The doorway fills the front view."],
    )

    assert result.next_anchor_view == "right"
    assert result.next_subtask.startswith("Navigate towards ")


def test_schema_repair_reuses_grounding_without_resending_images():
    incomplete = {
        "decision": "run_omninav",
        "task_summary": "Find a bedroom.",
        "task_success_conditions": ["A bed is visible."],
        "subtasks": ["Navigate towards the corridor opening"],
        "initial_visible_anchor": "the corridor opening",
        "initial_anchor_view": "front",
        "initial_subtask": "Navigate towards the corridor opening",
    }
    repaired = {
        **incomplete,
        "initial_subtask_success_conditions": ["The opening is approached and a new area is visible."],
        "initial_scene_description": "A corridor opening is visible in front.",
        "rationale_summary": "The opening reveals unexplored space.",
        "anchor_explanation": "This undeclared field must be discarded.",
    }

    class RepairingProvider:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, *, response_schema=None):
            self.calls.append(messages)
            parsed = incomplete if len(self.calls) == 1 else repaired
            return LLMResponse(model="fake", content="{}", raw={}, parsed_json=parsed)

    provider = RepairingProvider()
    result = asyncio.run(
        OmniNavSupervisor(provider)._complete_content(
            OmniTaskPlan,
            "system",
            [
                {"type": "text", "text": "ground this scene"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
            ],
        )
    )

    assert result.initial_visible_anchor == "the corridor opening"
    assert len(provider.calls) == 2
    assert isinstance(provider.calls[1][-1].content, str)
    assert "image_url" not in provider.calls[1][-1].content
