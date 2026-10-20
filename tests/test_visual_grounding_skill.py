from __future__ import annotations

import asyncio

import pytest

from agentos.core.events import InMemoryEventBus
from agentos.schemas.enums import SkillStatus
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation
from agentos.schemas.perception import (
    VisualDetection,
    VisualGroundingConstraints,
    VisualGroundingProviderResponse,
)
from agentos.schemas.skill import SkillRequest
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.perception import VisualGroundingSkill
from agentos.skills.registry import SkillRegistry
from agentos.providers.grounding import QwenVisualGroundingProvider


def _observation(make_trace, *, observation_id: str = "obs-grounding") -> Observation:
    trace = make_trace(observation_id=observation_id, skill_run_id="grounding-run")
    return Observation(
        trace=trace,
        rgb_refs={
            "left": "mock://left.jpg",
            "front": "mock://front.jpg",
            "right": "mock://right.jpg",
        },
    )


def _request(make_trace, **inputs) -> SkillRequest:
    return SkillRequest(
        trace=make_trace(observation_id="obs-grounding", skill_run_id="grounding-run"),
        skill_name="VisualGroundingSkill",
        command="Locate the requested object.",
        inputs={"query": "bed", **inputs},
    )


def test_grounding_constraints_reject_task_reasoning_fields():
    with pytest.raises(ValueError, match="spatial_relation"):
        VisualGroundingConstraints.model_validate(
            {"spatial_relation": "The route continues on the right side."}
        )

    with pytest.raises(ValueError, match="cot_fields"):
        VisualGroundingConstraints.model_validate(
            {"cot_fields": {"path_planning": "Continue right."}}
        )


def _context() -> SkillContext:
    return SkillContext(event_bus=InMemoryEventBus(), cancellation=CancellationToken())


class _FakeGroundingProvider:
    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    async def ground(self, request):
        self.requests.append(request)
        return VisualGroundingProviderResponse(
            provider="fake-grounder",
            detections=[
                VisualDetection(
                    label="bed",
                    view="left",
                    bbox_normalized_1000=[100, 200, 500, 800],
                    confidence=0.96,
                    evidence_summary="A bed is visible in the left camera.",
                ),
                VisualDetection(
                    label="bed-like region",
                    view="front",
                    bbox_normalized_1000=[200, 300, 600, 700],
                    confidence=0.70,
                ),
            ],
        )

    async def close(self):
        self.closed = True


def test_skill_is_discoverable_without_a_configured_provider():
    registry = SkillRegistry()
    registry.register(VisualGroundingSkill())

    descriptor = registry.describe("VisualGroundingSkill")

    assert descriptor.risk_level.value == "read_only"
    assert descriptor.required_modalities == ["rgb"]
    single_schema = descriptor.input_schema["anyOf"][0]
    batch_schema = descriptor.input_schema["anyOf"][1]
    assert single_schema["properties"]["query"]["minLength"] == 1
    assert batch_schema["properties"]["queries"]["maxItems"] == 2


def test_unconfigured_provider_returns_a_structured_failure(make_trace):
    result = asyncio.run(
        VisualGroundingSkill().run(_request(make_trace), _observation(make_trace), _context())
    )

    assert result.status == SkillStatus.FAILURE
    assert result.outputs == {
        "available": False,
        "query": "bed",
        "observation_id": "obs-grounding",
        "provider": None,
        "found": False,
        "detections": [],
        "reason": "provider_not_configured",
    }
    assert result.stop_reason == "provider_not_configured"


def test_provider_results_are_filtered_and_compressed(make_trace):
    provider = _FakeGroundingProvider()
    skill = VisualGroundingSkill(lambda: provider)

    result = asyncio.run(skill.run(_request(make_trace), _observation(make_trace), _context()))

    assert result.status == SkillStatus.SUCCESS
    assert result.outputs["found"] is True
    assert len(result.outputs["detections"]) == 1
    assert result.outputs["detections"][0]["view"] == "left"
    assert result.evidence[0].data["bottom_center_normalized_1000"] == [300, 800]
    assert result.compressed_history == "Visual grounding query='bed'; found=True; detections=1."
    assert provider.requests[0].observation_id == "obs-grounding"
    assert provider.closed is True


def test_stale_observation_is_rejected_before_provider_call(make_trace):
    provider = _FakeGroundingProvider()
    request = _request(make_trace).model_copy(
        update={"trace": make_trace(observation_id="obs-old", skill_run_id="grounding-run")}
    )

    result = asyncio.run(
        VisualGroundingSkill(lambda: provider).run(request, _observation(make_trace), _context())
    )

    assert result.status == SkillStatus.STALE
    assert result.outputs["reason"] == "request_observation_mismatch"
    assert provider.requests == []


def test_detection_bbox_must_be_normalized_and_nonempty():
    with pytest.raises(ValueError, match="positive width"):
        VisualDetection(
            label="bed",
            view="front",
            bbox_normalized_1000=[400, 200, 400, 800],
            confidence=0.9,
        )


class _FakeQwen:
    def __init__(self) -> None:
        self.messages = []
        self.schemas = []
        self.closed = False

    async def complete(self, messages, *, response_schema=None):
        self.messages.append(messages)
        self.schemas.append(response_schema)
        image_label = messages[1].content[1]["text"]
        view = next(
            item
            for item in ("left", "front", "right")
            if image_label.endswith(item.upper())
        )
        return LLMResponse(
            model="qwen3.6-vl-35b",
            content="{}",
            raw={},
            parsed_json={
                "detections": [
                    {
                        "label": "bed",
                        "view": view,
                        "bbox_normalized_1000": [50, 420, 320, 690],
                        "confidence": 0.97,
                        "evidence_summary": f"Current {view} view contains the bed.",
                    }
                ]
            },
        )

    async def close(self):
        self.closed = True


def test_qwen_grounder_receives_only_labelled_current_views(make_trace, tmp_path):
    qwen = _FakeQwen()
    skill = VisualGroundingSkill(lambda: QwenVisualGroundingProvider(qwen))
    observation = _observation(make_trace)
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"{view}.jpg"
        path.write_bytes(view.encode("ascii"))
        refs[view] = str(path)
    observation = observation.model_copy(update={"rgb_refs": refs})

    request = _request(
        make_trace,
        constraints={
            "preferred_view": "left",
            "visual_description": "A large furniture object covered with orange fabric.",
            "excluded_labels": ["robot", "cabinet"],
        },
    )
    result = asyncio.run(skill.run(request, observation, _context()))

    assert result.outputs["observation_id"] == "obs-grounding"
    assert result.outputs["detections"][0]["bbox_normalized_1000"] == [50, 420, 320, 690]
    assert len(qwen.messages) == 1
    labels = [
        item.get("text")
        for item in qwen.messages[0][1].content
        if item.get("type") == "text"
    ]
    assert labels[1:] == ["CURRENT_OBSERVATION_obs-grounding_LEFT"]
    prompt = labels[0]
    assert '"visual_description":"A large furniture object covered with orange fabric."' in prompt
    assert '"excluded_labels":["robot","cabinet"]' in prompt
    assert "Inside the bedroom near the wall." not in prompt
    assert "target_localization_and_key_context" not in prompt
    assert "requires independent confirmation" in prompt
    assert result.outputs["detections"][0]["view"] == "left"
    assert all("HISTORICAL" not in label for label in labels)
    assert qwen.closed is True


class _FakeBatchQwen(_FakeQwen):
    async def complete(self, messages, *, response_schema=None):
        self.messages.append(messages)
        self.schemas.append(response_schema)
        if "detections" in (response_schema or {}).get("properties", {}):
            prompt = messages[1].content[0]["text"]
            image_label = messages[1].content[1]["text"]
            view = next(
                item
                for item in ("left", "front", "right")
                if image_label.endswith(item.upper())
            )
            if "doorway leading into the corridor" in prompt and view == "front":
                detection = {
                    "label": "doorway leading into the corridor",
                    "view": "front",
                    "bbox_normalized_1000": [390, 290, 610, 890],
                    "confidence": 0.95,
                    "evidence_summary": "Single-view refinement confirms the doorway.",
                }
            elif "bed" in prompt and view == "right":
                detection = {
                    "label": "bed",
                    "view": "right",
                    "bbox_normalized_1000": [90, 310, 350, 710],
                    "confidence": 0.98,
                    "evidence_summary": "Single-view refinement confirms the bed.",
                }
            else:
                detection = None
            return LLMResponse(
                model="qwen3.6-vl-35b",
                content="{}",
                raw={},
                parsed_json={"detections": [detection] if detection else []},
            )
        return LLMResponse(
            model="qwen3.6-vl-35b",
            content="{}",
            raw={},
            parsed_json={
                "results": [
                    {
                        "role": "affordance",
                        "query": "doorway leading into the corridor",
                        "detections": [
                            {
                                "label": "doorway leading into the corridor",
                                "view": "front",
                                "bbox_normalized_1000": [380, 280, 620, 900],
                                "confidence": 0.94,
                                "evidence_summary": "The front doorway supports the next transition.",
                            }
                        ],
                    },
                    {
                        "role": "target",
                        "query": "bed",
                        "detections": [
                            {
                                "label": "bed",
                                "view": "right",
                                "bbox_normalized_1000": [80, 300, 360, 700],
                                "confidence": 0.97,
                                "evidence_summary": "The current right view contains the bed.",
                            }
                        ],
                    },
                ]
            },
        )


def test_qwen_batch_grounding_localizes_each_query_one_camera_at_a_time(
    make_trace,
    tmp_path,
):
    qwen = _FakeBatchQwen()
    observation = _observation(make_trace)
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"batch-{view}.jpg"
        path.write_bytes(view.encode("ascii"))
        refs[view] = str(path)
    observation = observation.model_copy(update={"rgb_refs": refs})
    request = SkillRequest(
        trace=make_trace(observation_id="obs-grounding", skill_run_id="grounding-run"),
        skill_name="VisualGroundingSkill",
        command="Ground target and affordance.",
        inputs={
            "queries": [
                {
                    "role": "affordance",
                    "query": "doorway leading into the corridor",
                    "constraints": {},
                    "min_confidence": 0.8,
                },
                {
                    "role": "target",
                    "query": "bed",
                    "constraints": {
                        "visual_description": "A bed inside the bedroom.",
                    },
                    "min_confidence": 0.9,
                },
            ],
            "candidate_views": ["left", "front", "right"],
            "max_results_per_query": 1,
        },
    )

    result = asyncio.run(
        VisualGroundingSkill(lambda: QwenVisualGroundingProvider(qwen)).run(
            request,
            observation,
            _context(),
        )
    )

    assert len(qwen.messages) == 6
    assert [item["role"] for item in result.outputs["results"]] == [
        "affordance",
        "target",
    ]
    assert result.outputs["results"][0]["detections"][0]["view"] == "front"
    assert result.outputs["results"][0]["detections"][0]["bbox_normalized_1000"] == [
        390,
        290,
        610,
        890,
    ]
    assert result.outputs["results"][1]["detections"][0]["view"] == "right"
    assert result.outputs["results"][1]["detections"][0]["bbox_normalized_1000"] == [
        90,
        310,
        350,
        710,
    ]
    labels = [
        [
            item.get("text")
            for item in messages[1].content
            if item.get("type") == "text"
        ]
        for messages in qwen.messages
    ]
    assert [items[1] for items in labels] == [
        "CURRENT_OBSERVATION_obs-grounding_LEFT",
        "CURRENT_OBSERVATION_obs-grounding_FRONT",
        "CURRENT_OBSERVATION_obs-grounding_RIGHT",
        "CURRENT_OBSERVATION_obs-grounding_LEFT",
        "CURRENT_OBSERVATION_obs-grounding_FRONT",
        "CURRENT_OBSERVATION_obs-grounding_RIGHT",
    ]
    assert all("Beyond the doorway." not in items[0] for items in labels)
    assert qwen.closed is True


class _VerboseLabelBatchProvider:
    async def ground_batch(self, request):
        from agentos.schemas.perception import (
            VisualGroundingBatchProviderResponse,
            VisualGroundingQueryResult,
        )

        query = request.queries[0]
        return VisualGroundingBatchProviderResponse(
            provider="verbose-label-grounder",
            results=[
                VisualGroundingQueryResult(
                    role=query.role,
                    query=query.query,
                    detections=[
                        VisualDetection(
                            label=(
                                "A long red carpet runner extending through the corridor with "
                                "white cabinets and several objects visible on the right side."
                            ),
                            view="front",
                            bbox_normalized_1000=[300, 400, 700, 900],
                            confidence=0.95,
                        )
                    ],
                )
            ],
        )

    async def close(self):
        return None


def test_batch_grounding_keeps_the_slow_query_as_the_detection_label(make_trace):
    request = SkillRequest(
        trace=make_trace(observation_id="obs-grounding", skill_run_id="grounding-run"),
        skill_name="VisualGroundingSkill",
        command="Ground one affordance.",
        inputs={
            "queries": [
                {
                    "role": "affordance",
                    "query": "red carpet corridor entrance",
                    "constraints": {
                        "preferred_view": "front",
                        "visual_description": "Red carpet bounded by white floor lines.",
                    },
                    "min_confidence": 0.8,
                }
            ],
            "candidate_views": ["left", "front", "right"],
            "max_results_per_query": 1,
        },
    )

    result = asyncio.run(
        VisualGroundingSkill(lambda: _VerboseLabelBatchProvider()).run(
            request,
            _observation(make_trace),
            _context(),
        )
    )

    detection = result.outputs["results"][0]["detections"][0]
    assert detection["label"] == "red carpet corridor entrance"
    assert "right side" not in detection["label"]


def test_qwen_batch_grounding_uses_only_slow_selected_single_views(
    make_trace,
    tmp_path,
):
    qwen = _FakeBatchQwen()
    observation = _observation(make_trace, observation_id="obs-selected")
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"selected-{view}.jpg"
        path.write_bytes(view.encode("ascii"))
        refs[view] = str(path)
    observation = observation.model_copy(update={"rgb_refs": refs})
    request = SkillRequest(
        trace=make_trace(observation_id="obs-selected", skill_run_id="grounding-run"),
        skill_name="VisualGroundingSkill",
        command="Ground Slow-selected views.",
        inputs={
            "queries": [
                {
                    "role": "affordance",
                    "query": "doorway leading into the corridor",
                    "constraints": {
                        "preferred_view": "front",
                        "visual_description": "A doorway frame in the front image.",
                    },
                    "min_confidence": 0.8,
                },
                {
                    "role": "target",
                    "query": "bed",
                    "constraints": {
                        "preferred_view": "right",
                        "visual_description": "A bed in the right image.",
                    },
                    "min_confidence": 0.9,
                },
            ],
            "candidate_views": ["left", "front", "right"],
            "max_results_per_query": 1,
        },
    )

    result = asyncio.run(
        VisualGroundingSkill(lambda: QwenVisualGroundingProvider(qwen)).run(
            request,
            observation,
            _context(),
        )
    )

    assert len(qwen.messages) == 2
    labels = [
        [item.get("text") for item in messages[1].content if item.get("type") == "text"]
        for messages in qwen.messages
    ]
    assert [items[1] for items in labels] == [
        "CURRENT_OBSERVATION_obs-selected_FRONT",
        "CURRENT_OBSERVATION_obs-selected_RIGHT",
    ]
    assert all("advances the global bedroom task" not in items[0] for items in labels)
    assert all("final object expected in the bedroom" not in items[0] for items in labels)
    assert result.outputs["results"][0]["detections"][0]["view"] == "front"
    assert result.outputs["results"][1]["detections"][0]["view"] == "right"
    assert qwen.closed is True
