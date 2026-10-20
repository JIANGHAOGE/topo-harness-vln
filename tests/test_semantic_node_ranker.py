from __future__ import annotations

import asyncio
import json

from agentos.planning.semantic_nodes import SemanticNodeRanker
from agentos.schemas.llm import LLMResponse
from agentos.schemas.task_reasoning import SemanticTaskDecompositionInput


class FakeSemanticProvider:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    async def complete(self, messages, *, response_schema=None):
        self.calls.append({"messages": messages, "schema": response_schema})
        return LLMResponse(
            model="fake-semantic-model",
            content="{}",
            raw={},
            parsed_json={
                "selected_topology_node_id": "backbone-002",
                "confidence": 0.96,
                "matched_concepts": ["fridge", "refrigerator"],
                "selection_reason": "The node contains explicit refrigerator evidence.",
            },
        )

    async def close(self):
        self.calls.append({"closed": True})


def test_semantic_node_ranker_resolves_paraphrase_without_a_synonym_table():
    calls: list[dict[str, object]] = []
    ranker = SemanticNodeRanker(lambda: FakeSemanticProvider(calls))

    selection = ranker.select(
        SemanticTaskDecompositionInput(
            global_goal="go to the office to find a fridge",
            destination_query="office",
            object_goal="fridge",
        ),
        [
            {
                "topology_node_id": "backbone-001",
                "node_semantics": {
                    "place_type": "office area",
                    "objects": ["white cabinets"],
                },
            },
            {
                "topology_node_id": "backbone-002",
                "node_semantics": {
                    "place_type": "open hall",
                    "objects": ["refrigerator"],
                    "landmarks": ["tall white refrigerator"],
                },
            },
        ],
    )

    assert selection.selected_topology_node_id == "backbone-002"
    assert selection.matched_concepts == ["fridge", "refrigerator"]
    prompt = json.loads(calls[0]["messages"][1].content)
    assert prompt["object_goal"] == "fridge"
    assert prompt["target_relation"] == {
        "subject": "fridge",
        "predicate": "located_in_or_associated_with",
        "object": "office",
    }
    assert prompt["nodes"][1]["node_semantics"]["objects"] == ["refrigerator"]
    assert "first form the candidate set from place_type" in calls[0]["messages"][0].content
    assert calls[-1] == {"closed": True}


def test_semantic_node_ranker_keeps_object_bound_to_requested_place():
    calls: list[dict[str, object]] = []
    ranker = SemanticNodeRanker(lambda: FakeSemanticProvider(calls))

    selection = ranker.select(
        SemanticTaskDecompositionInput(
            global_goal="find the robot in the kitchen",
            destination_query="kitchen",
            object_goal="robot",
        ),
        [
            {
                "topology_node_id": "backbone-001",
                "node_semantics": {
                    "place_type": "office",
                    "objects": ["white humanoid robot"],
                },
            },
            {
                "topology_node_id": "backbone-002",
                "node_semantics": {
                    "place_type": "kitchen",
                    "function": "food preparation",
                    "objects": ["white humanoid robot", "refrigerator"],
                },
            },
        ],
    )

    assert selection.selected_topology_node_id == "backbone-002"
    prompt = json.loads(calls[0]["messages"][1].content)
    assert prompt["target_relation"]["subject"] == "robot"
    assert prompt["target_relation"]["object"] == "kitchen"
    assert "one relation" in calls[0]["messages"][0].content


class _JointRegionProvider:
    calls = 0

    async def complete(self, messages, *, response_schema=None):
        type(self).calls += 1
        assert "one joint semantic ranking" in messages[0].content
        if response_schema["title"] == "RegionSearchPlan":
            payload = {
                "candidates": [
                    {
                        "topology_node_id": "backbone-001",
                        "confidence": 0.82,
                        "place_evidence": "Office area.",
                        "object_evidence": "No refrigerator prior.",
                        "relation_reason": "Exact destination place.",
                    },
                    {
                        "topology_node_id": "backbone-002",
                        "confidence": 0.70,
                        "place_evidence": "Open technical workspace conflicts with office.",
                        "object_evidence": "A refrigerator is mapped here.",
                        "relation_reason": "Strong object prior but weaker place relation.",
                    },
                ],
                "active_topology_node_id": "backbone-001",
                "update_reason": "The unsearched office remains the best joint hypothesis.",
            }
        else:
            raise AssertionError(response_schema["title"])
        return LLMResponse(
            model="fake-semantic-model",
            content="{}",
            raw={},
            parsed_json=payload,
        )

    async def close(self):
        return None


def test_region_ranker_keeps_destination_place_ahead_of_object_only_node():
    _JointRegionProvider.calls = 0
    ranker = SemanticNodeRanker(_JointRegionProvider)
    request = SemanticTaskDecompositionInput(
        global_goal="go to office to find a white refrigerator",
        destination_query="office",
        object_goal="white refrigerator",
    )
    nodes = [
        {
            "topology_node_id": "backbone-001",
            "node_semantics": {
                "place_type": "office area",
                "scene_summary": "Office circulation area.",
                "objects": ["white cabinets"],
            },
        },
        {
            "topology_node_id": "backbone-002",
            "node_semantics": {
                "place_type": "open hall",
                "scene_summary": "Technical workspace.",
                "objects": ["refrigerator"],
            },
        },
    ]

    ranking = asyncio.run(ranker.rank_regions_async(request, nodes))

    assert [item.topology_node_id for item in ranking.candidates] == [
        "backbone-001",
        "backbone-002",
    ]
    assert ranking.active_topology_node_id == "backbone-001"
    assert _JointRegionProvider.calls == 1


class _JointTieRegionProvider:
    calls = 0

    async def complete(self, messages, *, response_schema=None):
        type(self).calls += 1
        assert "break ties among compatible places" in messages[0].content
        if response_schema["title"] == "RegionSearchPlan":
            payload = {
                "candidates": [
                    {
                        "topology_node_id": "backbone-002",
                        "confidence": 0.95,
                        "place_evidence": "Exact office place type.",
                        "object_evidence": "Direct refrigerator evidence.",
                        "relation_reason": "Best joint place and object relation.",
                    },
                    {
                        "topology_node_id": "backbone-000",
                        "confidence": 0.85,
                        "place_evidence": "Exact office place type.",
                        "object_evidence": "No direct refrigerator evidence.",
                        "relation_reason": "Compatible place with weaker object evidence.",
                    },
                    {
                        "topology_node_id": "backbone-001",
                        "confidence": 0.75,
                        "place_evidence": "Exact office place type.",
                        "object_evidence": "No direct refrigerator evidence.",
                        "relation_reason": "Compatible place with weaker object evidence.",
                    },
                ],
                "active_topology_node_id": "backbone-002",
                "update_reason": "Joint evidence ranks backbone-002 first.",
            }
        else:
            raise AssertionError(response_schema["title"])
        return LLMResponse(
            model="fake-semantic-model",
            content="{}",
            raw={},
            parsed_json=payload,
        )

    async def close(self):
        return None


def test_region_ranker_uses_object_evidence_to_break_destination_place_ties():
    _JointTieRegionProvider.calls = 0
    ranker = SemanticNodeRanker(_JointTieRegionProvider)
    request = SemanticTaskDecompositionInput(
        global_goal="go to office to find a white refrigerator",
        destination_query="office",
        object_goal="white refrigerator",
    )
    nodes = [
        {
            "topology_node_id": node_id,
            "node_semantics": {"place_type": "office"},
        }
        for node_id in ("backbone-000", "backbone-001", "backbone-002")
    ]

    ranking = asyncio.run(ranker.rank_regions_async(request, nodes))

    assert [item.topology_node_id for item in ranking.candidates] == [
        "backbone-002",
        "backbone-000",
        "backbone-001",
    ]
    assert ranking.active_topology_node_id == "backbone-002"
    assert _JointTieRegionProvider.calls == 1


class _TaskDecompositionProvider:
    def __init__(self, calls):
        self.calls = calls

    async def complete(self, messages, *, response_schema=None):
        self.calls.append((messages, response_schema))
        return LLMResponse(
            model="fake-semantic-model",
            content="{}",
            raw={},
            parsed_json={
                "global_goal": "go to office to find a white refrigerator",
                "destination_query": "office",
                "object_goal": "white refrigerator",
                "node_local_radius_m": 5.0,
                "anchor_max_depth_m": 5.0,
            },
        )

    async def close(self):
        return None


def test_task_semantic_audit_preserves_destination_object_boundary_and_caches():
    calls = []
    ranker = SemanticNodeRanker(lambda: _TaskDecompositionProvider(calls))

    first = asyncio.run(
        ranker.decompose_task_async("go to office to find a white refrigerator")
    )
    second = asyncio.run(
        ranker.decompose_task_async("go to office to find a white refrigerator")
    )

    assert first.destination_query == "office"
    assert first.object_goal == "white refrigerator"
    assert second == first
    assert len(calls) == 1
    assert "without selecting a map node" in calls[0][0][0].content
