from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from agentos.providers.base import LLMProvider
from agentos.schemas.llm import ChatMessage
from agentos.schemas.task_reasoning import (
    RegionSearchPlan,
    SemanticNodeSelection,
    SemanticTaskDecompositionInput,
)


class SemanticNodeRanker:
    """Use one bounded text-model call to map a task onto fixed semantic nodes."""

    def __init__(self, provider_factory: Callable[[], LLMProvider]) -> None:
        self._provider_factory = provider_factory
        self._decomposition_cache: dict[str, SemanticTaskDecompositionInput] = {}

    async def decompose_task_async(
        self,
        global_goal: str,
    ) -> SemanticTaskDecompositionInput:
        """Audit the place/object boundary before any node receives a score."""
        cache_key = " ".join(global_goal.casefold().split())
        cached = self._decomposition_cache.get(cache_key)
        if cached is not None:
            return cached
        provider = self._provider_factory()
        try:
            response = await provider.complete(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Decompose one navigation instruction without selecting a map node. "
                            "Preserve the complete original instruction in global_goal. When the "
                            "instruction names a place and a separate object to find there, put the "
                            "place in destination_query and the physical object in object_goal. Do "
                            "not replace the place with a node's likely contents. When there is no "
                            "separate destination place, set destination_query equal to object_goal. "
                            "Return strict SemanticTaskDecompositionInput JSON."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {"navigation_instruction": global_goal},
                            ensure_ascii=False,
                        ),
                    ),
                ],
                response_schema=SemanticTaskDecompositionInput.model_json_schema(),
            )
        finally:
            await provider.close()
        decomposition = SemanticTaskDecompositionInput.model_validate(
            response.parsed_json
        )
        self._decomposition_cache[cache_key] = decomposition
        return decomposition

    def select(
        self,
        request: SemanticTaskDecompositionInput,
        nodes: list[dict[str, object]],
    ) -> SemanticNodeSelection:
        if not nodes:
            raise ValueError("semantic node ranking requires at least one node")
        return asyncio.run(self.select_async(request, nodes))

    async def select_async(
        self,
        request: SemanticTaskDecompositionInput,
        nodes: list[dict[str, object]],
    ) -> SemanticNodeSelection:
        provider = self._provider_factory()
        try:
            response = await provider.complete(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Select one fixed topology node for the navigation task. Use only the "
                            "provided node semantics. Interpret synonyms, paraphrases, spelling "
                            "variation, and multilingual wording by meaning rather than literal token "
                            "overlap. Treat destination_query and object_goal as one relation: the target "
                            "object is located in or functionally associated with the requested place. "
                            "When destination_query names an explicit place, first form the candidate set from "
                            "place_type. Use scene summary, aliases, and function only as secondary place evidence, "
                            "then use objects, landmarks, and affordances to rank nodes inside that compatible set. "
                            "An object or alias at a conflicting place_type must not override an explicit place_type "
                            "match. Only widen to another place after goal-conditioned search evidence shows the "
                            "matching place was searched without the target. Treat no_target_observed as negative "
                            "evidence for this task, target_observed/target_confirmed as positive evidence, and blocked "
                            "only as route failure rather than semantic absence. Never invent evidence or a node id. "
                            "Return only the structured selection."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "global_goal": request.global_goal,
                                "destination_query": request.destination_query,
                                "object_goal": request.object_goal,
                                "target_relation": {
                                    "subject": request.object_goal,
                                    "predicate": "located_in_or_associated_with",
                                    "object": request.destination_query,
                                },
                                "nodes": nodes,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                response_schema=SemanticNodeSelection.model_json_schema(),
            )
        finally:
            await provider.close()
        selection = SemanticNodeSelection.model_validate(response.parsed_json)
        allowed_ids = {str(node["topology_node_id"]) for node in nodes}
        if selection.selected_topology_node_id not in allowed_ids:
            raise ValueError(
                f"semantic ranker selected unknown node {selection.selected_topology_node_id}"
            )
        return selection

    async def rank_regions_async(
        self,
        request: SemanticTaskDecompositionInput,
        nodes: list[dict[str, object]],
    ) -> RegionSearchPlan:
        if not nodes:
            raise ValueError("semantic region ranking requires at least one node")
        provider = self._provider_factory()
        try:
            response = await provider.complete(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Perform one joint semantic ranking of one to four fixed topology regions "
                            "for a goal-conditioned object search. Reason about destination place, target "
                            "object, their spatial or functional relation, and current search evidence in "
                            "the same decision. Do not run or imitate a separate place-only winner stage. "
                            "When the instruction names an explicit destination, its place semantics are "
                            "the primary relation constraint. Use object, landmark, function, and affordance "
                            "evidence to break ties among compatible places. A target-rich but conflicting "
                            "place must not outrank an unsearched compatible place; widen only after current-task "
                            "no_target_observed evidence weakens those compatible hypotheses. Promote "
                            "target_observed or target_confirmed evidence, and treat blocked only as route "
                            "failure. Confidence is your joint semantic relevance estimate, not a separately "
                            "computed place score. Order candidates from best to worst and set the first as "
                            "active. Interpret synonyms and multilingual wording by meaning. Use only provided "
                            "node IDs and evidence. Return one strict RegionSearchPlan JSON object whose top-level "
                            "fields are candidates, active_topology_node_id, and update_reason. Never return a "
                            "bare candidate array."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "global_goal": request.global_goal,
                                "destination_query": request.destination_query,
                                "object_goal": request.object_goal,
                                "target_relation": {
                                    "subject": request.object_goal,
                                    "predicate": "located_in_or_associated_with",
                                    "object": request.destination_query,
                                },
                                "nodes": nodes,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                response_schema=RegionSearchPlan.model_json_schema(),
            )
        finally:
            await provider.close()
        ranking = RegionSearchPlan.model_validate(response.parsed_json)
        allowed_ids = {str(node["topology_node_id"]) for node in nodes}
        unknown = sorted(
            {
                item.topology_node_id
                for item in ranking.candidates
                if item.topology_node_id not in allowed_ids
            }
        )
        if unknown:
            raise ValueError(f"semantic region ranker selected unknown nodes: {unknown}")
        return ranking
