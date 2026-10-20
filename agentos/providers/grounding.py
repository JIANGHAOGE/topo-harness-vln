from __future__ import annotations

import asyncio
import json
from typing import Protocol

from pydantic import Field

from agentos.providers.base import LLMProvider
from agentos.providers.vision import image_ref_content
from agentos.schemas.base import SchemaModel
from agentos.schemas.llm import ChatMessage
from agentos.schemas.perception import (
    VisualDetection,
    VisualGroundingConstraints,
    VisualGroundingBatchProviderRequest,
    VisualGroundingBatchProviderResponse,
    VisualGroundingProviderRequest,
    VisualGroundingProviderResponse,
    VisualGroundingQueryResult,
)


class VisualGroundingProvider(Protocol):
    """Adapter contract for a local or remote open-vocabulary detector."""

    async def ground(
        self,
        request: VisualGroundingProviderRequest,
    ) -> VisualGroundingProviderResponse:
        ...

    async def ground_batch(
        self,
        request: VisualGroundingBatchProviderRequest,
    ) -> VisualGroundingBatchProviderResponse:
        ...

    async def close(self) -> None:
        ...


class _QwenGroundingResponse(SchemaModel):
    detections: list[VisualDetection] = Field(default_factory=list)


class QwenVisualGroundingProvider:
    """Current-observation-only Qwen3-VL adapter for VisualGroundingSkill."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def ground(
        self,
        request: VisualGroundingProviderRequest,
    ) -> VisualGroundingProviderResponse:
        if len(request.candidate_views) > 1:
            preferred = request.constraints.preferred_view
            views = [preferred] if preferred is not None else list(request.candidate_views)

            async def ground_view(view: str) -> VisualGroundingProviderResponse:
                return await self.ground(
                    VisualGroundingProviderRequest(
                        observation_id=request.observation_id,
                        query=request.query,
                        image_refs={view: request.image_refs[view]},
                        candidate_views=[view],
                        constraints=VisualGroundingConstraints(
                            visual_description=request.constraints.visual_description,
                            excluded_labels=request.constraints.excluded_labels,
                        ),
                        max_results=request.max_results,
                    )
                )

            responses = await asyncio.gather(*(ground_view(view) for view in views))
            detections = sorted(
                (
                    detection
                    for view, response in zip(views, responses)
                    for detection in response.detections
                    if detection.view == view
                ),
                key=lambda detection: detection.confidence,
                reverse=True,
            )[: request.max_results]
            return VisualGroundingProviderResponse(
                provider=responses[0].provider,
                detections=detections,
            )

        views = ", ".join(view.upper() for view in request.candidate_views)
        constraints = json.dumps(
            {
                "preferred_view": request.constraints.preferred_view,
                "visual_description": request.constraints.visual_description,
                "excluded_labels": request.constraints.excluded_labels,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = (
            f"Locate {request.query!r} only in synchronized CURRENT observation "
            f"{request.observation_id}. Search these camera views: {views}. "
            f"The current Slow-System semantic hypothesis is this structured object: {constraints}. "
            "Use it to disambiguate current visual appearance and distractors, but treat every field "
            "as a hypothesis that requires independent confirmation in the supplied current images. "
            "preferred_view is a search hint, not proof; inspect every candidate view. "
            "Historical images and previous coordinates are not provided and must not be inferred. "
            "For every clearly visible matching object, return its camera view and a tight bbox "
            "[xmin,ymin,xmax,ymax] normalized from 0 to 1000 within that single named image. "
            "Do not concatenate camera coordinates. Return no detections when the requested object is "
            "not visually supported. Confidence describes localization certainty, not task relevance. "
            f"Return at most {request.max_results} detections."
        )
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for view in request.candidate_views:
            content.extend(
                image_ref_content(
                    f"CURRENT_OBSERVATION_{request.observation_id}_{view.upper()}",
                    request.image_refs[view],
                )
            )
        response = await self._provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are the VisualGroundingSkill perception provider. Ground the requested "
                        "semantic object only in the explicitly labelled current camera images. "
                        "Never copy or extrapolate a bbox from textual reasoning. Return strict JSON."
                    ),
                ),
                ChatMessage(role="user", content=content),
            ],
            response_schema=_QwenGroundingResponse.model_json_schema(),
        )
        if not isinstance(response.parsed_json, dict):
            raise ValueError("Qwen visual grounding did not return one JSON object")
        parsed = _QwenGroundingResponse.model_validate(response.parsed_json)
        return VisualGroundingProviderResponse(
            provider=response.model,
            detections=parsed.detections,
        )

    async def ground_batch(
        self,
        request: VisualGroundingBatchProviderRequest,
    ) -> VisualGroundingBatchProviderResponse:
        return await self._ground_queries_per_view(request)

    async def _ground_queries_per_view(
        self,
        request: VisualGroundingBatchProviderRequest,
    ) -> VisualGroundingBatchProviderResponse:
        """Localize every query one physical camera at a time."""

        async def ground_query(
            query: VisualGroundingQuery,
        ) -> tuple[str, VisualGroundingQueryResult]:
            preferred = query.constraints.preferred_view
            views = [preferred] if preferred is not None else list(request.candidate_views)

            async def ground_view(view: str) -> VisualGroundingProviderResponse:
                return await self.ground(
                    VisualGroundingProviderRequest(
                        observation_id=request.observation_id,
                        query=query.query,
                        image_refs={view: request.image_refs[view]},
                        candidate_views=[view],
                        constraints=VisualGroundingConstraints(
                            visual_description=query.constraints.visual_description,
                            excluded_labels=query.constraints.excluded_labels,
                        ),
                        max_results=request.max_results_per_query,
                    )
                )

            responses = await asyncio.gather(*(ground_view(view) for view in views))
            detections = sorted(
                (
                    detection
                    for view, response in zip(views, responses)
                    for detection in response.detections
                    if detection.view == view
                ),
                key=lambda detection: detection.confidence,
                reverse=True,
            )[: request.max_results_per_query]
            return responses[0].provider, VisualGroundingQueryResult(
                role=query.role,
                query=query.query,
                detections=detections,
            )

        grounded = await asyncio.gather(*(ground_query(query) for query in request.queries))
        return VisualGroundingBatchProviderResponse(
            provider=grounded[0][0],
            results=[result for _, result in grounded],
        )

    async def close(self) -> None:
        await self._provider.close()
