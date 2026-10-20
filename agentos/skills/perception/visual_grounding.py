from __future__ import annotations

from collections.abc import Callable

from agentos.providers.grounding import VisualGroundingProvider
from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.schemas.enums import SkillStatus
from agentos.schemas.observation import Observation
from agentos.schemas.perception import (
    VisualGroundingBatchInput,
    VisualGroundingBatchOutput,
    VisualGroundingBatchProviderRequest,
    VisualGroundingInput,
    VisualGroundingOutput,
    VisualGroundingProviderRequest,
    VisualGroundingQueryResult,
)
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.skills.base import SkillContext


class VisualGroundingSkill:
    """Ground target and affordance semantics in synchronized camera observations."""

    name = "VisualGroundingSkill"
    descriptor = SkillDescriptor(
        name=name,
        description=(
            "Use a pluggable perception provider to locate target objects and navigable "
            "semantic affordances in the current synchronized camera observation."
        ),
        input_schema={
            "anyOf": [
                VisualGroundingInput.model_json_schema(),
                VisualGroundingBatchInput.model_json_schema(),
            ]
        },
        output_schema={
            "anyOf": [
                VisualGroundingOutput.model_json_schema(),
                VisualGroundingBatchOutput.model_json_schema(),
            ]
        },
        required_modalities=["rgb"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
        default_timeout_s=10.0,
    )

    def __init__(
        self,
        provider_factory: Callable[[], VisualGroundingProvider] | None = None,
    ) -> None:
        self._provider_factory = provider_factory

    async def run(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
    ) -> SkillResult:
        if "queries" in request.inputs:
            return await self._run_batch(request, observation, context)
        inputs = VisualGroundingInput.model_validate(request.inputs)
        observation_id = observation.trace.observation_id

        if request.trace.observation_id != observation_id:
            return self._result(
                request,
                VisualGroundingOutput(
                    available=False,
                    query=inputs.query,
                    observation_id=observation_id,
                    reason="request_observation_mismatch",
                ),
                status=SkillStatus.STALE,
                summary="Visual grounding request was discarded because its observation is stale.",
            )

        missing_views = [view for view in inputs.candidate_views if view not in observation.rgb_refs]
        if missing_views:
            return self._result(
                request,
                VisualGroundingOutput(
                    available=False,
                    query=inputs.query,
                    observation_id=observation_id,
                    reason=f"missing_rgb_views:{','.join(missing_views)}",
                ),
                status=SkillStatus.FAILURE,
                summary=f"Visual grounding requires unavailable RGB views: {missing_views}.",
            )

        if self._provider_factory is None:
            return self._result(
                request,
                VisualGroundingOutput(
                    available=False,
                    query=inputs.query,
                    observation_id=observation_id,
                    reason="provider_not_configured",
                ),
                status=SkillStatus.FAILURE,
                summary="Visual grounding provider is not configured.",
            )

        context.cancellation.raise_if_cancelled()
        await context.emit_progress(
            request,
            "Running visual grounding on the current observation.",
            observation_id=observation_id,
            query=inputs.query,
            candidate_views=inputs.candidate_views,
        )
        provider_request = VisualGroundingProviderRequest(
            observation_id=observation_id,
            query=inputs.query,
            image_refs={view: observation.rgb_refs[view] for view in inputs.candidate_views},
            candidate_views=inputs.candidate_views,
            constraints=inputs.constraints,
            max_results=inputs.max_results,
        )
        provider = self._provider_factory()
        try:
            response = await provider.ground(provider_request)
        finally:
            await provider.close()

        allowed_views = set(inputs.candidate_views)
        detections = [
            detection
            for detection in response.detections
            if detection.view in allowed_views and detection.confidence >= inputs.min_confidence
        ]
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        detections = detections[: inputs.max_results]
        output = VisualGroundingOutput(
            available=True,
            query=inputs.query,
            observation_id=observation_id,
            provider=response.provider,
            found=bool(detections),
            detections=detections,
            reason=None if detections else "no_detection_above_threshold",
        )
        evidence = [
            Evidence(
                kind="visual_grounding",
                summary=detection.evidence_summary or f"Detected {detection.label} in {detection.view}.",
                ref=observation.rgb_refs[detection.view],
                data={
                    "view": detection.view,
                    "bbox_normalized_1000": detection.bbox_normalized_1000,
                    "bottom_center_normalized_1000": detection.bottom_center(),
                    "confidence": detection.confidence,
                },
            )
            for detection in detections
        ]
        return self._result(
            request,
            output,
            status=SkillStatus.SUCCESS,
            summary=(
                f"Grounded {inputs.query!r} with {len(detections)} accepted detection(s)."
                if detections
                else f"No {inputs.query!r} detection met confidence {inputs.min_confidence:.2f}."
            ),
            evidence=evidence,
        )

    async def _run_batch(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
    ) -> SkillResult:
        inputs = VisualGroundingBatchInput.model_validate(request.inputs)
        observation_id = observation.trace.observation_id
        if request.trace.observation_id != observation_id:
            return self._batch_result(
                request,
                VisualGroundingBatchOutput(
                    available=False,
                    observation_id=observation_id,
                    reason="request_observation_mismatch",
                ),
                status=SkillStatus.STALE,
                summary="Batch visual grounding was discarded because its observation is stale.",
            )
        missing_views = [
            view for view in inputs.candidate_views if view not in observation.rgb_refs
        ]
        if missing_views:
            return self._batch_result(
                request,
                VisualGroundingBatchOutput(
                    available=False,
                    observation_id=observation_id,
                    reason=f"missing_rgb_views:{','.join(missing_views)}",
                ),
                status=SkillStatus.FAILURE,
                summary=f"Batch visual grounding requires unavailable RGB views: {missing_views}.",
            )
        if self._provider_factory is None:
            return self._batch_result(
                request,
                VisualGroundingBatchOutput(
                    available=False,
                    observation_id=observation_id,
                    reason="provider_not_configured",
                ),
                status=SkillStatus.FAILURE,
                summary="Batch visual grounding provider is not configured.",
            )

        context.cancellation.raise_if_cancelled()
        await context.emit_progress(
            request,
            "Grounding target and affordance queries in one current-observation call.",
            observation_id=observation_id,
            queries=[query.model_dump(mode="json") for query in inputs.queries],
            candidate_views=inputs.candidate_views,
        )
        provider = self._provider_factory()
        try:
            response = await provider.ground_batch(
                VisualGroundingBatchProviderRequest(
                    observation_id=observation_id,
                    queries=inputs.queries,
                    image_refs={
                        view: observation.rgb_refs[view]
                        for view in inputs.candidate_views
                    },
                    candidate_views=inputs.candidate_views,
                    max_results_per_query=inputs.max_results_per_query,
                )
            )
        finally:
            await provider.close()

        allowed_views = set(inputs.candidate_views)
        response_by_role = {result.role: result for result in response.results}
        results: list[VisualGroundingQueryResult] = []
        evidence: list[Evidence] = []
        for query in inputs.queries:
            provider_result = response_by_role.get(query.role)
            detections = []
            if provider_result is not None and provider_result.query == query.query:
                detections = [
                    detection.model_copy(update={"label": query.query})
                    for detection in provider_result.detections
                    if detection.view in allowed_views
                    and detection.confidence >= query.min_confidence
                ]
            detections.sort(key=lambda detection: detection.confidence, reverse=True)
            detections = detections[: inputs.max_results_per_query]
            results.append(
                VisualGroundingQueryResult(
                    role=query.role,
                    query=query.query,
                    detections=detections,
                )
            )
            evidence.extend(
                Evidence(
                    kind=f"visual_grounding_{query.role}",
                    summary=(
                        detection.evidence_summary
                        or f"Grounded {query.query} in {detection.view}."
                    ),
                    ref=observation.rgb_refs[detection.view],
                    data={
                        "role": query.role,
                        "view": detection.view,
                        "bbox_normalized_1000": detection.bbox_normalized_1000,
                        "bottom_center_normalized_1000": detection.bottom_center(),
                        "confidence": detection.confidence,
                    },
                )
                for detection in detections
            )
        output = VisualGroundingBatchOutput(
            available=True,
            observation_id=observation_id,
            provider=response.provider,
            results=results,
            reason=None,
        )
        found_roles = [result.role for result in results if result.detections]
        return self._batch_result(
            request,
            output,
            status=SkillStatus.SUCCESS,
            summary=(
                f"Batch-grounded roles: {', '.join(found_roles)}."
                if found_roles
                else "No batch grounding query met its confidence threshold."
            ),
            evidence=evidence,
        )

    def _result(
        self,
        request: SkillRequest,
        output: VisualGroundingOutput,
        *,
        status: SkillStatus,
        summary: str,
        evidence: list[Evidence] | None = None,
    ) -> SkillResult:
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=status,
            summary=summary,
            evidence=evidence or [],
            outputs=output.model_dump(mode="json"),
            compressed_history=(
                f"Visual grounding query={output.query!r}; found={output.found}; "
                f"detections={len(output.detections)}."
            ),
            stop_reason=output.reason if status != SkillStatus.SUCCESS else None,
            error=output.reason if status == SkillStatus.FAILURE else None,
        )

    def _batch_result(
        self,
        request: SkillRequest,
        output: VisualGroundingBatchOutput,
        *,
        status: SkillStatus,
        summary: str,
        evidence: list[Evidence] | None = None,
    ) -> SkillResult:
        detections = sum(len(result.detections) for result in output.results)
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=status,
            summary=summary,
            evidence=evidence or [],
            outputs=output.model_dump(mode="json"),
            compressed_history=(
                f"Batch visual grounding: queries={len(output.results)}; "
                f"detections={detections}."
            ),
            stop_reason=output.reason if status != SkillStatus.SUCCESS else None,
            error=output.reason if status == SkillStatus.FAILURE else None,
        )
