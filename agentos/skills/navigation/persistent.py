from __future__ import annotations

import asyncio
import inspect
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agentos.schemas.base import utc_now
from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.schemas.enums import SkillStatus
from agentos.schemas.observation import Observation
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.trace import TraceContext
from agentos.skills.base import SkillContext
from agentos.skills.navigation.omninav import OmniNavSkill


ObservationProvider = Callable[[TraceContext], Awaitable[Observation]]
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
GuidanceProvider = Callable[[Observation], Awaitable["OmniNavGuidance"]]
GuidanceCallback = Callable[["OmniNavGuidance"], Awaitable[None] | None]


@dataclass(frozen=True)
class OmniNavGuidance:
    """One Slow-System decision cached for asynchronous OmniNav consumption."""

    instruction: str
    source_observation: Observation
    payload: Any = None
    terminal_reason: str | None = None


@dataclass(frozen=True)
class GuidedOmniNavOutcome:
    result: SkillResult
    guidance: OmniNavGuidance


@dataclass
class PersistentOmniNavSkill:
    """Isolated sustained OmniNav runner with bounded progress guards."""

    inner: OmniNavSkill
    observe: ObservationProvider
    max_steps: int = 1000
    no_progress_window: int = 50
    minimum_window_progress_m: float = 1.0
    collision_window: int = 12
    max_collisions_per_window: int = 8
    on_progress: ProgressCallback | None = None
    name: str = "NavigationSkill"

    descriptor = SkillDescriptor(
        name="NavigationSkill",
        description=(
            "Sustain source-grounded OmniNav navigation with fresh observations until local STOP "
            "or a bounded runtime guard; returns semantic evidence for Main Agent verification."
        ),
        input_schema={"instruction": {"type": "string"}},
        output_schema={
            "omninav_stop": {"type": "boolean", "scope": "local_subgoal_evidence"},
            "distance_m": {"type": "number"},
            "steps": {"type": "integer"},
            "final_observation_id": {"type": "string"},
        },
        required_modalities=["rgb", "pose"],
        supported_embodiments=["habitat-gs", "robot-ros2"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["sustained_motion", "fresh_observation_capture"],
    )

    async def run(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
    ) -> SkillResult:
        started_at = utc_now()
        current = observation
        distance_m = 0.0
        collision_count = 0
        actions: list[str] = []
        distance_window: deque[float] = deque(maxlen=self.no_progress_window)
        collision_window: deque[int] = deque(maxlen=self.collision_window)

        for frame_index in range(self.max_steps):
            context.cancellation.raise_if_cancelled()
            inner_request = request.model_copy(
                update={
                    "inputs": {
                        **request.inputs,
                        "instruction": request.command,
                        "frame_index": frame_index,
                        "reset": frame_index == 0,
                    }
                }
            )
            result = await self.inner.run(inner_request, current, context)
            if result.status != SkillStatus.SUCCESS:
                return self._finish(
                    request,
                    started_at,
                    status=result.status,
                    summary=f"OmniNav execution failed at step {frame_index + 1}: {result.summary}",
                    current=current,
                    steps=frame_index + 1,
                    distance_m=distance_m,
                    collision_count=collision_count,
                    actions=actions,
                    error=result.error,
                )

            outputs = result.outputs
            execution = outputs.get("execution") if isinstance(outputs.get("execution"), dict) else {}
            step_distance = max(0.0, float(execution.get("distance_m", 0.0)))
            collided = bool(execution.get("collided", False))
            action = str(execution.get("action", "hold"))
            distance_m += step_distance
            collision_count += int(collided)
            actions.append(action)
            distance_window.append(step_distance)
            collision_window.append(int(collided))
            model_stop = bool(outputs.get("omninav_stop"))

            await context.emit_progress(
                request,
                f"OmniNav step {frame_index + 1}: action={action}",
                frame_index=frame_index,
                action=action,
                distance_m=round(distance_m, 4),
                collided=collided,
                omninav_stop=model_stop,
            )
            await self._notify(
                {
                    "step": frame_index + 1,
                    "action": action,
                    "distance_m": round(distance_m, 4),
                    "collided": collided,
                    "omninav_stop": model_stop,
                    "outputs": outputs,
                }
            )

            current = await self.observe(request.trace)
            if model_stop:
                return self._finish(
                    request,
                    started_at,
                    status=SkillStatus.SUCCESS,
                    summary="OmniNav emitted local STOP; semantic subgoal requires Main Agent verification.",
                    current=current,
                    steps=frame_index + 1,
                    distance_m=distance_m,
                    collision_count=collision_count,
                    actions=actions,
                    local_stop=True,
                )
            if len(distance_window) == distance_window.maxlen and sum(distance_window) < self.minimum_window_progress_m:
                return self._finish(
                    request,
                    started_at,
                    status=SkillStatus.RETRYABLE_FAILURE,
                    summary="Navigation stagnated before local subgoal completion.",
                    current=current,
                    steps=frame_index + 1,
                    distance_m=distance_m,
                    collision_count=collision_count,
                    actions=actions,
                    stop_reason="stagnation_guard",
                )
            if (
                len(collision_window) == collision_window.maxlen
                and sum(collision_window) >= self.max_collisions_per_window
            ):
                return self._finish(
                    request,
                    started_at,
                    status=SkillStatus.RETRYABLE_FAILURE,
                    summary="Repeated collisions triggered the runtime guard.",
                    current=current,
                    steps=frame_index + 1,
                    distance_m=distance_m,
                    collision_count=collision_count,
                    actions=actions,
                    stop_reason="collision_guard",
                )

        return self._finish(
            request,
            started_at,
            status=SkillStatus.RETRYABLE_FAILURE,
            summary="Navigation reached its bounded step budget without local STOP.",
            current=current,
            steps=self.max_steps,
            distance_m=distance_m,
            collision_count=collision_count,
            actions=actions,
            stop_reason="max_steps_guard",
        )

    async def run_guided(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
        *,
        initial_guidance: OmniNavGuidance,
        refresh_guidance: GuidanceProvider,
        on_guidance_accepted: GuidanceCallback | None = None,
        slow_update_interval_steps: int = 4,
        max_cached_steps: int = 8,
        max_source_age_s: float = 15.0,
        max_source_displacement_m: float = 1.5,
        max_source_yaw_rad: float = math.pi / 2,
    ) -> GuidedOmniNavOutcome:
        """Run OmniNav from cached Slow guidance while the next Slow update runs."""

        if slow_update_interval_steps < 1 or max_cached_steps < 1:
            raise ValueError("Slow/Fast guidance intervals must be positive")
        if slow_update_interval_steps > max_cached_steps:
            raise ValueError("Slow refresh must start before the cached guidance step limit")
        started_at = utc_now()
        current = observation
        active = initial_guidance
        await self._call_guidance_callback(on_guidance_accepted, active)
        if active.terminal_reason is not None:
            return GuidedOmniNavOutcome(
                result=self._finish(
                    request,
                    started_at,
                    status=SkillStatus.SUCCESS,
                    summary="Slow System produced terminal guidance before OmniNav motion.",
                    current=current,
                    steps=0,
                    distance_m=0.0,
                    collision_count=0,
                    actions=[],
                    stop_reason=active.terminal_reason,
                    extra_outputs={"guidance_terminal_reason": active.terminal_reason},
                ),
                guidance=active,
            )

        pending: asyncio.Task[OmniNavGuidance] | None = None
        guidance_revision = 0
        sent_revision = -1
        cached_steps = 0
        distance_m = 0.0
        collision_count = 0
        actions: list[str] = []
        distance_window: deque[float] = deque(maxlen=self.no_progress_window)
        collision_window: deque[int] = deque(maxlen=self.collision_window)
        scheduler_events: list[dict[str, Any]] = [
            self._guidance_event("slow_applied", active, current)
        ]

        try:
            for frame_index in range(self.max_steps):
                context.cancellation.raise_if_cancelled()
                candidate = self._finished_guidance(pending)
                if candidate is not None:
                    pending = None
                    if self._guidance_is_fresh(
                        candidate,
                        current,
                        max_source_age_s=max_source_age_s,
                        max_source_displacement_m=max_source_displacement_m,
                        max_source_yaw_rad=max_source_yaw_rad,
                    ):
                        active = candidate
                        guidance_revision += 1
                        cached_steps = 0
                        scheduler_events.append(
                            self._guidance_event("slow_applied", active, current)
                        )
                        await self._call_guidance_callback(on_guidance_accepted, active)
                        if active.terminal_reason is not None:
                            return GuidedOmniNavOutcome(
                                result=self._finish(
                                    request,
                                    started_at,
                                    status=SkillStatus.SUCCESS,
                                    summary="Slow System handed control to the terminal target capability.",
                                    current=current,
                                    steps=frame_index,
                                    distance_m=distance_m,
                                    collision_count=collision_count,
                                    actions=actions,
                                    stop_reason=active.terminal_reason,
                                    extra_outputs={
                                        "guidance_terminal_reason": active.terminal_reason,
                                        "slow_fast_scheduler": {
                                            "mode": "asynchronous_cached_guidance",
                                            "events": scheduler_events[-16:],
                                        },
                                    },
                                ),
                                guidance=active,
                            )
                    else:
                        scheduler_events.append(
                            self._guidance_event("stale_slow_dropped", candidate, current)
                        )

                if pending is None and (
                    cached_steps >= slow_update_interval_steps
                ):
                    pending = asyncio.create_task(refresh_guidance(current))
                    scheduler_events.append(
                        self._guidance_event("slow_started", active, current)
                    )

                if cached_steps >= max_cached_steps and pending is not None:
                    candidate = await pending
                    pending = None
                    if not self._guidance_is_fresh(
                        candidate,
                        current,
                        max_source_age_s=max_source_age_s,
                        max_source_displacement_m=max_source_displacement_m,
                        max_source_yaw_rad=max_source_yaw_rad,
                    ):
                        scheduler_events.append(
                            self._guidance_event("stale_slow_dropped", candidate, current)
                        )
                        continue
                    active = candidate
                    guidance_revision += 1
                    cached_steps = 0
                    scheduler_events.append(
                        self._guidance_event("slow_applied", active, current)
                    )
                    await self._call_guidance_callback(on_guidance_accepted, active)
                    if active.terminal_reason is not None:
                        return GuidedOmniNavOutcome(
                            result=self._finish(
                                request,
                                started_at,
                                status=SkillStatus.SUCCESS,
                                summary="Slow System handed control to the terminal target capability.",
                                current=current,
                                steps=frame_index,
                                distance_m=distance_m,
                                collision_count=collision_count,
                                actions=actions,
                                stop_reason=active.terminal_reason,
                                extra_outputs={
                                    "guidance_terminal_reason": active.terminal_reason,
                                    "slow_fast_scheduler": {
                                        "mode": "asynchronous_cached_guidance",
                                        "events": scheduler_events[-16:],
                                    },
                                },
                            ),
                            guidance=active,
                        )

                inner_request = request.model_copy(
                    update={
                        "inputs": {
                            **request.inputs,
                            "instruction": active.instruction,
                            "frame_index": frame_index,
                            "reset": frame_index == 0 or sent_revision != guidance_revision,
                            "slow_guidance_source_observation_id": (
                                active.source_observation.trace.observation_id
                            ),
                        }
                    }
                )
                sent_revision = guidance_revision
                result = await self.inner.run(inner_request, current, context)
                if result.status != SkillStatus.SUCCESS:
                    return GuidedOmniNavOutcome(
                        result=self._finish(
                            request,
                            started_at,
                            status=result.status,
                            summary=f"OmniNav execution failed at step {frame_index + 1}: {result.summary}",
                            current=current,
                            steps=frame_index + 1,
                            distance_m=distance_m,
                            collision_count=collision_count,
                            actions=actions,
                            error=result.error,
                            extra_outputs={
                                "slow_fast_scheduler": {
                                    "mode": "asynchronous_cached_guidance",
                                    "events": scheduler_events[-16:],
                                }
                            },
                        ),
                        guidance=active,
                    )

                outputs = result.outputs
                execution = outputs.get("execution") if isinstance(outputs.get("execution"), dict) else {}
                step_distance = max(0.0, float(execution.get("distance_m", 0.0)))
                collided = bool(execution.get("collided", False))
                action = str(execution.get("action", "hold"))
                distance_m += step_distance
                collision_count += int(collided)
                actions.append(action)
                distance_window.append(step_distance)
                collision_window.append(int(collided))
                cached_steps += 1
                model_stop = bool(outputs.get("omninav_stop"))

                await context.emit_progress(
                    request,
                    f"OmniNav Fast step {frame_index + 1}: action={action}",
                    frame_index=frame_index,
                    action=action,
                    distance_m=round(distance_m, 4),
                    collided=collided,
                    omninav_stop=model_stop,
                    cached_slow_guidance=active.instruction,
                    slow_guidance_source_observation_id=(
                        active.source_observation.trace.observation_id
                    ),
                )
                await self._notify(
                    {
                        "step": frame_index + 1,
                        "action": action,
                        "distance_m": round(distance_m, 4),
                        "collided": collided,
                        "omninav_stop": model_stop,
                        "outputs": outputs,
                    }
                )
                current = await self.observe(request.trace)

                if model_stop:
                    if pending is None:
                        pending = asyncio.create_task(refresh_guidance(current))
                    candidate = await pending
                    pending = None
                    if self._guidance_is_fresh(
                        candidate,
                        current,
                        max_source_age_s=max_source_age_s,
                        max_source_displacement_m=max_source_displacement_m,
                        max_source_yaw_rad=max_source_yaw_rad,
                    ):
                        active = candidate
                        guidance_revision += 1
                        cached_steps = 0
                        scheduler_events.append(
                            self._guidance_event("local_stop_slow_applied", active, current)
                        )
                        await self._call_guidance_callback(on_guidance_accepted, active)
                        if active.terminal_reason is not None:
                            return GuidedOmniNavOutcome(
                                result=self._finish(
                                    request,
                                    started_at,
                                    status=SkillStatus.SUCCESS,
                                    summary=(
                                        "OmniNav reached the cached affordance; Slow System "
                                        "selected the terminal target."
                                    ),
                                    current=current,
                                    steps=frame_index + 1,
                                    distance_m=distance_m,
                                    collision_count=collision_count,
                                    actions=actions,
                                    stop_reason=active.terminal_reason,
                                    extra_outputs={
                                        "guidance_terminal_reason": active.terminal_reason,
                                        "slow_fast_scheduler": {
                                            "mode": "asynchronous_cached_guidance",
                                            "events": scheduler_events[-16:],
                                        },
                                    },
                                ),
                                guidance=active,
                            )
                    else:
                        scheduler_events.append(
                            self._guidance_event("stale_slow_dropped", candidate, current)
                        )

                if (
                    len(distance_window) == distance_window.maxlen
                    and sum(distance_window) < self.minimum_window_progress_m
                ):
                    return GuidedOmniNavOutcome(
                        result=self._finish(
                            request,
                            started_at,
                            status=SkillStatus.RETRYABLE_FAILURE,
                            summary="Navigation stagnated before the Slow-guided affordance completed.",
                            current=current,
                            steps=frame_index + 1,
                            distance_m=distance_m,
                            collision_count=collision_count,
                            actions=actions,
                            stop_reason="stagnation_guard",
                        ),
                        guidance=active,
                    )
                if (
                    len(collision_window) == collision_window.maxlen
                    and sum(collision_window) >= self.max_collisions_per_window
                ):
                    return GuidedOmniNavOutcome(
                        result=self._finish(
                            request,
                            started_at,
                            status=SkillStatus.RETRYABLE_FAILURE,
                            summary="Repeated collisions triggered the Slow-guided OmniNav guard.",
                            current=current,
                            steps=frame_index + 1,
                            distance_m=distance_m,
                            collision_count=collision_count,
                            actions=actions,
                            stop_reason="collision_guard",
                        ),
                        guidance=active,
                    )
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        return GuidedOmniNavOutcome(
            result=self._finish(
                request,
                started_at,
                status=SkillStatus.RETRYABLE_FAILURE,
                summary="Slow-guided OmniNav reached its bounded step budget.",
                current=current,
                steps=self.max_steps,
                distance_m=distance_m,
                collision_count=collision_count,
                actions=actions,
                stop_reason="max_steps_guard",
            ),
            guidance=active,
        )

    async def _notify(self, payload: dict[str, Any]) -> None:
        if self.on_progress is None:
            return
        value = self.on_progress(payload)
        if inspect.isawaitable(value):
            await value

    @staticmethod
    def _finished_guidance(
        pending: asyncio.Task[OmniNavGuidance] | None,
    ) -> OmniNavGuidance | None:
        if pending is None or not pending.done():
            return None
        return pending.result()

    @staticmethod
    async def _call_guidance_callback(
        callback: GuidanceCallback | None,
        guidance: OmniNavGuidance,
    ) -> None:
        if callback is None:
            return
        value = callback(guidance)
        if inspect.isawaitable(value):
            await value

    @staticmethod
    def _guidance_is_fresh(
        guidance: OmniNavGuidance,
        current: Observation,
        *,
        max_source_age_s: float,
        max_source_displacement_m: float,
        max_source_yaw_rad: float,
    ) -> bool:
        source = guidance.source_observation
        if (
            source.trace.episode_id != current.trace.episode_id
            or source.trace.task_id != current.trace.task_id
            or source.trace.plan_revision != current.trace.plan_revision
        ):
            return False
        age_s = max(
            0.0,
            (current.trace.frame_timestamp - source.trace.frame_timestamp).total_seconds(),
        )
        if age_s > max_source_age_s:
            return False
        if source.pose is None or current.pose is None:
            return source.trace.observation_id == current.trace.observation_id
        displacement = math.hypot(
            current.pose.x - source.pose.x,
            current.pose.y - source.pose.y,
        )
        yaw_delta = abs(
            math.atan2(
                math.sin(current.pose.yaw - source.pose.yaw),
                math.cos(current.pose.yaw - source.pose.yaw),
            )
        )
        return (
            displacement <= max_source_displacement_m
            and yaw_delta <= max_source_yaw_rad
        )

    @staticmethod
    def _guidance_event(
        event: str,
        guidance: OmniNavGuidance,
        current: Observation,
    ) -> dict[str, Any]:
        return {
            "event": event,
            "source_observation_id": guidance.source_observation.trace.observation_id,
            "current_observation_id": current.trace.observation_id,
            "terminal_reason": guidance.terminal_reason,
        }

    def _finish(
        self,
        request: SkillRequest,
        started_at,
        *,
        status: SkillStatus,
        summary: str,
        current: Observation,
        steps: int,
        distance_m: float,
        collision_count: int,
        actions: list[str],
        local_stop: bool = False,
        stop_reason: str | None = None,
        error: str | None = None,
        extra_outputs: dict[str, Any] | None = None,
    ) -> SkillResult:
        pose = current.pose.model_dump(mode="json") if current.pose else None
        outputs = {
            "omninav_stop": local_stop,
            "steps": steps,
            "distance_m": round(distance_m, 4),
            "collision_count": collision_count,
            "actions": actions,
            "final_observation_id": current.trace.observation_id,
            "final_pose": pose,
            **(extra_outputs or {}),
        }
        evidence = [
            Evidence(
                kind="navigation_run",
                summary=summary,
                ref=f"observation:{current.trace.observation_id}",
                data=outputs,
            )
        ]
        history = (
            f"steps={steps}; distance_m={distance_m:.3f}; collisions={collision_count}; "
            f"local_stop={local_stop}; final_observation={current.trace.observation_id}"
        )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=status,
            summary=summary,
            evidence=evidence,
            outputs=outputs,
            compressed_history=history,
            stop_reason="omninav_local_stop" if local_stop else stop_reason,
            started_at=started_at,
            completed_at=utc_now(),
            error=error,
        )
