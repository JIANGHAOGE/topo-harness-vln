from __future__ import annotations

import asyncio

from agentos.schemas.capability import SkillDescriptor
from agentos.schemas.enums import SkillStatus
from agentos.schemas.observation import Observation
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.skills.base import SkillContext


class MockNavigationSkill:
    name = "NavigationSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Deterministic navigation skill used for AgentOS lifecycle tests.",
        required_modalities=["rgb", "pose"],
        supported_embodiments=["mock"],
    )

    def __init__(self, *, delay_s: float = 0.0, force_status: SkillStatus = SkillStatus.SUCCESS) -> None:
        self.delay_s = delay_s
        self.force_status = force_status

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, "Mock navigation received observation.", observation_id=observation.trace.observation_id)

        stop_requested = "stop" in request.command.lower()
        status = self.force_status
        evidence = [
            Evidence(
                kind="mock_navigation",
                summary="Generated a compact local navigation result.",
                data={
                    "omninav_stop": stop_requested,
                    "trajectory_coord_frame": "[forward,left,yaw]",
                },
            )
        ]
        outputs = {
            "trajectory": [] if stop_requested else [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "model_goal_r": None if stop_requested else 0.5,
            "model_goal_theta": None if stop_requested else 0.0,
            "omninav_stop": stop_requested,
        }
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=status,
            summary="Mock NavigationSkill completed a bounded skill run.",
            evidence=evidence,
            outputs=outputs,
            stop_reason="mock_omninav_stop" if stop_requested else None,
            compressed_history="1 mock observation, 1 compact trajectory/stop decision.",
        )
