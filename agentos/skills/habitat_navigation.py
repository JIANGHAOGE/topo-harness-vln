from __future__ import annotations

from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.embodiments.habitat_gs import HabitatGSAdapter
from agentos.schemas.enums import SkillStatus
from agentos.schemas.observation import Observation
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.skills.base import SkillContext


class HabitatNavigationSkill:
    """Deterministic smoke-test skill using explicit Habitat actions."""

    name = "NavigationSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Execute a bounded sequence of native Habitat navigation actions.",
        input_schema={"actions": {"type": "array", "items": {"type": "string"}}},
        output_schema={"executed_actions": {"type": "array"}, "final_pose": {"type": "object"}},
        required_modalities=["rgb", "pose"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["moves_robot"],
    )

    def __init__(self, adapter: HabitatGSAdapter) -> None:
        self.adapter = adapter

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        del observation
        actions = request.inputs.get("actions")
        if not isinstance(actions, list) or not actions:
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.FAILURE,
                summary="Habitat smoke skill requires a non-empty explicit actions list.",
                error="missing actions",
            )

        steps: list[dict[str, object]] = []
        for index, action in enumerate(actions):
            context.cancellation.raise_if_cancelled()
            step = self.adapter.step(str(action))
            steps.append(step)
            await context.emit_progress(
                request,
                f"Habitat action {index + 1}/{len(actions)} executed.",
                action=step["action"],
                collided=step["collided"],
                pose=step["pose"],
            )

        final_observation = await self.adapter.observe(request.trace)
        collided_count = sum(int(bool(step["collided"])) for step in steps)
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=f"Executed {len(steps)} native Habitat actions with {collided_count} collisions.",
            evidence=[
                Evidence(
                    kind="habitat_execution",
                    summary="Native Habitat-Sim actions produced a new RGB/pose observation.",
                    data={
                        "actions": [step["action"] for step in steps],
                        "collision_count": collided_count,
                        "final_rgb_refs": final_observation.rgb_refs,
                        "final_pose": final_observation.pose.model_dump(mode="json") if final_observation.pose else None,
                    },
                )
            ],
            outputs={
                "executed_actions": [step["action"] for step in steps],
                "steps": steps,
                "final_rgb_refs": final_observation.rgb_refs,
                "final_pose": final_observation.pose.model_dump(mode="json") if final_observation.pose else None,
            },
            stop_reason="explicit_habitat_stop" if steps[-1]["action"] == "stop" else None,
            compressed_history=f"{len(steps)} Habitat actions; collisions={collided_count}.",
        )
