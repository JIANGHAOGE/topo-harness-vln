from __future__ import annotations

from dataclasses import dataclass

from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillResult
from agentos.schemas.trace import TraceContext


@dataclass
class RobotROS2Adapter:
    name: str = "robot-ros2"

    async def observe(self, trace: TraceContext) -> Observation:
        del trace
        raise NotImplementedError("ROS2 robot adapter is planned for P11.")

    async def apply_skill_result(self, result: SkillResult) -> dict[str, object]:
        del result
        raise NotImplementedError("ROS2 control safety gate is planned for P11.")

