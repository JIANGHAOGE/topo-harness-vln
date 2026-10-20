from __future__ import annotations

from agentos.schemas.capability import SkillDescriptor
from agentos.skills.base import Skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._descriptors: dict[str, SkillDescriptor] = {}

    def register(self, skill: Skill, descriptor: SkillDescriptor | None = None) -> None:
        self._skills[skill.name] = skill
        declared = descriptor or getattr(skill, "descriptor", None)
        self._descriptors[skill.name] = declared or SkillDescriptor(
            name=skill.name,
            description=f"Registered embodied skill: {skill.name}",
        )

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Skill is not registered: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._skills)

    def describe(self, name: str) -> SkillDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            raise KeyError(f"Skill is not registered: {name}") from exc

    def catalog(self, *, embodiment: str | None = None) -> list[SkillDescriptor]:
        descriptors = [self._descriptors[name] for name in sorted(self._descriptors)]
        if embodiment is None:
            return descriptors
        return [
            descriptor
            for descriptor in descriptors
            if not descriptor.supported_embodiments or embodiment in descriptor.supported_embodiments
        ]
