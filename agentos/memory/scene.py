from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from agentos.schemas.hierarchical import SceneSemanticDescription
from agentos.schemas.observation import Observation


def _terms(scene: SceneSemanticDescription) -> set[str]:
    text = " ".join(
        [scene.room_hint, *scene.left, *scene.front, *scene.right, *scene.distinctive_landmarks]
    )
    return {term.casefold() for term in re.findall(r"[\w-]+", text) if len(term) > 1}


def _unique(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key and key not in seen:
            result.append(item)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


@dataclass
class SceneMemoryRecord:
    scene_id: str
    observation_id: str
    x: float
    z: float
    scene: SceneSemanticDescription
    visit_count: int = 1
    attempted_anchors: list[str] = field(default_factory=list)
    last_status: str | None = None
    last_reached: bool | None = None

    def compact(self) -> str:
        landmarks = _unique(
            [*self.scene.distinctive_landmarks, *self.scene.left, *self.scene.front, *self.scene.right],
            6,
        )
        tried = " | ".join(self.attempted_anchors[-4:]) or "none"
        return (
            f"{self.scene_id} @({self.x:.1f},{self.z:.1f}) visits={self.visit_count}; "
            f"place={self.scene.room_hint or 'unknown'}; seen={' | '.join(landmarks) or 'unspecified'}; "
            f"tried={tried}; last={self.last_status or 'pending'}"
        )


class TaskSceneMemory:
    """Small task-scoped memory for recognizing previously visited stopping places."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.records: list[SceneMemoryRecord] = []

    def record_visit(
        self,
        observation: Observation,
        scene: SceneSemanticDescription,
        *,
        selected_anchor: str,
    ) -> dict[str, object]:
        if observation.pose is None:
            raise ValueError("scene memory requires an observation pose")
        x = float(observation.pose.x)
        z = float(observation.pose.y)
        record = self._find_match(x, z, scene)
        revisited = record is not None
        if record is None:
            record = SceneMemoryRecord(
                scene_id=f"scene-{len(self.records) + 1:03d}",
                observation_id=observation.trace.observation_id,
                x=x,
                z=z,
                scene=scene,
            )
            self.records.append(record)
        else:
            record.visit_count += 1
            record.observation_id = observation.trace.observation_id
            record.x = x
            record.z = z
            record.scene = self._merge(record.scene, scene)

        anchor_repeated = selected_anchor.casefold() in {
            anchor.casefold() for anchor in record.attempted_anchors
        }
        if not anchor_repeated:
            record.attempted_anchors.append(selected_anchor)
        return {
            "scene_id": record.scene_id,
            "revisited": revisited,
            "visit_count": record.visit_count,
            "anchor_repeated": anchor_repeated,
            "previously_attempted_anchors": list(record.attempted_anchors[:-1] if not anchor_repeated else record.attempted_anchors),
        }

    def record_result(self, scene_id: str, *, status: str, reached: bool) -> None:
        record = next((item for item in self.records if item.scene_id == scene_id), None)
        if record is not None:
            record.last_status = status
            record.last_reached = reached

    def compact_index(self) -> list[str]:
        return [record.compact() for record in self.records]

    def summary(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "unique_scenes": len(self.records),
            "total_visits": sum(record.visit_count for record in self.records),
            "revisited_scenes": sum(record.visit_count > 1 for record in self.records),
            "scene_index": self.compact_index(),
        }

    def _find_match(
        self,
        x: float,
        z: float,
        scene: SceneSemanticDescription,
    ) -> SceneMemoryRecord | None:
        scene_terms = _terms(scene)
        candidates: list[tuple[float, SceneMemoryRecord]] = []
        for record in self.records:
            distance = math.hypot(record.x - x, record.z - z)
            if distance > 1.5:
                continue
            existing_terms = _terms(record.scene)
            union = scene_terms | existing_terms
            similarity = len(scene_terms & existing_terms) / len(union) if union else 0.0
            if distance <= 0.65 or similarity >= 0.45:
                candidates.append((distance, record))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _merge(
        current: SceneSemanticDescription,
        incoming: SceneSemanticDescription,
    ) -> SceneSemanticDescription:
        return SceneSemanticDescription(
            room_hint=incoming.room_hint or current.room_hint,
            left=_unique([*incoming.left, *current.left], 6),
            front=_unique([*incoming.front, *current.front], 6),
            right=_unique([*incoming.right, *current.right], 6),
            distinctive_landmarks=_unique(
                [*incoming.distinctive_landmarks, *current.distinctive_landmarks],
                8,
            ),
        )
