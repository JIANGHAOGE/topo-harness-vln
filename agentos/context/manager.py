from __future__ import annotations

from typing import Protocol

from agentos.memory.store import GraphMemoryStore
from agentos.schemas.context import ContextEvidence, EmbodiedContext
from agentos.schemas.events import Event
from agentos.schemas.memory import MemoryNodeType, MemoryQuery
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest


class ContextManager(Protocol):
    def build(
        self,
        *,
        task: TaskRequest,
        observation: Observation,
        plan: Plan,
        skill_results: list[SkillResult],
        events: list[Event],
    ) -> EmbodiedContext:
        ...


class BoundedContextManager:
    """Builds compact skill context while keeping raw sensor history external."""

    def __init__(
        self,
        memory: GraphMemoryStore | None = None,
        *,
        max_events: int = 6,
        max_results: int = 4,
        max_evidence: int = 6,
    ) -> None:
        self.memory = memory
        self.max_events = max_events
        self.max_results = max_results
        self.max_evidence = max_evidence

    def build(
        self,
        *,
        task: TaskRequest,
        observation: Observation,
        plan: Plan,
        skill_results: list[SkillResult],
        events: list[Event],
    ) -> EmbodiedContext:
        pose = observation.pose.model_dump(mode="json") if observation.pose else {}
        modalities = {
            "rgb_views": sorted(observation.rgb_refs),
            "depth_views": sorted(observation.depth_refs),
            "topdown_available": bool(observation.topdown_ref),
        }
        observation_parts = [observation.text or "current robot observation", str(modalities)]
        if observation.objects:
            observation_parts.append(f"objects={observation.objects[:8]}")

        evidence: list[ContextEvidence] = []
        if self.memory is not None:
            seen: set[str] = set()

            def add(node, score: float, *, persistent_only: bool = False) -> None:
                if node.node_id in seen or len(evidence) >= self.max_evidence:
                    return
                if persistent_only and node.task_id is not None:
                    return
                evidence.append(self._context_evidence(node, score))
                seen.add(node.node_id)

            entity_hits = self.memory.search(
                MemoryQuery(
                    text=task.instruction,
                    node_types=[MemoryNodeType.ENTITY],
                    persistent_only=True,
                    limit=max(2, self.max_evidence),
                )
            )
            for hit in entity_hits:
                add(hit.node, hit.score, persistent_only=True)
                for neighbor in self.memory.neighbors(hit.node.node_id, limit=3):
                    if neighbor.node_type == MemoryNodeType.PLACE:
                        add(neighbor, hit.score * 0.9, persistent_only=True)
                        break
                if len(evidence) >= self.max_evidence:
                    break
            if len(evidence) < self.max_evidence:
                for hit in self.memory.search(
                    MemoryQuery(
                        text=task.instruction,
                        node_types=[MemoryNodeType.PLACE],
                        persistent_only=True,
                        limit=self.max_evidence,
                    )
                ):
                    add(hit.node, hit.score, persistent_only=True)
                    if len(evidence) >= self.max_evidence:
                        break
            if len(evidence) < self.max_evidence:
                for hit in self.memory.search(
                    MemoryQuery(
                        text=task.instruction,
                        node_types=[MemoryNodeType.EVIDENCE],
                        task_id=task.trace.task_id,
                        limit=1,
                    )
                ):
                    add(hit.node, hit.score)

        recent_events = [
            f"{event.event_type.value}@{event.state.value}"
            for event in events[-self.max_events :]
        ]
        prior_results = [result.compressed_history or result.summary for result in skill_results[-self.max_results :]]
        return EmbodiedContext(
            instruction=task.instruction,
            plan_summary=plan.rationale_summary,
            observation_summary="; ".join(observation_parts),
            robot_state=pose,
            recent_events=recent_events,
            prior_results=prior_results,
            retrieved_evidence=evidence,
        )

    @staticmethod
    def _context_evidence(node, score: float) -> ContextEvidence:
        return ContextEvidence(
            node_id=node.node_id,
            node_type=node.node_type.value,
            summary=node.content or node.title,
            score=score,
            source_ref=node.source_ref,
        )
