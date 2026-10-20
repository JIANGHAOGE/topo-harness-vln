from __future__ import annotations

import re
import uuid

from agentos.memory.store import GraphMemoryStore, observation_node_id
from agentos.schemas.enums import VerificationStatus
from agentos.schemas.memory import MemoryEdge, MemoryEdgeType, MemoryNode, MemoryNodeType
from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult


class MemoryWriter:
    """Compress runtime streams into auditable graph records."""

    def __init__(self, store: GraphMemoryStore) -> None:
        self.store = store

    def record_task(self, task: TaskRequest) -> str:
        node_id = self.task_node_id(task)
        self.store.upsert_node(
            MemoryNode(
                node_id=node_id,
                node_type=MemoryNodeType.TASK,
                title=task.instruction,
                content=task.instruction,
                source_ref=f"trace:{task.trace.task_id}",
                episode_id=task.trace.episode_id,
                task_id=task.trace.task_id,
                metadata={"priority": task.priority, "user_id": task.user_id},
            )
        )
        return node_id

    def record_observation(self, task: TaskRequest, observation: Observation) -> str:
        node_id = observation_node_id(task.trace.episode_id, observation.trace.observation_id)
        pose = observation.pose.model_dump(mode="json") if observation.pose else None
        content_parts = [observation.text or "robot observation"]
        if observation.objects:
            content_parts.append(f"objects={observation.objects}")
        if pose:
            content_parts.append(f"pose={pose}")
        self.store.upsert_node(
            MemoryNode(
                node_id=node_id,
                node_type=MemoryNodeType.SOURCE,
                title=f"Observation {observation.trace.observation_id}",
                content="; ".join(content_parts),
                source_ref=f"observation:{observation.trace.observation_id}",
                episode_id=task.trace.episode_id,
                task_id=task.trace.task_id,
                metadata={
                    "rgb_refs": observation.rgb_refs,
                    "depth_refs": observation.depth_refs,
                    "topdown_ref": observation.topdown_ref,
                    "pose": pose,
                },
            )
        )
        self._link(node_id, self.task_node_id(task), MemoryEdgeType.OBSERVED, node_id)
        return node_id

    def record_skill_result(self, task: TaskRequest, result: SkillResult) -> str:
        node_id = f"skill:{result.trace.skill_run_id}"
        self.store.upsert_node(
            MemoryNode(
                node_id=node_id,
                node_type=MemoryNodeType.SKILL_RUN,
                title=f"{result.skill_name}: {result.status.value}",
                content=result.compressed_history or result.summary,
                source_ref=f"skill_run:{result.trace.skill_run_id}",
                episode_id=task.trace.episode_id,
                task_id=task.trace.task_id,
                metadata={"status": result.status.value, "stop_reason": result.stop_reason, "outputs": result.outputs},
            )
        )
        self._link(self.task_node_id(task), node_id, MemoryEdgeType.CONTAINS, node_id)
        for index, evidence in enumerate(result.evidence):
            evidence_id = f"evidence:{result.trace.skill_run_id}:{index}"
            self.store.upsert_node(
                MemoryNode(
                    node_id=evidence_id,
                    node_type=MemoryNodeType.EVIDENCE,
                    title=evidence.kind,
                    content=evidence.summary,
                    source_ref=evidence.ref or f"skill_run:{result.trace.skill_run_id}",
                    episode_id=task.trace.episode_id,
                    task_id=task.trace.task_id,
                    metadata=evidence.data,
                )
            )
            self._link(evidence_id, node_id, MemoryEdgeType.PROVENANCE, evidence_id)
        self._record_grounded_facts(task, result, node_id)
        return node_id

    def record_verification(self, task: TaskRequest, result: VerificationResult) -> str:
        node_id = f"verification:{result.level.value}:{uuid.uuid4().hex}"
        self.store.upsert_node(
            MemoryNode(
                node_id=node_id,
                node_type=MemoryNodeType.SEMANTIC_EVENT,
                title=f"{result.level.value} verification: {result.status.value}",
                content=result.summary,
                source_ref=f"trace:{result.trace.skill_run_id}",
                episode_id=task.trace.episode_id,
                task_id=task.trace.task_id,
                metadata={
                    "status": result.status.value,
                    "passed": result.status == VerificationStatus.SUCCESS,
                    "reasons": result.reasons,
                    **result.metadata,
                },
            )
        )
        self._link(self.task_node_id(task), node_id, MemoryEdgeType.CONTAINS, node_id)
        return node_id

    @staticmethod
    def task_node_id(task: TaskRequest) -> str:
        return f"task:{task.trace.episode_id}:{task.trace.task_id}"

    def _link(self, source_id: str, target_id: str, edge_type: MemoryEdgeType, source_ref: str) -> None:
        self.store.add_edge(
            MemoryEdge(
                edge_id=f"edge:{uuid.uuid4().hex}",
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                source_ref=source_ref,
            )
        )

    def _record_grounded_facts(
        self,
        task: TaskRequest,
        result: SkillResult,
        skill_node_id: str,
    ) -> None:
        outputs = result.outputs or {}
        grounding = outputs.get("slow_brain_grounding")
        slow_input = outputs.get("slow_brain_input")
        if not isinstance(grounding, dict) or not isinstance(slow_input, dict):
            return
        topology = slow_input.get("topology_context")
        if not isinstance(topology, dict):
            return
        map_id = str(topology.get("map_id") or "").strip()
        place_id = str(topology.get("current_topology_node_id") or "").strip()
        if not map_id or not place_id:
            return

        source_observation_id = str(
            outputs.get("slow_guidance_source_observation_id")
            or (outputs.get("fast_system_contract") or {}).get("reference_observation_id")
            or result.trace.observation_id
        )
        place_node_id = f"place:{self._slug(map_id)}:{self._slug(place_id)}"
        place_summary = self._place_summary(topology, place_id)
        existing_place = self.store.get_node(place_node_id)
        self.store.upsert_node(
            MemoryNode(
                node_id=place_node_id,
                node_type=MemoryNodeType.PLACE,
                title=f"{place_id} on {map_id}",
                content=place_summary,
                source_ref=f"observation:{source_observation_id}",
                episode_id=None,
                task_id=None,
                metadata={
                    **(existing_place.metadata if existing_place else {}),
                    "map_id": map_id,
                    "topology_node_id": place_id,
                    "last_seen_observation_id": source_observation_id,
                },
            )
        )

        for role in ("target", "affordance"):
            fact = grounding.get(f"{role}_grounding")
            if not isinstance(fact, dict):
                continue
            confidence = float(fact.get("confidence") or 0.0)
            label = str(fact.get("label") or "").strip()
            bbox = fact.get("bbox_normalized_1000")
            if confidence < 0.90 or not label or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            entity_node_id = (
                f"entity:{self._slug(map_id)}:{self._slug(place_id)}:{self._slug(label)}"
            )
            existing_entity = self.store.get_node(entity_node_id)
            self.store.upsert_node(
                MemoryNode(
                    node_id=entity_node_id,
                    node_type=MemoryNodeType.ENTITY,
                    title=label,
                    content=f"{label} was visually grounded at {place_id} on map {map_id}.",
                    source_ref=f"observation:{source_observation_id}",
                    episode_id=None,
                    task_id=None,
                    metadata={
                        **(existing_entity.metadata if existing_entity else {}),
                        "canonical_label": label.casefold(),
                        "map_id": map_id,
                        "topology_node_id": place_id,
                        "role": role,
                        "confidence": confidence,
                        "view": fact.get("view"),
                        "bbox_normalized_1000": bbox,
                        "last_seen_observation_id": source_observation_id,
                    },
                )
            )
            evidence_id = f"evidence:grounding:{result.trace.skill_run_id}:{role}"
            self.store.upsert_node(
                MemoryNode(
                    node_id=evidence_id,
                    node_type=MemoryNodeType.EVIDENCE,
                    title=f"Visual grounding: {label}",
                    content=(
                        str(fact.get("evidence_summary") or "").strip()
                        or f"Current visual evidence grounded {label} with confidence {confidence:.2f}."
                    ),
                    source_ref=f"observation:{source_observation_id}",
                    episode_id=task.trace.episode_id,
                    task_id=task.trace.task_id,
                    metadata={
                        "confidence": confidence,
                        "view": fact.get("view"),
                        "bbox_normalized_1000": bbox,
                    },
                )
            )
            self._link(entity_node_id, place_node_id, MemoryEdgeType.LOCATED_AT, evidence_id)
            self._link(evidence_id, entity_node_id, MemoryEdgeType.PROVENANCE, evidence_id)
            self._link(skill_node_id, evidence_id, MemoryEdgeType.CONTAINS, evidence_id)

    @staticmethod
    def _place_summary(topology: dict, place_id: str) -> str:
        for node in topology.get("nodes") or []:
            if not isinstance(node, dict) or node.get("topology_node_id") != place_id:
                continue
            parts = [
                str(node.get("place_type") or "place"),
                str(node.get("scene_summary") or ""),
                str(node.get("function") or ""),
            ]
            objects = node.get("objects") or []
            landmarks = node.get("landmarks") or []
            if objects:
                parts.append("objects=" + ", ".join(str(item) for item in objects[:12]))
            if landmarks:
                parts.append("landmarks=" + ", ".join(str(item) for item in landmarks[:8]))
            return "; ".join(part for part in parts if part)
        return f"Semantic topology place {place_id}."

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return slug or "unknown"
