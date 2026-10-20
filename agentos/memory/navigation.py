from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass

from agentos.memory.store import GraphMemoryStore
from agentos.schemas.base import utc_now
from agentos.schemas.hierarchical import SemanticMapUpdate
from agentos.schemas.memory import MemoryEdge, MemoryEdgeType, MemoryNode, MemoryNodeType, MemoryQuery
from agentos.schemas.navigation_memory import (
    BackboneExplorationCandidate,
    DirectionalSemanticObservation,
    MemoryDelta,
    MetricPoseEvidence,
    NavigationMemoryView,
    NavigationVisualMemory,
    PlaceMemory,
    PlaceMemoryView,
    TaskWorkingMemory,
    TopoMemoryBlock,
    TopologyGoalSearchRecord,
    TopoMemoryFunction,
    TopoMemoryObjectPOI,
    TopoMemoryRoad,
    TopoMemoryView,
)
from agentos.schemas.observation import Observation
from agentos.schemas.semantic_map import EditableSemanticMap, NodeSemantics
from agentos.schemas.task_reasoning import AdvanceNavigationInput
from agentos.topology import BackboneTopology


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"[_-]+", " ", text)
    return {
        term.casefold()
        for term in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if len(term) > 1
    }


def _visited_this_task(place: PlaceMemory) -> bool:
    return place.visit_count > 1 or bool(place.semantic_observations)


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
class AgentNavigationMemory:
    store: GraphMemoryStore
    environment_id: str
    task_id: str
    episode_id: str
    max_relevant_places: int = 5
    place_merge_radius_m: float = 0.75
    visual_evidence_merge_radius_m: float = 0.10

    def __post_init__(self) -> None:
        self._state = self._load_state()
        self._topology_edges: dict[tuple[str, str], dict[str, object]] = {}

    @property
    def revision(self) -> int:
        return self._state.revision

    @property
    def place_count(self) -> int:
        return len(self.places())

    @property
    def route_node_count(self) -> int:
        return sum(place.is_route_node for place in self.places())

    def start_task(self, *, task_id: str, episode_id: str) -> None:
        self.task_id = task_id
        self.episode_id = episode_id
        self._state = self._load_state()
        self._topology_edges = {}

    def record_reasoning(self, request: AdvanceNavigationInput) -> NavigationMemoryView:
        rejected = list(self._state.rejected_branches)
        if request.rejected_branch and request.rejected_branch not in rejected:
            rejected.append(request.rejected_branch)
        self._state = self._state.model_copy(
            update={
                "global_goal": request.global_goal,
                "target_evidence": _unique(request.target_evidence, 8),
                "route_hypothesis": request.route_hypothesis.statement,
                "route_confidence": request.route_hypothesis.confidence,
                "rejected_branches": rejected[-8:],
                "previous_assessment": request.previous_assessment.model_dump(mode="json"),
                "navigation_task_type": request.navigation_task_type,
                "cot": request.cot.model_dump(mode="json") if request.cot is not None else {},
                "last_action": request.action.model_dump(mode="json"),
                "revision": self._state.revision + 1,
            }
        )
        self._save_state()
        return self.view(request.global_goal)

    def record_slow_output(self, output: dict[str, object]) -> None:
        self._state = self._state.model_copy(
            update={
                "last_slow_output": output,
                "revision": self._state.revision + 1,
            }
        )
        self._save_state()

    @property
    def last_slow_output(self) -> dict[str, object] | None:
        return dict(self._state.last_slow_output) if self._state.last_slow_output else None

    def record_place(self, observation: Observation, semantics: SemanticMapUpdate) -> MemoryDelta:
        if observation.pose is None:
            raise ValueError("navigation memory requires a pose")
        pose = self._pose_evidence(observation)
        places = self.places()
        nearest: PlaceMemory | None = None
        nearest_distance = float("inf")
        for place in places:
            if place.source == "map_backbone":
                continue
            if place.pose.frame != pose.frame or place.pose.map_id != pose.map_id:
                continue
            distance = math.hypot(place.pose.x - pose.x, place.pose.y - pose.y)
            if distance < nearest_distance:
                nearest, nearest_distance = place, distance

        revisited = nearest is not None and nearest_distance <= self.place_merge_radius_m
        labels = _unique([*semantics.left, *semantics.front, *semantics.right], 12)
        # Ordinary observations remain searchable evidence. Topology promotion is explicit.
        is_route_node = False
        semantic_observation = DirectionalSemanticObservation(
            yaw=pose.yaw,
            scene_summary=semantics.scene_summary,
            left=semantics.left,
            front=semantics.front,
            right=semantics.right,
            traversable_directions=semantics.traversable_directions,
            observation_id=observation.trace.observation_id,
        )
        if revisited:
            assert nearest is not None
            preserve_route_node = nearest.is_route_node and not is_route_node
            place = nearest.model_copy(
                update={
                    "pose": nearest.pose if preserve_route_node else pose,
                    "navigation_point_xyz_m": (
                        nearest.navigation_point_xyz_m
                        if preserve_route_node
                        else self._navigation_point(observation)
                    ),
                    "scene_summary": (
                        nearest.scene_summary if preserve_route_node else semantics.scene_summary
                    ),
                    "semantic_labels": _unique([*nearest.semantic_labels, *labels], 12),
                    "semantic_observations": [*nearest.semantic_observations, semantic_observation][-4:],
                    "is_route_node": nearest.is_route_node or is_route_node,
                    "route_node_type": (
                        semantics.route_node_type if is_route_node else nearest.route_node_type
                    ),
                    "traversable_directions": (
                        nearest.traversable_directions
                        if preserve_route_node
                        else semantics.traversable_directions
                    ),
                    "visit_count": nearest.visit_count + 1,
                    "last_observation_id": observation.trace.observation_id,
                    "updated_at": utc_now(),
                }
            )
        else:
            place = PlaceMemory(
                place_id=self._next_place_id(places),
                environment_id=self.environment_id,
                episode_id=self.episode_id,
                task_id=self.task_id,
                pose=pose,
                navigation_point_xyz_m=self._navigation_point(observation),
                scene_summary=semantics.scene_summary,
                semantic_labels=labels,
                semantic_observations=[semantic_observation],
                is_route_node=is_route_node,
                route_node_type="none",
                traversable_directions=semantics.traversable_directions,
                last_observation_id=observation.trace.observation_id,
            )

        previous_place_id = self._state.current_place_id
        prior_previous_place_id = self._state.previous_place_id
        self._save_place(place)
        state_update: dict[str, object] = {
            "previous_place_id": (
                prior_previous_place_id if previous_place_id == place.place_id else previous_place_id
            ),
            "current_place_id": place.place_id,
            "revision": self._state.revision + 1,
        }
        if place.is_route_node:
            previous_route_node_id = self._state.current_route_node_id
            if previous_route_node_id and previous_route_node_id != place.place_id:
                self._link_places(
                    previous_route_node_id,
                    place.place_id,
                    observation.trace.observation_id,
                )
                state_update["previous_route_node_id"] = previous_route_node_id
            state_update["current_route_node_id"] = place.place_id
        self._state = self._state.model_copy(update=state_update)
        self._save_state()
        return MemoryDelta(
            revision=self._state.revision,
            current_place=self._place_view(place, relation="current", relevance=1.0),
            created=not revisited,
            revisited=revisited,
            distance_to_place_m=round(nearest_distance, 3) if revisited else None,
            updated_fields=["pose", "semantics", "visit_count"],
        )

    def mark_known_visit(
        self,
        observation: Observation,
        target: PlaceMemory,
        *,
        reach_tolerance_m: float = 0.75,
    ) -> MemoryDelta:
        if observation.pose is None:
            raise ValueError("navigation memory requires a pose")
        distance = math.hypot(target.pose.x - observation.pose.x, target.pose.y - observation.pose.y)
        reached = distance <= reach_tolerance_m
        place = target
        if reached:
            place = target.model_copy(
                update={
                    "visit_count": target.visit_count + 1,
                    "last_observation_id": observation.trace.observation_id,
                    "updated_at": utc_now(),
                }
            )
            self._save_place(place)
            state_update: dict[str, object] = {
                "previous_place_id": self._state.current_place_id,
                "current_place_id": place.place_id,
                "revision": self._state.revision + 1,
            }
            if place.is_route_node:
                previous_route_node_id = self._state.current_route_node_id
                if previous_route_node_id and previous_route_node_id != place.place_id:
                    state_update["previous_route_node_id"] = previous_route_node_id
                state_update["current_route_node_id"] = place.place_id
            self._state = self._state.model_copy(update=state_update)
            self._save_state()
        return MemoryDelta(
            revision=self._state.revision,
            current_place=self._place_view(place, relation="current", relevance=1.0) if reached else None,
            revisited=reached,
            distance_to_place_m=round(distance, 3),
            updated_fields=["visit_count"] if reached else [],
        )

    def record_topology_search(
        self,
        *,
        topology_node_id: str,
        goal: str,
        status: str,
        observation_id: str,
        evidence: str = "",
        planned_topology_node_id: str | None = None,
        observed_topology_node_id: str | None = None,
        increment_attempt: bool = True,
    ) -> TopologyGoalSearchRecord:
        """Update the task overlay without mutating the frozen semantic map."""
        place = self.get_topology_node(topology_node_id)
        if place is None:
            raise ValueError(f"unknown topology node: {topology_node_id}")
        normalized_goal = " ".join(goal.split())[:160]
        if not normalized_goal:
            raise ValueError("topology exploration goal is required")

        records = list(self._state.topology_goal_search)
        matching_index = next(
            (
                index
                for index, record in enumerate(records)
                if record.topology_node_id == topology_node_id
                and record.goal.casefold() == normalized_goal.casefold()
            ),
            None,
        )
        previous = records[matching_index] if matching_index is not None else None
        record = TopologyGoalSearchRecord(
            topology_node_id=topology_node_id,
            planned_topology_node_id=planned_topology_node_id,
            observed_topology_node_id=observed_topology_node_id,
            goal=normalized_goal,
            status=status,
            attempt_count=(
                previous.attempt_count + (1 if increment_attempt else 0)
                if previous is not None
                else 1
            ),
            last_observation_id=observation_id,
            evidence=" ".join(evidence.split())[:240],
        )
        if matching_index is None:
            records.append(record)
        else:
            records[matching_index] = record

        if not _visited_this_task(place):
            place = place.model_copy(
                update={
                    "visit_count": 2,
                    "last_observation_id": observation_id,
                    "updated_at": utc_now(),
                }
            )
            self._save_place(place)
        observed_place = (
            self.get_topology_node(observed_topology_node_id)
            if observed_topology_node_id is not None
            else None
        )
        current_route_place = observed_place or place
        self._state = self._state.model_copy(
            update={
                "previous_route_node_id": (
                    self._state.current_route_node_id
                    if self._state.current_route_node_id != current_route_place.place_id
                    else self._state.previous_route_node_id
                ),
                "current_route_node_id": current_route_place.place_id,
                "topology_goal_search": records[-64:],
                "revision": self._state.revision + 1,
            }
        )
        self._save_state()
        return record

    def record_trajectory_keyframes(
        self,
        keyframes: list[dict[str, object]],
        *,
        source_target: str,
    ) -> list[dict[str, object]]:
        """Persist pose-aware visual evidence independently from the topology graph."""
        places = self.places()
        registered: list[dict[str, object]] = []
        subtask_id = f"navigation-subtask-{uuid.uuid4().hex}"
        for frame in keyframes:
            parsed = self._trajectory_pose(frame)
            if parsed is None:
                continue
            pose, observation_id, navigation_point = parsed

            place = self._nearby_visual_place(places, pose)
            refs = frame.get("rgb_refs")
            image_refs = {
                str(view): str(image_ref)
                for view, image_ref in (refs.items() if isinstance(refs, dict) else [])
                if view in {"left", "front", "right"} and image_ref
            }
            front_ref = image_refs.get("front")
            if place is None:
                frame_index = int(frame.get("frame_index", 0))
                summary = f"Trajectory keyframe while navigating toward {source_target}"[:160]
                place = PlaceMemory(
                    place_id=self._next_place_id(places),
                    environment_id=self.environment_id,
                    episode_id=self.episode_id,
                    task_id=self.task_id,
                    pose=pose,
                    navigation_point_xyz_m=navigation_point,
                    scene_summary=summary,
                    semantic_labels=_unique(["trajectory-keyframe", source_target], 12),
                    source="trajectory_keyframe",
                    trajectory_frame_index=frame_index,
                    front_image_ref=front_ref,
                    image_refs=image_refs,
                    last_observation_id=observation_id,
                )
                self._save_place(place)
                places.append(place)
            elif image_refs:
                place = place.model_copy(
                    update={
                        "front_image_ref": place.front_image_ref or front_ref,
                        "image_refs": {**place.image_refs, **image_refs},
                        "trajectory_frame_index": int(frame.get("frame_index", 0)),
                        "updated_at": utc_now(),
                    }
                )
                self._save_place(place)
            registered.append({**frame, "location_id": place.place_id})
            for view in ("left", "front", "right"):
                image_ref = image_refs.get(view)
                if image_ref is None:
                    continue
                self._save_visual_memory(
                    NavigationVisualMemory(
                        visual_id=f"visual-{uuid.uuid4().hex}-{view}",
                        subtask_id=subtask_id,
                        kind="trajectory_keyframe",
                        environment_id=self.environment_id,
                        episode_id=self.episode_id,
                        task_id=self.task_id,
                        location_id=place.place_id,
                        view=view,
                        image_ref=image_ref,
                        source_target=source_target[:160],
                        observation_id=observation_id,
                        frame_index=int(frame.get("frame_index", 0)),
                        pose=pose,
                        navigation_point_xyz_m=navigation_point,
                    )
                )

        return registered

    def record_route_waypoints(
        self,
        waypoints: list[dict[str, object]],
        *,
        source_target: str,
    ) -> None:
        """Keep dense control waypoints out of the semantic topology graph."""
        del waypoints, source_target

    def record_backbone_topology(self, topology: BackboneTopology) -> dict[str, object]:
        """Materialize a cached geometric backbone in the current task's node table."""
        self._topology_edges = {
            tuple(sorted((edge.source, edge.target))): edge.model_dump(mode="json")
            for edge in topology.edges
        }
        existing = {
            place.last_observation_id: place
            for place in self.places()
            if place.source == "map_backbone"
        }
        node_locations: dict[str, str] = {}
        places = self.places()
        for node in topology.nodes:
            source_id = f"topology:{topology.cache_key}:{node.node_id}"
            place = existing.get(source_id)
            if place is None:
                place = PlaceMemory(
                    place_id=self._next_place_id(places),
                    environment_id=self.environment_id,
                    episode_id=self.episode_id,
                    task_id=self.task_id,
                    pose=MetricPoseEvidence(
                        x=node.x,
                        y=node.y,
                        yaw=0.0,
                        frame=topology.frame,
                        source=topology.source,
                        map_id=topology.map_id,
                    ),
                    navigation_point_xyz_m=node.navigation_point_xyz_m,
                    scene_summary=(
                        "Cached map junction"
                        if node.node_type == "junction"
                        else "Cached open-space center"
                    ),
                    semantic_labels=["map-backbone", node.node_type],
                    topology_node_id=node.node_id,
                    source="map_backbone",
                    is_route_node=True,
                    route_node_type=node.node_type,
                    last_observation_id=source_id,
                )
                self._save_place(place)
                places.append(place)
            elif place.topology_node_id != node.node_id:
                place = place.model_copy(update={"topology_node_id": node.node_id})
                self._save_place(place)
            node_locations[node.node_id] = place.place_id

        for edge in topology.edges:
            source = node_locations.get(edge.source)
            target = node_locations.get(edge.target)
            if source and target:
                self._link_places(source, target, f"topology:{topology.cache_key}")

        if topology.nodes:
            self._state = self._state.model_copy(update={"revision": self._state.revision + 1})
            self._save_state()
        return {
            "map_id": topology.map_id,
            "geometry_revision": topology.geometry_revision,
            "cache_key": topology.cache_key,
            "node_count": len(node_locations),
            "edge_count": len(topology.edges),
            "node_locations": node_locations,
        }

    def apply_semantic_map(self, semantic_map: EditableSemanticMap) -> dict[str, object]:
        """Apply orientation-invariant map knowledge to current fixed backbone nodes."""
        applied: list[str] = []
        available = {
            place.topology_node_id: place
            for place in self.places()
            if place.source == "map_backbone" and place.topology_node_id is not None
        }
        for topology_node_id, node_semantics in semantic_map.nodes.items():
            place = available.get(topology_node_id)
            if place is None:
                continue
            place = self._with_node_semantics(place, node_semantics)
            self._save_place(place)
            applied.append(topology_node_id)
        if applied:
            self._state = self._state.model_copy(
                update={"revision": self._state.revision + 1}
            )
            self._save_state()
        return {
            "map_id": semantic_map.map_id,
            "path_node_count": len(available),
            "semantic_node_count": len(semantic_map.nodes),
            "applied_node_count": len(applied),
            "applied_topology_node_ids": applied,
            "unmatched_topology_node_ids": sorted(set(semantic_map.nodes) - set(applied)),
        }

    def promote_semantic_landmark(
        self,
        observation: Observation,
        semantics: SemanticMapUpdate,
        *,
        label: str,
        confidence: float,
        task_relevance: float,
        node_semantics: NodeSemantics | None = None,
        minimum_confidence: float = 0.90,
    ) -> MemoryDelta | None:
        """Attach strong task semantics to a nearby fixed backbone node without creating one."""
        if confidence < minimum_confidence or task_relevance < minimum_confidence:
            return None
        if observation.pose is None:
            return None
        candidates = [
            place
            for place in self.places()
            if place.source == "map_backbone"
            and place.pose.frame == observation.pose.frame
            and place.pose.map_id == str(observation.metadata.get("map_id") or self.environment_id)
        ]
        if not candidates:
            return None
        place = min(
            candidates,
            key=lambda item: math.hypot(
                item.pose.x - observation.pose.x,
                item.pose.y - observation.pose.y,
            ),
        )
        distance = math.hypot(
            place.pose.x - observation.pose.x,
            place.pose.y - observation.pose.y,
        )
        if distance > self.place_merge_radius_m:
            return None
        pose = self._pose_evidence(observation)
        semantic_observation = DirectionalSemanticObservation(
            yaw=pose.yaw,
            scene_summary=semantics.scene_summary,
            left=semantics.left,
            front=semantics.front,
            right=semantics.right,
            traversable_directions=semantics.traversable_directions,
            observation_id=observation.trace.observation_id,
        )
        labels = _unique([label, *semantics.left, *semantics.front, *semantics.right], 12)
        scene_summary = semantics.scene_summary
        if node_semantics is not None:
            labels = _unique([*node_semantics.search_terms(), *labels], 12)
            scene_summary = node_semantics.summary()
        place = place.model_copy(
            update={
                "scene_summary": scene_summary,
                "semantic_labels": _unique([*labels, *place.semantic_labels], 12),
                "semantic_observations": [*place.semantic_observations, semantic_observation][-4:],
                "node_semantics": node_semantics or place.node_semantics,
                "semantic_confidence": confidence,
                "task_relevance": task_relevance,
                "traversable_directions": semantics.traversable_directions,
                "visit_count": place.visit_count + 1,
                "last_observation_id": observation.trace.observation_id,
                "updated_at": utc_now(),
            }
        )
        self._save_place(place)
        state_update: dict[str, object] = {
            "current_route_node_id": place.place_id,
            "revision": self._state.revision + 1,
        }
        previous_route_node_id = self._state.current_route_node_id
        if previous_route_node_id and previous_route_node_id != place.place_id:
            state_update["previous_route_node_id"] = previous_route_node_id
        self._state = self._state.model_copy(update=state_update)
        self._save_state()
        return MemoryDelta(
            revision=self._state.revision,
            current_place=self._place_view(place, relation="current", relevance=1.0),
            created=False,
            revisited=True,
            distance_to_place_m=round(distance, 3),
            updated_fields=["semantics", "semantic_confidence", "task_relevance"],
        )

    def nearby_backbone_nodes(
        self,
        current_observation: Observation | None = None,
        *,
        limit: int = 4,
    ) -> list[BackboneExplorationCandidate]:
        """Return adjacent fixed-backbone choices, preferring unexplored nodes."""
        current_route = self.get_place(self._state.current_route_node_id or "")
        adjacent: list[PlaceMemory] = []
        if current_route is not None:
            for node in self.store.neighbors(self._place_node_id(current_route.place_id), limit=100):
                payload = node.metadata.get("place")
                if not isinstance(payload, dict):
                    continue
                place = PlaceMemory.model_validate(payload)
                if place.source == "map_backbone":
                    adjacent.append(place)
        candidates = adjacent or [
            place for place in self.places() if place.source == "map_backbone"
        ]
        candidates = [
            place
            for place in candidates
            if current_route is None or place.place_id != current_route.place_id
        ]
        if current_observation is not None and current_observation.pose is not None:
            origin_x = current_observation.pose.x
            origin_y = current_observation.pose.y
        elif current_route is not None:
            origin_x = current_route.pose.x
            origin_y = current_route.pose.y
        else:
            return []
        ranked = sorted(
            candidates,
            key=lambda place: (
                _visited_this_task(place),
                math.hypot(place.pose.x - origin_x, place.pose.y - origin_y),
            ),
        )
        return [
            BackboneExplorationCandidate(
                location_id=place.place_id,
                topology_node_id=place.topology_node_id,
                route_node_type=place.route_node_type,
                distance_m=round(math.hypot(place.pose.x - origin_x, place.pose.y - origin_y), 3),
                visited=_visited_this_task(place),
                visit_count=max(0, place.visit_count - 1),
                scene_summary=place.scene_summary,
                semantic_labels=place.semantic_labels[:6],
                semantic_confidence=place.semantic_confidence,
                task_relevance=place.task_relevance,
            )
            for place in ranked[:limit]
        ]

    def next_backbone_exploration_node(
        self, current_observation: Observation | None = None
    ) -> PlaceMemory | None:
        candidates = self.nearby_backbone_nodes(current_observation, limit=4)
        selected = next((item for item in candidates if not item.visited), None)
        if selected is None:
            return None
        return self.get_place(selected.location_id)

    def topology_node_visited(self, topology_node_id: str) -> bool:
        place = self.get_topology_node(topology_node_id)
        return place is not None and _visited_this_task(place)

    def mark_global_anchor(
        self,
        observation: Observation,
        semantics: SemanticMapUpdate,
        instruction: str,
    ) -> PlaceMemory:
        """Persist the first task observation as a stable instruction-grounded landmark."""
        if self._state.global_anchor_place_id:
            existing = self.get_place(self._state.global_anchor_place_id)
            if existing is not None:
                updates: dict[str, object] = {}
                if instruction and instruction != existing.anchor_instruction:
                    updates["anchor_instruction"] = instruction
                if observation.rgb_refs and not existing.image_refs:
                    updates["image_refs"] = dict(observation.rgb_refs)
                if updates:
                    existing = existing.model_copy(update={**updates, "updated_at": utc_now()})
                    self._save_place(existing)
                self._save_global_anchor_visuals(existing)
                return existing

        if observation.pose is None:
            raise ValueError("navigation memory requires a pose")
        pose = self._pose_evidence(observation)
        labels = _unique([*semantics.left, *semantics.front, *semantics.right], 12)
        places = self.places()
        same_observation_place = next(
            (
                candidate
                for candidate in places
                if candidate.source != "map_backbone"
                and candidate.pose.frame == pose.frame
                and candidate.pose.map_id == pose.map_id
                and math.hypot(candidate.pose.x - pose.x, candidate.pose.y - pose.y)
                <= self.visual_evidence_merge_radius_m
            ),
            None,
        )
        if same_observation_place is not None:
            place = same_observation_place.model_copy(
                update={
                    "is_global_anchor": True,
                    "is_route_node": False,
                    "route_node_type": "none",
                    "anchor_instruction": instruction,
                    "image_refs": dict(observation.rgb_refs),
                    "updated_at": utc_now(),
                }
            )
        else:
            place = PlaceMemory(
                place_id=self._next_place_id(places),
                environment_id=self.environment_id,
                episode_id=self.episode_id,
                task_id=self.task_id,
                pose=pose,
                navigation_point_xyz_m=self._navigation_point(observation),
                scene_summary=semantics.scene_summary,
                semantic_labels=labels,
                semantic_observations=[
                    DirectionalSemanticObservation(
                        yaw=pose.yaw,
                        scene_summary=semantics.scene_summary,
                        left=semantics.left,
                        front=semantics.front,
                        right=semantics.right,
                        traversable_directions=semantics.traversable_directions,
                        observation_id=observation.trace.observation_id,
                    )
                ],
                is_global_anchor=True,
                is_route_node=False,
                route_node_type="none",
                anchor_instruction=instruction,
                image_refs=dict(observation.rgb_refs),
                last_observation_id=observation.trace.observation_id,
            )
        self._save_place(place)
        self._save_global_anchor_visuals(place)
        self._state = self._state.model_copy(
            update={
                "global_anchor_place_id": place.place_id,
                "current_place_id": place.place_id,
                "revision": self._state.revision + 1,
            }
        )
        self._save_state()
        return place

    def _save_global_anchor_visuals(
        self,
        place: PlaceMemory,
    ) -> None:
        for view, image_ref in place.image_refs.items():
            if view not in {"left", "front", "right"} or not image_ref:
                continue
            self._save_visual_memory(
                NavigationVisualMemory(
                    visual_id=f"global-anchor-{self.task_id}-{view}",
                    subtask_id=f"global-anchor-{self.task_id}",
                    kind="global_anchor",
                    environment_id=self.environment_id,
                    episode_id=self.episode_id,
                    task_id=self.task_id,
                    location_id=place.place_id,
                    view=view,
                    image_ref=image_ref,
                    source_target=place.anchor_instruction[:160],
                    observation_id=place.last_observation_id,
                    pose=place.pose,
                    navigation_point_xyz_m=place.navigation_point_xyz_m,
                )
            )

    def get_place(self, place_id: str) -> PlaceMemory | None:
        node = self.store.get_node(self._place_node_id(place_id))
        if node is None:
            return None
        return PlaceMemory.model_validate(node.metadata["place"])

    def get_topology_node(self, topology_node_id: str) -> PlaceMemory | None:
        return next(
            (
                place
                for place in self.places()
                if place.source == "map_backbone"
                and place.topology_node_id == topology_node_id
            ),
            None,
        )

    def previous_place(self, current_observation: Observation | None = None) -> PlaceMemory | None:
        place_id = self._state.current_route_node_id
        if place_id and self._at_place(current_observation, place_id):
            place_id = self._state.previous_route_node_id
        elif current_observation is None and place_id == self._state.current_place_id:
            place_id = self._state.previous_route_node_id
        if not place_id:
            place_id = self._state.previous_place_id
        if not place_id:
            place_id = self._state.global_anchor_place_id
        if not place_id or self._at_place(current_observation, place_id):
            return None
        if current_observation is None and place_id == self._state.current_place_id:
            return None
        return self.get_place(place_id)

    def places(self) -> list[PlaceMemory]:
        hits = self.store.search(
            MemoryQuery(node_types=[MemoryNodeType.PLACE], task_id=self.task_id, limit=500)
        )
        places = [PlaceMemory.model_validate(hit.node.metadata["place"]) for hit in hits]
        return sorted(places, key=lambda place: place.created_at)

    def visual_memories(self) -> list[NavigationVisualMemory]:
        hits = self.store.search(
            MemoryQuery(node_types=[MemoryNodeType.EVIDENCE], task_id=self.task_id, limit=1000)
        )
        visuals = [
            NavigationVisualMemory.model_validate(hit.node.metadata["visual"])
            for hit in hits
            if isinstance(hit.node.metadata.get("visual"), dict)
        ]
        return sorted(visuals, key=lambda visual: visual.created_at)

    def recall_visuals(
        self,
        *,
        scope: str = "task_overview",
        location_id: str | None = None,
        query: str = "",
        limit: int = 4,
    ) -> list[NavigationVisualMemory]:
        """Recall a bounded visual window without placing the full archive in the prompt."""
        visuals = self.visual_memories()
        if location_id:
            visuals = [visual for visual in visuals if visual.location_id == location_id]
        if scope == "latest_subtask":
            trajectory = [visual for visual in visuals if visual.kind == "trajectory_keyframe"]
            if not trajectory:
                return []
            latest_subtask_id = trajectory[-1].subtask_id
            visuals = [visual for visual in trajectory if visual.subtask_id == latest_subtask_id]
        if query:
            query_terms = _terms(query)
            scored: list[tuple[float, NavigationVisualMemory]] = []
            places = {place.place_id: place for place in self.places()}
            for visual in visuals:
                place = places.get(visual.location_id)
                searchable = " ".join(
                    [
                        visual.source_target,
                        place.scene_summary if place else "",
                        " ".join(place.semantic_labels) if place else "",
                        place.anchor_instruction if place else "",
                    ]
                )
                overlap = len(query_terms & _terms(searchable)) / max(1, len(query_terms))
                if overlap > 0:
                    scored.append((overlap, visual))
            scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
            return [visual for _, visual in scored[:limit]]
        if scope == "task_overview":
            trajectory = [visual for visual in visuals if visual.kind == "trajectory_keyframe"]
            return self._sample_evenly(trajectory or visuals, limit)
        return visuals[-limit:]

    def view(
        self,
        query: str = "",
        *,
        current_observation: Observation | None = None,
    ) -> NavigationMemoryView:
        places = self.places()
        visual_memories = self.visual_memories()
        trajectory_visuals = [
            visual for visual in visual_memories if visual.kind == "trajectory_keyframe"
        ]
        latest_subtask_id = trajectory_visuals[-1].subtask_id if trajectory_visuals else None
        current = next((place for place in places if place.place_id == self._state.current_place_id), None)
        global_anchor = next(
            (place for place in places if place.place_id == self._state.global_anchor_place_id),
            None,
        )
        backtrack = self.previous_place(current_observation)
        query_terms = _terms(" ".join([query, *self._state.target_evidence]))
        ranked: list[tuple[float, PlaceMemory]] = []
        for index, place in enumerate(places):
            if current is not None and place.place_id == current.place_id:
                continue
            if global_anchor is not None and place.place_id == global_anchor.place_id:
                continue
            place_terms = _terms(f"{place.scene_summary} {' '.join(place.semantic_labels)}")
            overlap = len(query_terms & place_terms) / len(query_terms) if query_terms else 0.0
            recency = (index + 1) / max(1, len(places))
            topology_bonus = 0.15 if place.is_route_node else 0.0
            score = min(1.0, overlap * 0.7 + recency * 0.15 + topology_bonus)
            ranked.append((score, place))
        ranked.sort(key=lambda item: item[0], reverse=True)
        relevant = [
            self._place_view(place, relation="relevant" if score > 0.2 else "recent", relevance=score)
            for score, place in ranked[: self.max_relevant_places]
        ]
        return NavigationMemoryView(
            revision=self._state.revision,
            task_id=self.task_id,
            global_goal=self._state.global_goal,
            route_hypothesis=self._state.route_hypothesis,
            route_confidence=self._state.route_confidence,
            global_anchor=(
                self._place_view(global_anchor, relation="relevant", relevance=1.0)
                if global_anchor
                else None
            ),
            current_place=self._place_view(current, relation="current", relevance=1.0) if current else None,
            backtrack_location=(
                self._place_view(backtrack, relation="relevant", relevance=1.0)
                if backtrack
                else None
            ),
            relevant_places=relevant,
            nearby_backbone_nodes=self.nearby_backbone_nodes(current_observation),
            rejected_branches=self._state.rejected_branches,
            total_places=len(places),
            route_node_count=sum(place.is_route_node for place in places),
            visual_archive_count=len(visual_memories),
            latest_subtask_id=latest_subtask_id,
            latest_subtask_keyframe_count=len(
                {
                    (visual.location_id, visual.frame_index)
                    for visual in trajectory_visuals
                    if visual.subtask_id == latest_subtask_id
                }
            ),
        )

    def search_semantic_nodes(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        """Return fixed nodes whose orientation-invariant semantics match the query."""
        query_terms = _terms(query) - {
            "a",
            "an",
            "find",
            "go",
            "locate",
            "navigate",
            "please",
            "the",
            "to",
        }
        if not query_terms:
            return []
        matches: list[tuple[float, PlaceMemory, list[str]]] = []
        for place in self.places():
            semantics = place.node_semantics
            if not place.is_route_node or semantics is None:
                continue
            searchable = " ".join(
                [
                    semantics.place_type,
                    semantics.function,
                    semantics.spatial_character,
                    *semantics.objects,
                    *semantics.landmarks,
                    *semantics.task_affordances,
                    *semantics.aliases,
                ]
            )
            searchable_terms = _terms(searchable)
            matched_terms = sorted(query_terms & searchable_terms)
            if not matched_terms:
                continue
            overlap = len(matched_terms) / len(query_terms)
            identity_terms = _terms(" ".join([semantics.place_type, *semantics.aliases]))
            identity_match = bool(query_terms & identity_terms)
            score = min(
                1.0,
                overlap * 0.65 + semantics.confidence * 0.15 + (0.2 if identity_match else 0.0),
            )
            matches.append((score, place, matched_terms))
        matches.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "location_id": place.place_id,
                "topology_node_id": place.topology_node_id,
                "score": round(score, 3),
                "matched_terms": matched_terms,
                "node_semantics": place.node_semantics.model_dump(mode="json"),
            }
            for score, place, matched_terms in matches[:limit]
        ]

    def semantic_node_documents(self) -> list[dict[str, object]]:
        documents: list[dict[str, object]] = []
        for place in self.places():
            if not place.is_route_node or place.node_semantics is None:
                continue
            documents.append(
                {
                    "topology_node_id": place.topology_node_id,
                    "node_semantics": place.node_semantics.model_dump(mode="json"),
                }
            )
        return sorted(
            documents,
            key=lambda item: str(item["topology_node_id"]),
        )

    def summary(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "revision": self._state.revision,
            "total_places": self.place_count,
            "route_node_count": self.route_node_count,
            "visual_archive_count": len(self.visual_memories()),
            "route_hypothesis": self._state.route_hypothesis,
            "rejected_branches": list(self._state.rejected_branches),
            "topology_goal_search_count": len(self._state.topology_goal_search),
        }

    def route_graph(self) -> dict[str, object]:
        """Return a lightweight task route graph for relay visualization."""
        places = [place for place in self.places() if place.is_route_node]
        places_by_id = {place.place_id: place for place in places}
        place_ids = {place.place_id for place in places}
        edges: set[tuple[str, str]] = set()
        for place in places:
            for neighbor in self.store.neighbors(self._place_node_id(place.place_id), limit=100):
                payload = neighbor.metadata.get("place")
                if not isinstance(payload, dict):
                    continue
                neighbor_id = str(payload.get("place_id") or "")
                if neighbor_id in place_ids and neighbor_id != place.place_id:
                    edges.add(tuple(sorted((place.place_id, neighbor_id))))

        return {
            "revision": self._state.revision,
            "topology_policy": "fixed_cached_backbone_with_semantic_annotations",
            "place_merge_radius_m": self.place_merge_radius_m,
            "visual_evidence_merge_radius_m": self.visual_evidence_merge_radius_m,
            "nodes": [
                {
                    "location_id": place.place_id,
                    "topology_node_id": place.topology_node_id,
                    "x": place.pose.x,
                    "y": place.pose.y,
                    "yaw": place.pose.yaw,
                    "frame": place.pose.frame,
                    "map_id": place.pose.map_id,
                    "source": place.source,
                    "route_node_type": place.route_node_type,
                    "semantic_confidence": place.semantic_confidence,
                    "task_relevance": place.task_relevance,
                    "scene_summary": place.scene_summary,
                    "semantic_labels": place.semantic_labels[:6],
                    "node_semantics": (
                        place.node_semantics.model_dump(mode="json")
                        if place.node_semantics is not None
                        else None
                    ),
                    "visited": _visited_this_task(place),
                    "visit_count": max(0, place.visit_count - 1),
                    "goal_search": [
                        record.model_dump(mode="json")
                        for record in self._topology_goal_search_for(place.topology_node_id)
                    ],
                    "traversable_directions": place.traversable_directions,
                    "is_current": place.place_id == self._state.current_route_node_id,
                    "is_global_anchor": place.place_id == self._state.global_anchor_place_id,
                }
                for place in places
            ],
            "edges": [
                self._route_graph_edge(source, target, places_by_id)
                for source, target in sorted(edges)
            ],
        }

    def topo_memory_view(self) -> TopoMemoryView:
        """Project fixed geometry and semantic annotations into ABot-N0's four layers."""
        places = sorted(
            (place for place in self.places() if place.is_route_node),
            key=lambda place: str(place.topology_node_id),
        )
        place_by_id = {place.place_id: place for place in places}
        blocks: list[TopoMemoryBlock] = []
        functions: list[TopoMemoryFunction] = []
        objects: list[TopoMemoryObjectPOI] = []
        for place in places:
            if place.topology_node_id is None:
                continue
            semantics = place.node_semantics
            blocks.append(
                TopoMemoryBlock(
                    topology_node_id=place.topology_node_id,
                    place_type=(
                        semantics.place_type
                        if semantics is not None
                        else place.route_node_type.replace("_", " ")
                    ),
                    scene_summary=place.scene_summary,
                    position_xy_m=[place.pose.x, place.pose.y],
                    visited=_visited_this_task(place),
                    visit_count=max(0, place.visit_count - 1),
                    goal_search=self._topology_goal_search_for(
                        place.topology_node_id
                    ),
                )
            )
            if semantics is not None and (
                semantics.function or semantics.task_affordances
            ):
                functions.append(
                    TopoMemoryFunction(
                        topology_node_id=place.topology_node_id,
                        function=semantics.function,
                        task_affordances=semantics.task_affordances,
                    )
                )
            if semantics is not None and (semantics.objects or semantics.landmarks):
                objects.append(
                    TopoMemoryObjectPOI(
                        topology_node_id=place.topology_node_id,
                        objects=semantics.objects,
                        landmarks=semantics.landmarks,
                    )
                )

        roads: list[TopoMemoryRoad] = []
        seen: set[tuple[str, str]] = set()
        for place in places:
            if place.topology_node_id is None:
                continue
            for neighbor in self.store.neighbors(
                self._place_node_id(place.place_id), limit=100
            ):
                payload = neighbor.metadata.get("place")
                if not isinstance(payload, dict):
                    continue
                neighbor_id = str(payload.get("place_id") or "")
                other = place_by_id.get(neighbor_id)
                if other is None or other.topology_node_id is None:
                    continue
                pair = tuple(sorted((place.topology_node_id, other.topology_node_id)))
                if pair in seen:
                    continue
                seen.add(pair)
                edge = self._topology_edges.get(pair) or {}
                roads.append(
                    TopoMemoryRoad(
                        source=pair[0],
                        target=pair[1],
                        distance_m=(
                            float(edge["distance_m"])
                            if edge.get("distance_m") is not None
                            else None
                        ),
                        transition=(
                            "junction connection"
                            if "junction"
                            in {place.route_node_type, other.route_node_type}
                            else "open-space connection"
                        ),
                    )
                )
        current = self.get_place(self._state.current_route_node_id or "")
        map_id = places[0].pose.map_id if places else self.environment_id
        return TopoMemoryView(
            map_id=map_id,
            revision=self._state.revision,
            current_topology_node_id=(
                current.topology_node_id if current is not None else None
            ),
            block_layer=blocks,
            road_layer=roads,
            function_layer=functions,
            object_poi_layer=objects,
        )

    def _topology_goal_search_for(
        self, topology_node_id: str | None
    ) -> list[TopologyGoalSearchRecord]:
        if topology_node_id is None:
            return []
        return [
            record
            for record in self._state.topology_goal_search
            if record.topology_node_id == topology_node_id
        ][-4:]

    def _route_graph_edge(
        self,
        source: str,
        target: str,
        places_by_id: dict[str, PlaceMemory],
    ) -> dict[str, object]:
        payload: dict[str, object] = {"source": source, "target": target}
        source_node = places_by_id[source].topology_node_id
        target_node = places_by_id[target].topology_node_id
        if source_node is None or target_node is None:
            return payload
        topology_edge = self._topology_edges.get(tuple(sorted((source_node, target_node))))
        if topology_edge is None:
            return payload
        path = topology_edge.get("path_world_xy_m")
        if path:
            payload["distance_m"] = topology_edge["distance_m"]
            payload["path_world_xy_m"] = path
        return payload

    def _load_state(self) -> TaskWorkingMemory:
        node = self.store.get_node(self._task_node_id())
        if node is None:
            state = TaskWorkingMemory(task_id=self.task_id)
            self._state = state
            self._save_state()
            return state
        return TaskWorkingMemory.model_validate(node.metadata["state"])

    def _save_state(self) -> None:
        self.store.upsert_node(
            MemoryNode(
                node_id=self._task_node_id(),
                node_type=MemoryNodeType.TASK,
                title=f"Navigation task {self.task_id}",
                content=self._state.route_hypothesis or self._state.global_goal or "active navigation task",
                source_ref=f"task:{self.task_id}",
                episode_id=self.episode_id,
                task_id=self.task_id,
                metadata={"environment_id": self.environment_id, "state": self._state.model_dump(mode="json")},
            )
        )

    def _save_place(self, place: PlaceMemory) -> None:
        self.store.upsert_node(
            MemoryNode(
                node_id=self._place_node_id(place.place_id),
                node_type=MemoryNodeType.PLACE,
                title=place.scene_summary,
                content=" ".join([place.scene_summary, place.anchor_instruction, *place.semantic_labels]),
                source_ref=f"observation:{place.last_observation_id}",
                episode_id=place.episode_id,
                task_id=place.task_id,
                metadata={"environment_id": self.environment_id, "place": place.model_dump(mode="json")},
            )
        )

    @staticmethod
    def _with_node_semantics(place: PlaceMemory, semantics: NodeSemantics) -> PlaceMemory:
        labels = _unique([*semantics.search_terms(), *place.semantic_labels], 12)
        return place.model_copy(
            update={
                "scene_summary": semantics.summary(),
                "semantic_labels": labels,
                "node_semantics": semantics,
                "semantic_confidence": semantics.confidence,
                "updated_at": utc_now(),
            }
        )

    def _save_visual_memory(self, visual: NavigationVisualMemory) -> None:
        place = self.get_place(visual.location_id)
        searchable = " ".join(
            [
                visual.source_target,
                place.scene_summary if place else "",
                " ".join(place.semantic_labels) if place else "",
                place.anchor_instruction if place else "",
            ]
        )
        self.store.upsert_node(
            MemoryNode(
                node_id=f"navigation-visual:{self.task_id}:{visual.visual_id}",
                node_type=MemoryNodeType.EVIDENCE,
                title=f"{visual.kind} at {visual.location_id}",
                content=searchable,
                source_ref=f"image:{visual.image_ref}",
                episode_id=visual.episode_id,
                task_id=visual.task_id,
                metadata={
                    "environment_id": self.environment_id,
                    "visual": visual.model_dump(mode="json"),
                },
            )
        )

    def _link_places(self, source: str, target: str, observation_id: str) -> None:
        self.store.add_edge(
            MemoryEdge(
                edge_id=f"spatial:{self.task_id}:{source}:{target}:{uuid.uuid4().hex}",
                source_id=self._place_node_id(source),
                target_id=self._place_node_id(target),
                edge_type=MemoryEdgeType.SPATIAL,
                source_ref=f"observation:{observation_id}",
                metadata={"relation": "traversed_to", "environment_id": self.environment_id},
            )
        )

    def _pose_evidence(self, observation: Observation) -> MetricPoseEvidence:
        assert observation.pose is not None
        return MetricPoseEvidence(
            x=observation.pose.x,
            y=observation.pose.y,
            yaw=observation.pose.yaw,
            frame=observation.pose.frame,
            source=observation.pose.source,
            map_id=str(observation.metadata.get("map_id") or self.environment_id),
            confidence=float(observation.metadata.get("pose_confidence", 1.0)),
            timestamp=observation.trace.pose_timestamp,
        )

    @staticmethod
    def _navigation_point(observation: Observation) -> list[float]:
        point = observation.metadata.get("metric_position_xyz")
        if isinstance(point, list) and len(point) == 3:
            return [round(float(value), 3) for value in point]
        assert observation.pose is not None
        return [round(observation.pose.x, 3), 0.0, round(observation.pose.y, 3)]

    @staticmethod
    def _place_view(
        place: PlaceMemory,
        *,
        relation: str,
        relevance: float,
    ) -> PlaceMemoryView:
        return PlaceMemoryView(
            place_id=place.place_id,
            scene_summary=place.scene_summary,
            semantic_labels=place.semantic_labels[:6],
            topology_node_id=place.topology_node_id,
            node_semantics=place.node_semantics,
            visit_count=place.visit_count,
            branch_status=place.branch_status,
            pose=place.pose,
            source=place.source,
            trajectory_frame_index=place.trajectory_frame_index,
            is_global_anchor=place.is_global_anchor,
            is_route_node=place.is_route_node,
            route_node_type=place.route_node_type,
            semantic_confidence=place.semantic_confidence,
            task_relevance=place.task_relevance,
            traversable_directions=place.traversable_directions,
            anchor_instruction=place.anchor_instruction,
            relation=relation,
            relevance=round(relevance, 3),
        )

    def _nearby_visual_place(
        self, places: list[PlaceMemory], pose: MetricPoseEvidence
    ) -> PlaceMemory | None:
        matches = [
            place
            for place in places
            if place.pose.frame == pose.frame
            and place.pose.map_id == pose.map_id
            and math.hypot(place.pose.x - pose.x, place.pose.y - pose.y)
            <= self.visual_evidence_merge_radius_m
        ]
        return min(
            matches,
            key=lambda place: math.hypot(place.pose.x - pose.x, place.pose.y - pose.y),
            default=None,
        )

    def _at_place(self, observation: Observation | None, place_id: str) -> bool:
        if observation is None or observation.pose is None:
            return False
        place = self.get_place(place_id)
        if place is None or place.pose.frame != observation.pose.frame:
            return False
        map_id = str(observation.metadata.get("map_id") or self.environment_id)
        if place.pose.map_id != map_id:
            return False
        return (
            math.hypot(place.pose.x - observation.pose.x, place.pose.y - observation.pose.y)
            <= self.place_merge_radius_m
        )

    def _trajectory_pose(
        self, frame: dict[str, object]
    ) -> tuple[MetricPoseEvidence, str, list[float]] | None:
        pose_data = frame.get("pose")
        if not isinstance(pose_data, dict):
            return None
        observation_id = str(frame.get("observation_id") or "")
        if not observation_id:
            return None
        pose = MetricPoseEvidence(
            x=float(pose_data.get("x", 0.0)),
            y=float(pose_data.get("y", 0.0)),
            yaw=float(pose_data.get("yaw", 0.0)),
            frame=str(pose_data.get("frame") or "unknown"),
            source=str(pose_data.get("source") or "unknown"),
            map_id=str(frame.get("map_id") or self.environment_id),
            confidence=float(frame.get("pose_confidence", 1.0)),
            timestamp=frame.get("timestamp") or utc_now(),
        )
        point = frame.get("navigation_point_xyz_m")
        if not isinstance(point, list) or len(point) != 3:
            point = [pose.x, 0.0, pose.y]
        return pose, observation_id, [round(float(value), 3) for value in point]

    def _task_node_id(self) -> str:
        return f"navigation-task:{self.task_id}"

    def _place_node_id(self, place_id: str) -> str:
        return f"navigation-place:{self.task_id}:{place_id}"

    @staticmethod
    def _sample_evenly(
        visuals: list[NavigationVisualMemory], limit: int
    ) -> list[NavigationVisualMemory]:
        if len(visuals) <= limit:
            return visuals
        if limit == 1:
            return [visuals[-1]]
        indices = [round(index * (len(visuals) - 1) / (limit - 1)) for index in range(limit)]
        return [visuals[index] for index in indices]

    @staticmethod
    def _next_place_id(places: list[PlaceMemory]) -> str:
        highest = max((int(place.place_id.rsplit("-", 1)[1]) for place in places), default=0)
        return f"location-{highest + 1:03d}"
