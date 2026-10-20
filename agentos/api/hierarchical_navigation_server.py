from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentos.context.manager import BoundedContextManager
from agentos.core.runtime import AgentRuntime, RuntimeResult
from agentos.api.habitat_tool_server import ThreadBoundHabitatAdapter
from agentos.core.events import InMemoryEventBus
from agentos.evolution import TraceDrivenSkillEvolvement
from agentos.harness.verifier import Verifier
from agentos.embodiments.habitat_gs import HabitatGSAdapter, HabitatGSConfig
from agentos.memory.writer import MemoryWriter
from agentos.providers.config import build_provider_from_config, load_llm_config
from agentos.providers.base import LLMProvider
from agentos.providers.grounding import QwenVisualGroundingProvider
from agentos.providers.vision import audit_observation_views, audited_observation_content
from agentos.harness.navigation_session import (
    AgentOSMainAgentSession,
    N1NavigationSkillSession,
)
from agentos.planning.agentos_hierarchical import (
    AgentOSHierarchicalPlanner,
    compress_navigation_handoff,
)
from agentos.planning.semantic_nodes import SemanticNodeRanker
from agentos.schemas.base import utc_now
from agentos.schemas.enums import EventType, RuntimeState
from agentos.schemas.events import Event
from agentos.memory.navigation import AgentNavigationMemory
from agentos.memory.scene import TaskSceneMemory
from agentos.memory.semantic_map import SemanticMapStore
from agentos.memory.store import InMemoryGraphMemory, SQLiteGraphMemory
from agentos.mapping import HabitatNavmeshMapProvider, NavigabilityMapProvider
from agentos.schemas.hierarchical import (
    AnchorPixel,
    DualPixelGoal,
    GeneratedNavigationPoint,
    HierarchicalNavigateInput,
    HierarchicalTaskFinishInput,
    KnownLocationNavigateInput,
    LocalizationMapInput,
    SemanticMapUpdate,
    ViewpointAdjustmentInput,
)
from agentos.schemas.abot import (
    ABotFastExecution,
    ABotPlannerContext,
    ABotSlowContext,
    ABotSlowOutput,
    ABotTargetEvidence,
)
from agentos.schemas.observation import Observation
from agentos.schemas.llm import ChatMessage
from agentos.schemas.navigation_memory import NavigationVisualRecallInput
from agentos.schemas.navigation import EmbodimentExecutionTelemetry
from agentos.schemas.semantic_map import EditableSemanticMap, NodeSemantics, SemanticMapSearchInput
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task_reasoning import (
    AdvanceNavigationInput,
    AgenticPlan,
    AgenticPlannerOutput,
    AgenticPlannerSubtask,
    AgenticSubtask,
    FinishVerifierAssessment,
    FinishVerifierDecision,
    InstructionFollowingCoT,
    ObjectGoalCoT,
    RegionalAnchorGuidance,
    SelectedNavigationAction,
    SemanticNodeSelection,
    SemanticRoutePlan,
    SemanticTaskDecompositionInput,
    SubtaskReflection,
)
from agentos.schemas.trace import TraceContext
from agentos.schemas.task import TaskRequest
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.navigation import (
    AgentOSHierarchicalNavigationSkill,
    AgentOSMappingSkill,
    AnchorGenerationSkill,
    AnchorGroundingSkill,
    LocalizationMapSkill,
    PathExecutionSkill,
    PathPlanningSkill,
    PixelToPointSkill,
    PointNavigationSkill,
    OmniNavSkill,
    PersistentOmniNavSkill,
    ViewpointAdjustmentSkill,
)
from agentos.skills.registry import SkillRegistry
from agentos.skills.perception import VisualGroundingSkill
from agentos.topology import BackboneTopology, BackboneTopologyCache
from agentos.verification.llm import LLMFinishVerifier, LLMSkillVerifier, ProgressRuntimeVerifier


def _compact_error_message(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_input=False)
        messages = []
        for error in errors[:6]:
            location = ".".join(str(item) for item in error.get("loc") or ()) or "request"
            messages.append(f"{location}: {error.get('msg', 'invalid value')}")
        if len(errors) > 6:
            messages.append(f"and {len(errors) - 6} more validation errors")
        return ("Validation failed: " + "; ".join(messages))[:760]
    return str(exc)[:760]


class HierarchicalNavigationService:
    """Independent Qwen-grounding -> depth -> navmesh -> Habitat-step service."""

    anchor_confidence_threshold = 0.80
    target_confidence_threshold = max(
        0.0,
        min(1.0, float(os.environ.get("CEE_TARGET_CONFIDENCE_THRESHOLD", "0.90"))),
    )
    anchor_max_depth_m = max(
        0.1,
        min(20.0, float(os.environ.get("CEE_ANCHOR_MAX_DEPTH_M", "5.0"))),
    )
    semantic_topology_threshold = 0.90
    semantic_node_selection_threshold = 0.70
    point_navigation_timeout_s = 300.0
    point_navigation_total_step_limit = 1000
    execution_strategy = "anchor_chain"

    def __init__(
        self,
        adapter: ThreadBoundHabitatAdapter,
        skills: SkillRegistry,
        *,
        scene_memory_enabled: bool = False,
        navigation_memory: AgentNavigationMemory | None = None,
        environment_id: str = "unknown",
        topology_provider: NavigabilityMapProvider | None = None,
        topology_cache: BackboneTopologyCache | None = None,
        semantic_map_store: SemanticMapStore | None = None,
        semantic_node_provider_factory: Callable[[], LLMProvider] | None = None,
        finish_verifier_provider_factory: Callable[[], LLMProvider] | None = None,
        agent_provider_factory: Callable[[], LLMProvider] | None = None,
        omninav_endpoint_url: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.skills = skills
        self.episode_id = self._new_episode_id()
        self.task_id = self._new_task_id()
        self.scene_memory_enabled = scene_memory_enabled
        self._scene_memory = TaskSceneMemory(self.task_id)
        if navigation_memory is None:
            self.navigation_memory = AgentNavigationMemory(
                InMemoryGraphMemory(),
                environment_id=environment_id,
                task_id=self.task_id,
                episode_id=self.episode_id,
            )
        else:
            self.navigation_memory = navigation_memory
            self.navigation_memory.start_task(task_id=self.task_id, episode_id=self.episode_id)
        self.observation_index = 0
        self._observations: dict[str, Observation] = {}
        self._latest_observation: dict[str, Any] | None = None
        self._latest_pipeline: dict[str, Any] | None = None
        self._pipeline_lock = threading.RLock()
        self._topology_provider = topology_provider
        self._topology_cache = topology_cache or (
            BackboneTopologyCache(topology_provider.cache_dir)
            if topology_provider is not None
            else None
        )
        self._topology_loaded_for_task = False
        self._topology_state: dict[str, object] | None = None
        self._topology_model: BackboneTopology | None = None
        self._semantic_map_store = semantic_map_store
        self._semantic_map_state: dict[str, object] | None = None
        self._semantic_map_model: EditableSemanticMap | None = None
        self._semantic_map_load_attempted = False
        self._semantic_route_plan: SemanticRoutePlan | None = None
        self._active_target_evidence: ABotTargetEvidence | None = None
        self._pending_global_goal = ""
        self._planner_required = False
        self._semantic_node_ranker = (
            SemanticNodeRanker(semantic_node_provider_factory)
            if semantic_node_provider_factory is not None
            else None
        )
        self._finish_verifier_provider_factory = finish_verifier_provider_factory
        self._agent_provider_factory = agent_provider_factory
        self._omninav_endpoint_url = (
            omninav_endpoint_url.strip() if omninav_endpoint_url else None
        )
        self._latest_omninav: dict[str, Any] | None = None
        self._agent_task_lock = threading.Lock()
        self._agent_task_state: dict[str, Any] = {"status": "idle"}
        self._active_agent_runtime: AgentRuntime | None = None

    def load_environment_map(self) -> dict[str, object]:
        """Load and freeze the environment topology for this service lifetime."""
        with self._pipeline_lock:
            return self._load_topology_for_task()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "cee-agentos-hierarchical-navigation",
            "version_profile": "hierarchical",
            "simulator_ready": self.adapter.sim is not None,
            "skill_count": len(self.skills.names()),
            "task_id": self.task_id,
            "scene_memory_enabled": self.scene_memory_enabled,
            "scene_memory_count": len(self._scene_memory.records),
            "explored_location_count": self.navigation_memory.place_count,
            "topology_route_node_count": self.navigation_memory.route_node_count,
            "memory_revision": self.navigation_memory.revision,
            "anchor_confidence_threshold": self.anchor_confidence_threshold,
            "target_confidence_threshold": self.target_confidence_threshold,
            "anchor_max_depth_m": self.anchor_max_depth_m,
            "semantic_topology_threshold": self.semantic_topology_threshold,
            "execution_strategy": self.execution_strategy,
            "agentos_ready": self._agent_provider_factory is not None,
            "agent_harness": {
                "main_agent": "AgentOSHierarchicalPlanner",
                "skill_runner": "AgentOSHierarchicalNavigationSkill",
                "slow_brain": "QwenNavigationSlowBrain(semantic_queries_only)",
                "visual_grounding": "VisualGroundingSkill(Qwen3-VL Slow-selected single-view parallel)",
                "fast_system": (
                    "point_goal_astar_or_object_goal_pixel_astar_or_"
                    "slow_guided_omninav_cached_trajectory"
                ),
                "navigation_capabilities": [
                    "point_goal",
                    "object_goal",
                    "instruction_following",
                ],
                "verifier": {
                    "component": "Verifier",
                    "stages": ["runtime", "skill_level", "finish_time"],
                },
                "context_management": "bounded_graph_retrieval_and_compression",
                "long_term_memory": "sqlite_typed_graph_high_confidence_visual_facts",
                "skill_evolvement": "trace_driven_gated_assets",
                "hermes_role": "task_client",
                "omninav_enabled": self._omninav_endpoint_url is not None,
            },
            "topology": self._topology_state,
            "semantic_map": self._semantic_map_state,
            "semantic_route_plan": (
                self._semantic_route_plan.model_dump(mode="json")
                if self._semantic_route_plan is not None
                else None
            ),
            "pipeline": [
                "MainAgentSkillDelegation",
                "QwenNavigationSlowBrainSemanticQueries",
                "VisualGroundingSkillSlowSelectedSingleViewParallel",
                "DualPixelGoalAssemblyAndProjection",
                "PointNavigationSkill",
                "SlowAffordanceToAsyncOmniNavFastCapability",
                "HighConfidenceTargetToObjectGoalHandoff",
                "LLMSkillVerifier",
                "LLMFinishVerifier",
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version_profile": "hierarchical",
            "skills": [descriptor.model_dump(mode="json") for descriptor in self.skills.catalog()],
        }

    def agent_task_status(self) -> dict[str, Any]:
        with self._agent_task_lock:
            return dict(self._agent_task_state)

    def start_agent_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("instruction is required")
        if self._agent_provider_factory is None:
            raise ValueError("AgentOS provider is not configured")
        timeout_s = float(payload.get("timeout_s", 1800.0))
        if timeout_s <= 0.0 or timeout_s > 7200.0:
            raise ValueError("timeout_s must be between 0 and 7200 seconds")
        max_revisions = int(payload.get("max_plan_revisions", 20))
        if max_revisions < 1 or max_revisions > 100:
            raise ValueError("max_plan_revisions must be between 1 and 100")

        with self._agent_task_lock:
            if self._agent_task_state.get("status") in {"starting", "running"}:
                raise ValueError("an AgentOS task is already running")
            task_id = f"agent-task-{uuid.uuid4().hex}"
            self._agent_task_state = {
                "ok": True,
                "task_id": task_id,
                "instruction": instruction,
                "status": "starting",
                "agent_state": RuntimeState.PLAN.value,
                "step": 0,
                "distance_m": 0.0,
                "plan_revision": 0,
                "max_plan_revisions": max_revisions,
                "agent_harness": self.health()["agent_harness"],
                "started_at": utc_now().isoformat(),
            }
        worker = threading.Thread(
            target=self._run_agent_task,
            args=(task_id, instruction, timeout_s, max_revisions, payload),
            name=f"hierarchical-agent-{task_id[-8:]}",
            daemon=True,
        )
        worker.start()
        return self.agent_task_status()

    def _run_agent_task(
        self,
        task_id: str,
        instruction: str,
        timeout_s: float,
        max_revisions: int,
        payload: dict[str, Any],
    ) -> None:
        try:
            asyncio.run(
                self._run_agent_task_async(
                    task_id,
                    instruction,
                    timeout_s,
                    max_revisions,
                    payload,
                )
            )
        except Exception as exc:
            self._update_agent_task(
                task_id,
                status="failed",
                agent_state=RuntimeState.FAILED.value,
                reason=f"{type(exc).__name__}: {exc}",
                finished_at=utc_now().isoformat(),
            )

    async def _run_agent_task_async(
        self,
        task_id: str,
        instruction: str,
        timeout_s: float,
        max_revisions: int,
        payload: dict[str, Any],
    ) -> None:
        assert self._agent_provider_factory is not None
        begin = await asyncio.to_thread(
            self.begin_navigation,
            {"global_goal": instruction},
        )
        initial = Observation.model_validate(begin["observation"])
        provider = self._agent_provider_factory()
        event_bus = InMemoryEventBus()
        event_bus.subscribe(lambda event: self._on_agent_event(task_id, event))
        initial_handoff = compress_navigation_handoff(begin)
        main_agent_session = AgentOSMainAgentSession(
            global_goal=instruction,
            context_snapshot=initial_handoff,
        )
        navigation_skill_session = N1NavigationSkillSession(
            global_goal=instruction,
            handoff=initial_handoff,
        )
        registry = SkillRegistry()
        omninav_skill = None
        if self._omninav_endpoint_url is not None:
            async def observe_after_omninav(_: TraceContext) -> Observation:
                return await asyncio.to_thread(self.capture_skill_observation)

            async def report_omninav_progress(update: dict[str, Any]) -> None:
                self._latest_omninav = {
                    "ok": True,
                    "instruction": instruction,
                    "result": {"outputs": update.get("outputs", {})},
                }
                self._update_agent_task(
                    task_id,
                    omninav_step=update.get("step"),
                    omninav_action=update.get("action"),
                    active_skill_distance_m=update.get("distance_m"),
                )

            omninav_skill = PersistentOmniNavSkill(
                inner=OmniNavSkill(
                    endpoint_url=self._omninav_endpoint_url,
                    embodiment=self.adapter,
                    name="OmniNavFastCapability",
                ),
                observe=observe_after_omninav,
                max_steps=max(1, min(10000, int(payload.get("max_omninav_steps", 1000)))),
                on_progress=report_omninav_progress,
                name="NavigationSkill",
            )
        registry.register(
            AgentOSHierarchicalNavigationSkill(
                provider,
                self,
                navigation_skill_session,
                visual_grounding_skill=(
                    self.skills.get("VisualGroundingSkill")
                    if "VisualGroundingSkill" in self.skills.names()
                    else None
                ),
                omninav_skill=omninav_skill,
            )
        )
        registry.register(AgentOSMappingSkill(self))
        planner = AgentOSHierarchicalPlanner(
            provider,
            self,
            main_agent_session,
            skill_catalog=registry.catalog(),
            memory=self.navigation_memory.store,
            semantic_node_ranker=self._semantic_node_ranker,
        )

        async def current_observation(_: TraceContext) -> Observation:
            return self.current_observation()

        runtime = AgentRuntime(
            planner=planner,
            replanner=planner,
            skills=registry,
            verifier=Verifier(
                runtime=ProgressRuntimeVerifier(),
                skill=LLMSkillVerifier(provider),
                finish=LLMFinishVerifier(provider),
            ),
            context_manager=BoundedContextManager(self.navigation_memory.store),
            memory_writer=MemoryWriter(self.navigation_memory.store),
            skill_evolvement=TraceDrivenSkillEvolvement("logs/agentos_evolution/traces"),
            max_plan_revisions=max_revisions,
            event_bus=event_bus,
            observation_provider=current_observation,
        )
        self._active_agent_runtime = runtime
        self._update_agent_task(task_id, status="running")
        now = utc_now()
        trace = initial.trace.model_copy(
            update={
                "task_id": task_id,
                "plan_revision": 0,
                "skill_run_id": "agent-root",
                "deadline": now + timedelta(seconds=timeout_s),
            }
        )
        task = TaskRequest(
            trace=trace,
            instruction=instruction,
            success_conditions=[str(item) for item in payload.get("success_conditions", [])],
            metadata={
                "embodiment": "habitat-gs",
                "execution_profile": "hierarchical_tools",
                "omninav_enabled": self._omninav_endpoint_url is not None,
            },
        )
        try:
            result = await runtime.run_task(task, initial)
            self._complete_agent_task(task_id, result)
        finally:
            await provider.close()
            self._active_agent_runtime = None

    def current_observation(self) -> Observation:
        with self._pipeline_lock:
            observation_id = ((self._latest_observation or {}).get("trace") or {}).get(
                "observation_id"
            )
            observation = self._observations.get(str(observation_id or ""))
            if observation is None:
                raise RuntimeError("Hierarchical AgentOS has no current observation")
            return observation

    def capture_skill_observation(self) -> Observation:
        """Capture a fresh observation for a sustained NavigationSkill capability."""

        return self._capture_observation()

    def localization_topology_snapshot(
        self, observation: Observation
    ) -> dict[str, Any]:
        """Return embodiment-neutral pose and bounded Topo-Memory evidence."""
        topo_memory = self.navigation_memory.topo_memory_view()
        nearby_nodes = self.navigation_memory.nearby_backbone_nodes(
            observation,
            limit=4,
        )
        return {
            "observation_id": observation.trace.observation_id,
            "pose": (
                observation.pose.model_dump(mode="json")
                if observation.pose is not None
                else None
            ),
            "topdown_ref": observation.topdown_ref,
            "map_id": topo_memory.map_id,
            "current_topology_node_id": topo_memory.current_topology_node_id,
            "nearby_nodes": [
                item.model_dump(mode="json") for item in nearby_nodes
            ],
            "topo_memory": topo_memory.model_dump(mode="json"),
            "localization_source": (
                observation.pose.source if observation.pose is not None else "unknown"
            ),
        }

    def record_topology_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist task-scoped regional search evidence after one navigation chunk."""
        with self._pipeline_lock:
            observation_id = str(payload.get("observation_id") or "")
            observation = self._observations.get(observation_id)
            if observation is None or observation.pose is None:
                raise ValueError(f"unknown topology exploration observation: {observation_id}")
            topology = self._topology_model
            if topology is None or not topology.nodes:
                raise ValueError("topology exploration requires a loaded backbone map")
            nearest = min(
                topology.nodes,
                key=lambda node: math.hypot(
                    node.x - observation.pose.x,
                    node.y - observation.pose.y,
                ),
            )
            planned_node_id = str(
                payload.get("planned_topology_node_id") or ""
            ) or None
            explicit_node_id = str(payload.get("topology_node_id") or "") or None
            known_node_ids = {node.node_id for node in topology.nodes}
            attributed_node_id = explicit_node_id or planned_node_id or nearest.node_id
            if attributed_node_id not in known_node_ids:
                raise ValueError(
                    f"unknown attributed topology exploration node: {attributed_node_id}"
                )
            record = self.navigation_memory.record_topology_search(
                topology_node_id=attributed_node_id,
                goal=str(payload.get("goal") or ""),
                status=str(payload.get("status") or "no_target_observed"),
                observation_id=observation_id,
                evidence=str(payload.get("evidence") or ""),
                planned_topology_node_id=planned_node_id,
                observed_topology_node_id=nearest.node_id,
                increment_attempt=bool(payload.get("increment_attempt", True)),
            )
            self._refresh_regional_guidance(observation)
            return {
                "record": record.model_dump(mode="json"),
                "topo_memory": self.navigation_memory.topo_memory_view().model_dump(
                    mode="json"
                ),
                "route_graph": self.navigation_memory.route_graph(),
                "slow_context": self._slow_context(observation).model_dump(mode="json"),
            }

    @staticmethod
    def _execution_telemetry(
        execution: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize Habitat-private execution details at the adapter boundary."""
        reached = bool(execution.get("reached"))
        stalled = bool(execution.get("stalled"))
        progress_m = float(execution.get("distance_m") or 0.0)
        motion_attempted = bool(execution.get("motion_performed")) or (
            int(execution.get("control_steps") or 0) > 0
        )
        reason = str(execution.get("termination_reason") or "")
        if reached:
            state = "reached"
        elif stalled:
            state = "stalled"
        elif reason in {"cancelled", "timeout"}:
            state = "cancelled" if reason == "cancelled" else "failed"
        elif reason in {"blocked", "path_invalid", "no_visible_pixel_goal"}:
            state = "blocked"
        elif motion_attempted:
            state = "progressing"
        else:
            state = "idle"
        return EmbodimentExecutionTelemetry(
            state=state,
            goal_reached=reached,
            motion_attempted=motion_attempted,
            progress_m=progress_m,
            remaining_distance_m=(
                float(execution["remaining_distance_m"])
                if execution.get("remaining_distance_m") is not None
                else None
            ),
            stalled=stalled,
            termination_reason=reason,
            source="habitat-gs-adapter",
        ).model_dump(mode="json")

    def stop_agent_task(self) -> dict[str, Any]:
        runtime = self._active_agent_runtime
        if runtime is not None:
            runtime.cancel("user_stop")
        control = self.adapter.step("stop")
        return {"ok": True, "step": control, "navigation_task": self.agent_task_status()}

    def _on_agent_event(self, task_id: str, event: Event) -> None:
        updates: dict[str, Any] = {
            "agent_state": event.state.value,
            "last_event": event.event_type.value,
            "last_event_at": event.created_at.isoformat(),
            "event_duration_s": event.payload.get("duration_s"),
        }
        if event.event_type in {EventType.PLAN_CREATED, EventType.PLAN_REVISED}:
            plan = event.payload.get("plan") or {}
            updates["plan"] = plan
            updates["plan_revision"] = plan.get("plan_revision", 0)
            updates["active_action"] = plan.get("metadata", {}).get("macro_action")
            updates["main_agent_timing"] = {
                "operation": (
                    "initial_plan"
                    if event.event_type == EventType.PLAN_CREATED
                    else "replan"
                ),
                "duration_s": event.payload.get("duration_s"),
            }
        elif event.event_type == EventType.SKILL_STARTED:
            updates["active_skill"] = event.payload.get("skill_name")
            updates["skill_progress"] = None
        elif event.event_type == EventType.SKILL_PROGRESS:
            updates["skill_progress"] = self._compact_skill_progress(event.payload)
        elif event.event_type == EventType.SKILL_CONTEXT_OPENED:
            updates["skill_runner"] = {
                "status": "running",
                "session": event.payload.get("session"),
            }
        elif event.event_type == EventType.SKILL_CONTEXT_CLOSED:
            outcome = event.payload.get("outcome") or {}
            updates["skill_runner"] = {
                "status": "returned",
                "session": event.payload.get("session"),
                "outcome": outcome,
            }
            updates["step"] = int(self.agent_task_status().get("step", 0)) + 1
        elif event.event_type == EventType.SKILL_RESULT:
            result = event.payload.get("result") or {}
            distance_m = self._skill_distance_m(result)
            if distance_m > 0.0:
                current_distance = float(
                    self.agent_task_status().get("distance_m") or 0.0
                )
                updates["distance_m"] = round(current_distance + distance_m, 3)
            updates["execution_summary"] = self._skill_execution_summary(result)
        elif event.event_type in {
            EventType.RUNTIME_VERIFIED,
            EventType.SKILL_VERIFIED,
            EventType.FINISH_VERIFIED,
        }:
            updates["verification"] = event.payload.get("verification")
            updates["verification_timing"] = {
                "event": event.event_type.value,
                "duration_s": event.payload.get("duration_s"),
            }
            if event.event_type == EventType.SKILL_VERIFIED:
                self._reconcile_verified_topology_target(event)
        self._update_agent_task(task_id, **updates)

    @staticmethod
    def _compact_skill_progress(payload: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "summary",
            "stage",
            "operation",
            "status",
            "completed_stage",
            "duration_s",
            "slow_inference_latency_s",
            "slow_view_binding_latency_s",
            "visual_grounding_latency_s",
            "decision_pipeline_latency_s",
            "target_query",
            "affordance_query",
            "executed_role",
            "topology_node_id",
            "goal_reached",
            "progress_m",
            "observation_id",
        )
        return {key: payload.get(key) for key in fields if payload.get(key) is not None}

    @staticmethod
    def _skill_execution_summary(result: dict[str, Any]) -> dict[str, Any]:
        outputs = result.get("outputs") or {}
        generated = outputs.get("generated_point") or {}
        completion = outputs.get("semantic_completion") or {}
        telemetry = outputs.get("execution_telemetry") or {}
        slow_output = outputs.get("slow_brain_output") or {}
        semantic_decision = outputs.get("slow_brain_semantic_decision") or {}
        grounding = outputs.get("slow_brain_grounding") or {}
        route_plan = outputs.get("semantic_route_plan") or {}
        guidance = route_plan.get("regional_guidance") or {}

        role = outputs.get("anchor_role") or generated.get("anchor_role")
        label = generated.get("anchor")
        view = generated.get("view")
        confidence = generated.get("confidence")
        if role == "target":
            label = completion.get("target_label") or label
            view = completion.get("target_view") or view
            confidence = completion.get("target_confidence") or confidence
        if not role and result.get("stop_reason") == "topology_point_goal_reached":
            role = "point_goal"
            label = (
                guidance.get("target_topology_node_id")
                or guidance.get("next_topology_node_id")
            )

        query = semantic_decision.get(f"{role}_query") or {}
        grounded = grounding.get(f"{role}_grounding") or {}
        cot = slow_output.get("cot") or semantic_decision.get("cot")
        if isinstance(cot, dict):
            reasoning = "; ".join(str(value) for value in cot.values() if value)
        else:
            reasoning = str(cot or "")

        return {
            "skill_name": result.get("skill_name"),
            "role": role,
            "label": label,
            "view": view,
            "confidence": confidence,
            "observed_evidence": grounded.get("evidence_summary"),
            "reasoning": reasoning or None,
            "selection_reason": query.get("spatial_relation"),
            "state": telemetry.get("state") or result.get("status"),
            "goal_reached": telemetry.get("goal_reached"),
            "progress_m": telemetry.get("progress_m"),
            "termination_reason": (
                telemetry.get("termination_reason") or result.get("stop_reason")
            ),
            "source_observation_id": generated.get("source_observation_id"),
            "post_observation_id": completion.get("post_observation_id")
            or outputs.get("next_observation_id"),
            "slow_brain_latency_s": outputs.get("slow_brain_latency_s")
            or outputs.get("slow_inference_latency_s"),
            "slow_view_binding_latency_s": outputs.get(
                "slow_view_binding_latency_s"
            ),
            "visual_grounding_latency_s": outputs.get(
                "visual_grounding_latency_s"
            ),
            "decision_pipeline_latency_s": outputs.get(
                "decision_pipeline_latency_s"
            ),
            "fast_system_latency_s": outputs.get("fast_system_latency_s"),
        }

    @staticmethod
    def _skill_distance_m(result: dict[str, Any]) -> float:
        outputs = result.get("outputs") or {}
        candidates = [
            outputs.get("distance_m"),
            (outputs.get("execution_summary") or {}).get("distance_m"),
            (outputs.get("result") or {}).get("distance_m"),
            ((outputs.get("execution_report") or {}).get("distance_m")),
        ]
        for candidate in candidates:
            if isinstance(candidate, (int, float)) and candidate > 0.0:
                return float(candidate)
        return 0.0

    def _reconcile_verified_topology_target(self, event: Event) -> None:
        verification = event.payload.get("verification") or {}
        completion = event.payload.get("semantic_completion") or {}
        search = event.payload.get("topology_goal_search") or {}
        if (
            completion.get("task_type") != "object_goal"
            or not bool(completion.get("satisfied"))
            or search.get("status") != "target_observed"
        ):
            return
        observation_id = str(search.get("last_observation_id") or "")
        topology_node_id = str(search.get("topology_node_id") or "")
        if not observation_id or not topology_node_id:
            return
        verified = verification.get("status") == "success"
        try:
            self.record_topology_search(
                {
                    "observation_id": observation_id,
                    "topology_node_id": topology_node_id,
                    "planned_topology_node_id": search.get(
                        "planned_topology_node_id"
                    ),
                    "goal": search.get("goal"),
                    "status": (
                        "target_confirmed" if verified else "no_target_observed"
                    ),
                    "evidence": (
                        "Independent Skill Verifier confirmed the current object_goal after "
                        "target-role execution."
                        if verified
                        else "Independent Skill Verifier rejected the apparent target after "
                        "target-role execution."
                    ),
                    "increment_attempt": False,
                }
            )
        except ValueError as exc:
            self._update_agent_task(
                event.trace.task_id,
                topology_memory_error=str(exc),
            )

    def _complete_agent_task(self, task_id: str, result: RuntimeResult) -> None:
        status = {
            RuntimeState.DONE: "completed",
            RuntimeState.CANCELLED: "cancelled",
            RuntimeState.FAILED: "failed",
        }.get(result.state, "failed")
        finish = (
            result.finish_verification.model_dump(mode="json")
            if result.finish_verification is not None
            else None
        )
        compact_results = [
            {
                "skill_name": item.skill_name,
                "status": item.status.value,
                "summary": item.compressed_history or item.summary,
                "stop_reason": item.stop_reason,
                "evidence": [evidence.model_dump(mode="json") for evidence in item.evidence],
            }
            for item in result.skill_results
        ]
        terminal_reason = result.state.value
        if result.state == RuntimeState.FAILED:
            failed_result = next(
                (item for item in reversed(result.skill_results) if item.error),
                None,
            )
            if failed_result is not None:
                terminal_reason = failed_result.error or failed_result.summary
            elif result.finish_verification is not None:
                terminal_reason = result.finish_verification.summary
        elif result.finish_verification is not None:
            terminal_reason = result.finish_verification.summary
        self._update_agent_task(
            task_id,
            status=status,
            agent_state=result.state.value,
            reason=terminal_reason,
            plan=result.plan.model_dump(mode="json") if result.plan else None,
            skill_results=compact_results,
            main_agent_results=[item.model_dump(mode="json") for item in result.main_agent_results],
            finish_verification=finish,
            finished_at=utc_now().isoformat(),
        )

    def _update_agent_task(self, task_id: str, **updates: Any) -> None:
        with self._agent_task_lock:
            if self._agent_task_state.get("task_id") != task_id:
                return
            self._agent_task_state.update(updates)

    def observe(self) -> dict[str, Any]:
        observation = self._capture_observation()
        return {"ok": True, "observation": observation.model_dump(mode="json")}

    def begin_navigation(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = payload or {}
        global_goal = str(raw.get("global_goal") or "").strip()
        decomposition = None
        if raw.get("destination_query") and raw.get("object_goal"):
            decomposition = SemanticTaskDecompositionInput.model_validate(raw)
            global_goal = decomposition.global_goal
        explicit_task = bool(global_goal)
        if not global_goal:
            global_goal = "Active navigation task"
        with self._pipeline_lock:
            repeated_active_task = (
                explicit_task
                and decomposition is None
                and self._semantic_route_plan is not None
                and self._semantic_route_plan.global_goal.casefold() == global_goal.casefold()
            )
            replacing_active_task = (
                explicit_task
                and self._semantic_route_plan is not None
                and not repeated_active_task
            )
            if replacing_active_task:
                self.task_id = self._new_task_id()
                self._scene_memory = TaskSceneMemory(self.task_id)
                self.navigation_memory.start_task(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                )
                self._topology_loaded_for_task = False
                self._topology_state = None
                self._semantic_route_plan = None
                self._active_target_evidence = None
                self._observations.clear()
                self._latest_observation = None
                self._latest_pipeline = None
            self._pending_global_goal = global_goal
            self._planner_required = (
                explicit_task and decomposition is None and not repeated_active_task
            )
            topology = self._load_topology_for_task()
            observation = self._capture_observation()
            semantic_route_plan = (
                self._create_semantic_route_plan(decomposition, observation)
                if decomposition is not None
                else self._semantic_route_plan
            )
            result = {
                "ok": True,
                "observation": observation.model_dump(mode="json"),
                "memory_view": self.navigation_memory.view().model_dump(mode="json"),
                "topology": topology,
                "route_graph": self.navigation_memory.route_graph(),
                "semantic_route_plan": (
                    semantic_route_plan.model_dump(mode="json")
                    if semantic_route_plan is not None
                    else None
                ),
                "topo_memory": self.navigation_memory.topo_memory_view().model_dump(
                    mode="json"
                ),
                "slow_context": self._slow_context(observation).model_dump(mode="json"),
                "planner_required": self._planner_required,
                "next_call_contract": self._next_call_contract(
                    observation.trace.observation_id
                ),
            }
            if not repeated_active_task:
                self._latest_pipeline = {
                    "ok": True,
                    "status": "ready",
                    "profile": "hierarchical",
                    "phase": "plan" if self._planner_required else "execute",
                    "instruction": global_goal,
                    "finish_required": False,
                    "next_call_contract": result["next_call_contract"],
                }
            return result

    def prepare_agentic_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Install a Main-Agent plan before an isolated navigation Skill runs."""
        observation = self._require_observation(payload)
        planner_output = AgenticPlannerOutput.model_validate(payload.get("planner_output"))
        with self._pipeline_lock:
            latest_id = ((self._latest_observation or {}).get("trace") or {}).get(
                "observation_id"
            )
            if observation.trace.observation_id != latest_id:
                raise ValueError(
                    f"stale observation_id: {observation.trace.observation_id}; latest is {latest_id}"
                )
            planner_output = self._complete_planner_output(planner_output)
            route_plan = self._create_semantic_route_plan(planner_output, observation)
            return {
                "ok": True,
                "status": "skill_ready",
                "observation": observation.model_dump(mode="json"),
                "semantic_route_plan": route_plan.model_dump(mode="json"),
                "slow_context": self._slow_context(observation).model_dump(mode="json"),
                "active_target_evidence": self._active_target_evidence,
            }

    def recall_visual_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = NavigationVisualRecallInput.model_validate(payload)
        with self._pipeline_lock:
            memories = self.navigation_memory.recall_visuals(
                scope=request.scope,
                location_id=request.location_id,
                query=request.query,
                limit=request.limit,
            )
            return {
                "ok": True,
                "task_id": self.task_id,
                "scope": request.scope,
                "location_id": request.location_id,
                "query": request.query,
                "count": len(memories),
                "archive_count": len(self.navigation_memory.visual_memories()),
                "memories": [memory.model_dump(mode="json") for memory in memories],
            }

    def _create_semantic_route_plan(
        self,
        request: SemanticTaskDecompositionInput | AgenticPlannerOutput,
        observation: Observation,
    ) -> SemanticRoutePlan:
        if isinstance(request, AgenticPlannerOutput) and self._pending_global_goal:
            request = request.model_copy(
                update={"global_instruction": self._pending_global_goal}
            )
        plan_revision = (
            self._semantic_route_plan.agentic_plan.revision + 1
            if self._semantic_route_plan is not None
            else 0
        )
        global_goal = (
            request.global_instruction
            if isinstance(request, AgenticPlannerOutput)
            else request.global_goal
        )
        decomposition = SemanticTaskDecompositionInput(
            global_goal=global_goal,
            destination_query=request.destination_query,
            object_goal=request.object_goal,
            node_local_radius_m=request.node_local_radius_m,
            anchor_max_depth_m=request.anchor_max_depth_m,
        )
        documents = self.navigation_memory.semantic_node_documents()
        ranked_selection = None
        if (
            not isinstance(request, AgenticPlannerOutput)
            and self._semantic_node_ranker is not None
            and documents
        ):
            try:
                ranked_selection = self._semantic_node_ranker.select(
                    decomposition, documents
                )
            except Exception:
                ranked_selection = None
        destination_selection = (
            request.node_selection
            if isinstance(request, AgenticPlannerOutput)
            and request.node_selection is not None
            and request.node_selection.confidence
            >= self.semantic_node_selection_threshold
            else ranked_selection
            if ranked_selection is not None
            and ranked_selection.confidence >= self.semantic_node_selection_threshold
            else None
        )
        destination_topology_node_id = (
            destination_selection.selected_topology_node_id
            if destination_selection is not None
            else None
        )
        if destination_topology_node_id is not None:
            destination = self.navigation_memory.get_topology_node(
                destination_topology_node_id
            )
            if destination is None:
                raise ValueError(
                    "Model selected unknown destination topology node: "
                    f"{destination_topology_node_id}"
                )
        if isinstance(request, AgenticPlannerOutput):
            request = self._complete_planner_output(
                request,
                destination_topology_node_id=destination_topology_node_id,
                destination_selection=destination_selection,
            )
        planner_subtasks = (
            request.subtasks if isinstance(request, AgenticPlannerOutput) else []
        )
        requested_topology_node_id = (
            planner_subtasks[0].topology_node_id if planner_subtasks else None
        )
        matches = (
            self.navigation_memory.search_semantic_nodes(
                request.destination_query,
                limit=5,
            )
            if not isinstance(request, AgenticPlannerOutput)
            and self._semantic_node_ranker is None
            else []
        )
        selected_topology_node_id: str | None = None
        selected_semantics: NodeSemantics | None = None
        match_score: float | None = None
        matched_terms: list[str] = []
        if requested_topology_node_id is not None:
            selected_place = self.navigation_memory.get_topology_node(
                requested_topology_node_id
            )
            if selected_place is None:
                raise ValueError(
                    f"Agentic Planner selected unknown topology node: "
                    f"{requested_topology_node_id}"
                )
            selected_topology_node_id = requested_topology_node_id
            selected_semantics = selected_place.node_semantics
            match_score = 1.0
            matched_terms = [request.destination_query]
            phase = "local_anchor_search"
            reason = (
                f"Agentic Planner selected {requested_topology_node_id} from the "
                "four-layer Topo-Memory."
            )
        if selected_topology_node_id is not None:
            pass
        elif destination_selection is not None:
            selected_topology_node_id = destination_selection.selected_topology_node_id
            selected_place = self.navigation_memory.get_topology_node(
                selected_topology_node_id
            )
            selected_semantics = (
                selected_place.node_semantics if selected_place is not None else None
            )
            match_score = destination_selection.confidence
            matched_terms = list(destination_selection.matched_concepts)
            phase = "local_anchor_search"
            reason = (
                f"Semantic task ranker selected regional context {selected_topology_node_id}: "
                f"{destination_selection.selection_reason}"
            )[:240]
        elif matches:
            selected = matches[0]
            selected_topology_node_id = str(selected["topology_node_id"])
            selected_semantics = NodeSemantics.model_validate(selected["node_semantics"])
            match_score = float(selected["score"])
            matched_terms = [str(item) for item in selected["matched_terms"]]
            phase = "local_anchor_search"
            reason = (
                f"Topo-Memory matched regional context {request.destination_query!r} to "
                f"{selected_topology_node_id}."
            )
        elif not isinstance(request, AgenticPlannerOutput):
            nearby = self.navigation_memory.nearby_backbone_nodes(observation, limit=1)
            if nearby:
                selected = nearby[0]
                selected_topology_node_id = selected.topology_node_id
                place = self.navigation_memory.get_topology_node(
                    selected.topology_node_id or ""
                )
                selected_semantics = place.node_semantics if place is not None else None
                phase = "local_anchor_search"
                reason = (
                    f"No semantic node matched {request.destination_query!r}; use nearest fixed node "
                    f"{selected.topology_node_id} only as regional context for visual exploration."
                )
            else:
                phase = "explore_nodes"
                reason = (
                    f"No semantic node matched {request.destination_query!r} and no backbone node is available."
                )
        else:
            phase = "explore_nodes"
            reason = (
                "The Main Agent delegated no topology node. Runtime preserved that semantic "
                "decision and left localization or visual exploration to the selected Skill."
            )
        plan = SemanticRoutePlan(
            plan_id=f"semantic-route-{uuid.uuid4().hex[:12]}",
            global_goal=global_goal,
            destination_query=request.destination_query,
            object_goal=request.object_goal,
            phase=phase,
            selected_topology_node_id=selected_topology_node_id,
            destination_topology_node_id=destination_topology_node_id,
            destination_selection=destination_selection,
            selected_node_semantics=selected_semantics,
            match_score=match_score,
            matched_terms=matched_terms,
            selection_reason=reason,
            node_local_radius_m=request.node_local_radius_m,
            anchor_max_depth_m=request.anchor_max_depth_m,
            agentic_plan=self._create_agentic_plan(
                request,
                selected_topology_node_id=selected_topology_node_id,
                revision=plan_revision,
            ),
        )
        self._pending_global_goal = global_goal
        self._semantic_route_plan = plan
        self._active_target_evidence = None
        self._planner_required = False
        self._refresh_regional_guidance(observation)
        return self._semantic_route_plan or plan

    @staticmethod
    def _complete_planner_output(
        request: AgenticPlannerOutput,
        *,
        destination_topology_node_id: str | None = None,
        destination_selection: SemanticNodeSelection | None = None,
    ) -> AgenticPlannerOutput:
        proposals = list(request.subtasks)
        fallback_topology_node_id = next(
            (
                subtask.topology_node_id
                for subtask in reversed(proposals)
                if subtask.topology_node_id is not None
            ),
            None,
        )
        target_topology_node_id = (
            destination_topology_node_id or fallback_topology_node_id
        )
        reaching_indexes = [
            index for index, subtask in enumerate(proposals) if subtask.stage == "reaching"
        ]
        if not reaching_indexes:
            proposals.insert(
                len(proposals),
                AgenticPlannerSubtask(
                    stage="reaching",
                    task_type="object_goal",
                    instruction=(
                        f"Locate and verify {request.object_goal} within or functionally "
                        f"associated with {request.destination_query}."
                    ),
                    target=request.object_goal,
                    topology_node_id=target_topology_node_id,
                    success_condition=(
                        f"Current visual evidence confirms {request.object_goal} in the "
                        f"{request.destination_query} context."
                    ),
                ),
            )
        elif target_topology_node_id is not None:
            final_reaching_index = reaching_indexes[-1]
            proposals[final_reaching_index] = proposals[final_reaching_index].model_copy(
                update={
                    "instruction": (
                        f"Locate and verify {request.object_goal} within or functionally "
                        f"associated with {request.destination_query}."
                    ),
                    "target": request.object_goal,
                    "topology_node_id": target_topology_node_id,
                    "success_condition": (
                        f"Current visual evidence confirms {request.object_goal} in the "
                        f"{request.destination_query} context."
                    ),
                }
            )

        if destination_topology_node_id is not None:
            destination_subtask = AgenticPlannerSubtask(
                stage="approaching",
                task_type="point_goal",
                instruction=(
                    f"Approach {destination_topology_node_id}, the model-selected region "
                    f"for {request.object_goal} in or associated with {request.destination_query}."
                ),
                target=request.destination_query,
                topology_node_id=destination_topology_node_id,
                success_condition=(
                    f"Runtime confirms arrival at {destination_topology_node_id}."
                ),
            )
            normalized_proposals: list[AgenticPlannerSubtask] = []
            destination_added = False
            for subtask in proposals:
                if subtask.stage != "approaching":
                    normalized_proposals.append(subtask)
                    continue
                if subtask.topology_node_id is None:
                    continue
                if subtask.topology_node_id == destination_topology_node_id:
                    if not destination_added:
                        normalized_proposals.append(destination_subtask)
                        destination_added = True
                    continue
                normalized_proposals.append(subtask)
            if not destination_added:
                final_reaching_index = next(
                    (
                        index
                        for index in range(len(normalized_proposals) - 1, -1, -1)
                        if normalized_proposals[index].stage == "reaching"
                    ),
                    len(normalized_proposals),
                )
                if len(normalized_proposals) < 8:
                    normalized_proposals.insert(
                        final_reaching_index, destination_subtask
                    )
                else:
                    approaching_indexes = [
                        index
                        for index, subtask in enumerate(normalized_proposals)
                        if subtask.stage == "approaching"
                    ]
                    if approaching_indexes:
                        normalized_proposals[approaching_indexes[-1]] = destination_subtask
            proposals = normalized_proposals

        update: dict[str, object] = {"subtasks": proposals}
        if destination_selection is not None:
            update["node_selection"] = destination_selection
        return request.model_copy(update=update)

    @staticmethod
    def _create_agentic_plan(
        request: SemanticTaskDecompositionInput | AgenticPlannerOutput,
        *,
        selected_topology_node_id: str | None,
        revision: int = 0,
    ) -> AgenticPlan:
        if isinstance(request, AgenticPlannerOutput):
            subtasks = [
                AgenticSubtask(
                    subtask_id=f"subtask-{index:03d}",
                    stage=proposal.stage,
                    task_type=proposal.task_type,
                    instruction=proposal.instruction,
                    target=proposal.target,
                    topology_node_id=proposal.topology_node_id,
                    success_condition=proposal.success_condition,
                    status="active" if index == 1 else "pending",
                )
                for index, proposal in enumerate(request.subtasks, start=1)
            ]
            global_instruction = request.global_instruction
        else:
            subtasks = [
                AgenticSubtask(
                    subtask_id="subtask-001",
                    stage="approaching",
                    task_type="point_goal",
                    instruction=f"Approach the {request.destination_query} region.",
                    target=request.destination_query,
                    topology_node_id=selected_topology_node_id,
                    success_condition=(
                        f"Reach the connected topology region associated with "
                        f"{request.destination_query}."
                    ),
                    status="active",
                ),
                AgenticSubtask(
                    subtask_id="subtask-002",
                    stage="reaching",
                    task_type="object_goal",
                    instruction=(
                        f"Locate and reach {request.object_goal} within or functionally "
                        f"associated with {request.destination_query}."
                    ),
                    target=request.object_goal,
                    topology_node_id=selected_topology_node_id,
                    success_condition=(
                        f"Current visual evidence confirms {request.object_goal} in the "
                        f"{request.destination_query} context."
                    ),
                ),
            ]
            global_instruction = request.global_goal
        return AgenticPlan(
            plan_id=f"agentic-plan-{uuid.uuid4().hex[:12]}",
            global_instruction=global_instruction,
            subtasks=subtasks,
            revision=revision,
        )

    def _apply_subtask_reflection(
        self,
        reflection: SubtaskReflection | None,
        observation: Observation,
    ) -> None:
        if reflection is None or self._semantic_route_plan is None:
            return
        plan = self._semantic_route_plan.agentic_plan
        active = plan.subtasks[plan.active_subtask_index]
        if reflection.subtask_id != active.subtask_id:
            raise ValueError(
                f"self_reflection references {reflection.subtask_id}; active subtask is "
                f"{active.subtask_id}"
            )
        subtasks = list(plan.subtasks)
        next_index = plan.active_subtask_index
        if reflection.completed:
            subtasks[next_index] = active.model_copy(update={"status": "completed"})
            if next_index + 1 < len(subtasks):
                next_index += 1
                subtasks[next_index] = subtasks[next_index].model_copy(
                    update={"status": "active"}
                )
        elif reflection.should_replan:
            # The planner may change the route or replace the plan in the same Slow
            # invocation. Keep the typed subtask active until that replacement exists.
            subtasks[next_index] = active.model_copy(update={"status": "active"})
        elif reflection.outcome == "failed":
            subtasks[next_index] = active.model_copy(update={"status": "failed"})
        updated_plan = plan.model_copy(
            update={
                "subtasks": subtasks,
                "active_subtask_index": next_index,
                "revision": plan.revision + 1,
            }
        )
        self._semantic_route_plan = self._semantic_route_plan.model_copy(
            update={"agentic_plan": updated_plan}
        )
        self._sync_route_to_active_subtask(observation)

    def _preflight_advance_transition(
        self,
        request: AdvanceNavigationInput,
        observation: Observation,
        *,
        validate_task_type: bool,
    ) -> bool:
        """Validate one Slow transition before mutating any Runtime state."""
        route_plan = self._semantic_route_plan
        reflection = request.self_reflection
        active = None
        next_index = None
        final_completion = False
        if reflection is not None:
            if route_plan is None:
                raise ValueError("self_reflection requires an existing Agentic Plan")
            plan = route_plan.agentic_plan
            active = plan.subtasks[plan.active_subtask_index]
            if reflection.subtask_id != active.subtask_id:
                raise ValueError(
                    f"self_reflection references {reflection.subtask_id}; active subtask is "
                    f"{active.subtask_id}"
                )
            if reflection.outcome == "failed" and not reflection.should_replan:
                raise ValueError("a failed subtask must request replanning")
            if reflection.completed:
                self._validate_subtask_completion(active, observation)
                final_completion = plan.active_subtask_index + 1 == len(plan.subtasks)
                if not final_completion:
                    next_index = plan.active_subtask_index + 1

        if request.planner_output is not None:
            if route_plan is not None and not (
                reflection is not None and reflection.should_replan
            ):
                raise ValueError(
                    "planner_output is accepted only for the initial plan or after "
                    "Self-Reflector requests replanning"
                )
            expected_task_type = request.planner_output.subtasks[0].task_type
        elif route_plan is not None and not final_completion:
            plan = route_plan.agentic_plan
            expected_index = next_index if next_index is not None else plan.active_subtask_index
            expected_task_type = plan.subtasks[expected_index].task_type
        else:
            expected_task_type = None

        if (
            validate_task_type
            and expected_task_type is not None
            and request.navigation_task_type != expected_task_type
        ):
            raise ValueError(
                f"navigation_task_type={request.navigation_task_type} does not match "
                f"the next active task_type={expected_task_type}"
            )
        return final_completion

    def _normalize_point_goal_reflection(
        self,
        request: AdvanceNavigationInput,
        observation: Observation,
    ) -> tuple[AdvanceNavigationInput, dict[str, object] | None]:
        reflection = request.self_reflection
        route_plan = self._semantic_route_plan
        if reflection is None or not reflection.completed or route_plan is None:
            return request, None
        plan = route_plan.agentic_plan
        active = plan.subtasks[plan.active_subtask_index]
        if active.task_type != "point_goal" or active.topology_node_id is None:
            return request, None
        try:
            self._validate_subtask_completion(active, observation)
        except ValueError as exc:
            if not str(exc).startswith("cannot complete point-goal"):
                raise
            normalized_reflection = reflection.model_copy(
                update={
                    "completed": False,
                    "outcome": "partial",
                    "feedback": (
                        f"Runtime has not confirmed arrival at {active.topology_node_id}; "
                        "keep this Point-Goal active and follow the next connected node."
                    ),
                }
            )
            return request.model_copy(
                update={"self_reflection": normalized_reflection}
            ), {
                "corrected_self_reflection": True,
                "active_subtask_id": active.subtask_id,
                "reason": str(exc),
            }
        return request, None

    def _normalize_transition_task_type(
        self,
        request: AdvanceNavigationInput,
    ) -> tuple[AdvanceNavigationInput, dict[str, object] | None]:
        expected_task_type = None
        if request.planner_output is not None:
            expected_task_type = request.planner_output.subtasks[0].task_type
        elif self._semantic_route_plan is not None:
            plan = self._semantic_route_plan.agentic_plan
            expected_index = plan.active_subtask_index
            reflection = request.self_reflection
            if reflection is not None and reflection.completed:
                if expected_index + 1 >= len(plan.subtasks):
                    return request, None
                expected_index += 1
            expected_task_type = plan.subtasks[expected_index].task_type
        if (
            expected_task_type is None
            or request.navigation_task_type == expected_task_type
        ):
            return request, None

        if expected_task_type == "object_goal":
            cot = ObjectGoalCoT(
                target_localization_and_key_context=(
                    f"Use the current observation and retained target evidence for "
                    f"{request.action.target}."
                )[:240],
                path_planning=request.route_hypothesis.statement[:240],
            )
        elif expected_task_type == "instruction_following":
            cot = InstructionFollowingCoT(
                finished_sub_instruction=request.previous_assessment.evidence[:160],
                ongoing_sub_instruction=(
                    f"Execute {request.action.action_type}: {request.action.target}."
                )[:160],
            )
        else:
            cot = None
        original_task_type = request.navigation_task_type
        return request.model_copy(
            update={"navigation_task_type": expected_task_type, "cot": cot}
        ), {
            "corrected_navigation_task_type": True,
            "from": original_task_type,
            "to": expected_task_type,
            "reason": "Agentic Plan active-subtask contract is authoritative.",
        }

    def _validate_subtask_completion(
        self,
        active: AgenticSubtask,
        observation: Observation,
    ) -> None:
        if active.task_type in {"object_goal", "poi_goal"}:
            # Object identity and task satisfaction are semantic judgments. The
            # model Self-Reflector proposes completion and the independent Finish
            # Verifier validates the latest tri-view before global termination.
            return
        if active.task_type != "point_goal" or active.topology_node_id is None:
            return
        route_plan = self._semantic_route_plan
        topology = self._topology_model
        if route_plan is None or topology is None or observation.pose is None:
            return
        target = next(
            (node for node in topology.nodes if node.node_id == active.topology_node_id),
            None,
        )
        if target is None:
            raise ValueError(
                f"cannot verify point-goal completion for unknown node {active.topology_node_id}"
            )
        distance = math.hypot(
            target.x - observation.pose.x,
            target.y - observation.pose.y,
        )
        arrival_radius = self._node_arrival_radius(route_plan, target)
        if distance > arrival_radius:
            raise ValueError(
                f"cannot complete point-goal {active.subtask_id}: robot is {distance:.2f} m "
                f"from {active.topology_node_id}, outside the {arrival_radius:.2f} m arrival radius"
            )

    def _sync_route_to_active_subtask(self, observation: Observation) -> None:
        plan = self._semantic_route_plan
        if plan is None:
            return
        agentic_plan = plan.agentic_plan
        active = agentic_plan.subtasks[agentic_plan.active_subtask_index]
        if active.status != "active" or active.topology_node_id is None:
            self._refresh_regional_guidance(observation)
            return
        if active.topology_node_id != plan.selected_topology_node_id:
            target = self.navigation_memory.get_topology_node(active.topology_node_id)
            if target is None:
                raise ValueError(
                    f"active subtask references unknown topology node: {active.topology_node_id}"
                )
            self._activate_semantic_node(
                plan,
                target,
                reason=(
                    f"Active {active.subtask_id} selected {active.topology_node_id} "
                    "from the Agentic Plan."
                ),
            )
        self._refresh_regional_guidance(observation)

    def _await_finish_verification(
        self,
        request: AdvanceNavigationInput,
        observation: Observation,
    ) -> dict[str, Any]:
        assert self._semantic_route_plan is not None
        self._semantic_route_plan = self._semantic_route_plan.model_copy(
            update={"phase": "verify_goal"}
        )
        state = {
            "ok": True,
            "status": "awaiting_finish_verification",
            "profile": "hierarchical",
            "phase": "verify_goal",
            "finish_required": True,
            "decision": None,
            "result": {},
            "self_reflection": request.self_reflection.model_dump(mode="json"),
            "semantic_route_plan": self._semantic_route_plan.model_dump(mode="json"),
            "agentic_plan": self._semantic_route_plan.agentic_plan.model_dump(mode="json"),
            "slow_context": self._slow_context(observation).model_dump(mode="json"),
            "post_observation": observation.model_dump(mode="json"),
            "next_observation_id": observation.trace.observation_id,
            "active_target_evidence": (
                self._active_target_evidence.model_dump(mode="json")
                if self._active_target_evidence is not None
                else None
            ),
            "next_call_contract": self._next_call_contract(
                observation.trace.observation_id,
                finish_required=True,
            ),
        }
        self._latest_pipeline = dict(state)
        return state

    def _validate_active_task_type(self, request: AdvanceNavigationInput) -> None:
        if self._semantic_route_plan is None:
            raise ValueError(
                "Agentic Plan is missing; the first Slow decision must include planner_output"
            )
        plan = self._semantic_route_plan.agentic_plan
        active = plan.subtasks[plan.active_subtask_index]
        if active.status != "active":
            raise ValueError(
                "Agentic Plan has no active subtask; request independent finish verification "
                "or submit a replanned planner_output"
            )
        if request.navigation_task_type != active.task_type:
            raise ValueError(
                f"navigation_task_type={request.navigation_task_type} does not match "
                f"active {active.subtask_id} task_type={active.task_type}"
            )

    def _slow_context(self, observation: Observation) -> ABotSlowContext:
        visuals = [
            visual
            for visual in self.navigation_memory.visual_memories()
            if visual.kind == "trajectory_keyframe" and visual.view == "front"
        ]
        latest_subtask_id = visuals[-1].subtask_id if visuals else None
        historical_ids: list[str] = []
        for visual in visuals:
            if visual.subtask_id != latest_subtask_id:
                continue
            if visual.observation_id not in historical_ids:
                historical_ids.append(visual.observation_id)
        task_specification = (
            self._semantic_route_plan.global_goal
            if self._semantic_route_plan is not None
            else self._pending_global_goal
            or self.navigation_memory.view().global_goal
            or "Active navigation task"
        )
        agentic_plan = (
            self._semantic_route_plan.agentic_plan
            if self._semantic_route_plan is not None
            else None
        )
        active_subtask = None
        if agentic_plan is not None:
            candidate = agentic_plan.subtasks[agentic_plan.active_subtask_index]
            if candidate.status == "active":
                active_subtask = candidate
        previous_decision = self.navigation_memory.last_slow_output
        return ABotSlowContext(
            current_observation_id=observation.trace.observation_id,
            historical_front_observation_ids=historical_ids[-4:],
            task_specification=task_specification,
            previous_decision=(
                ABotSlowOutput.model_validate(previous_decision)
                if previous_decision is not None
                else None
            ),
            active_target_evidence=self._active_target_evidence,
            topology_context=ABotPlannerContext(
                topo_memory=self.navigation_memory.topo_memory_view(),
                agentic_plan=agentic_plan,
                active_subtask=active_subtask,
            ),
        )

    def _next_call_contract(
        self,
        observation_id: str | None,
        *,
        finish_required: bool = False,
    ) -> dict[str, object]:
        if finish_required:
            return {
                "next_tool": "finish_navigation_task",
                "observation_id": observation_id,
                "planner_output": "forbidden",
                "self_reflection": "already_accepted",
                "self_reflection_subtask_id": None,
                "navigation_task_type": None,
                "active_target_evidence": self._active_target_evidence is not None,
            }
        active = None
        if self._semantic_route_plan is not None:
            plan = self._semantic_route_plan.agentic_plan
            candidate = plan.subtasks[plan.active_subtask_index]
            if candidate.status == "active":
                active = candidate
        planner_state = (
            "required"
            if self._planner_required or self._semantic_route_plan is None
            else "only_when_self_reflection.should_replan"
        )
        reflection_required = self.navigation_memory.last_slow_output is not None
        reflection_contract = "required" if reflection_required else "null"
        if (
            reflection_required
            and active is not None
            and active.task_type in {"object_goal", "poi_goal"}
            and self._active_target_evidence is not None
        ):
            reflection_contract = (
                "required_complete_or_contradict_target_evidence"
            )
        return {
            "next_tool": "advance_navigation" if observation_id else "begin_navigation_task",
            "observation_id": observation_id,
            "planner_output": planner_state if observation_id else "required",
            "self_reflection": reflection_contract,
            "self_reflection_subtask_id": active.subtask_id if reflection_required and active else None,
            "navigation_task_type": active.task_type if active is not None else None,
            "active_target_evidence": self._active_target_evidence is not None,
            "prose_completion_allowed": False,
        }

    def _refresh_regional_guidance(self, observation: Observation) -> None:
        plan = self._semantic_route_plan
        if plan is None:
            return
        guidance = self._regional_anchor_guidance(plan, observation)
        update: dict[str, object] = {"regional_guidance": guidance}
        if plan.phase != "verify_goal":
            if guidance.mode == "route_to_region":
                update["phase"] = "approach_node"
            elif guidance.mode == "search_within_region":
                update["phase"] = "local_anchor_search"
        self._semantic_route_plan = plan.model_copy(update=update)

    def _regional_anchor_guidance(
        self,
        plan: SemanticRoutePlan,
        observation: Observation,
    ) -> RegionalAnchorGuidance:
        target_id = plan.selected_topology_node_id
        topology = self._topology_model
        if topology is None or observation.pose is None or target_id is None:
            return RegionalAnchorGuidance(
                mode="unavailable",
                target_topology_node_id=target_id,
                reason="Topology, target region, or robot pose is unavailable.",
            )
        nodes = {node.node_id: node for node in topology.nodes}
        target = nodes.get(target_id)
        if target is None or not nodes:
            return RegionalAnchorGuidance(
                mode="unavailable",
                target_topology_node_id=target_id,
                reason="The selected regional node is absent from the frozen topology.",
            )
        target_distance = math.hypot(
            target.x - observation.pose.x,
            target.y - observation.pose.y,
        )
        target_arrival_radius = self._node_arrival_radius(plan, target)
        previous = plan.regional_guidance
        reuse_route = (
            previous is not None
            and previous.target_topology_node_id == target_id
            and bool(previous.route_node_ids)
            and previous.route_node_ids[-1] == target_id
            and all(node_id in nodes for node_id in previous.route_node_ids)
            and previous.route_progress_index < len(previous.route_node_ids)
        )
        if reuse_route:
            route = list(previous.route_node_ids)
            progress_index = previous.route_progress_index
        else:
            nearest = min(
                nodes.values(),
                key=lambda node: math.hypot(
                    node.x - observation.pose.x,
                    node.y - observation.pose.y,
                ),
            )
            route = self._shortest_topology_route(topology, nearest.node_id, target_id)
            progress_index = 0
        if not route:
            return RegionalAnchorGuidance(
                mode="unavailable",
                target_topology_node_id=target_id,
                distance_to_target_node_m=round(target_distance, 3),
                node_arrival_radius_m=round(target_arrival_radius, 3),
                reason=f"No backbone route connects the current region to {target_id}.",
            )

        if progress_index < len(route) - 1:
            candidate_next = nodes[route[progress_index + 1]]
            distance_to_candidate = math.hypot(
                candidate_next.x - observation.pose.x,
                candidate_next.y - observation.pose.y,
            )
            candidate_radius = self._node_arrival_radius(plan, candidate_next)
            if distance_to_candidate <= candidate_radius:
                progress_index += 1

        current_id = route[progress_index]
        if progress_index == len(route) - 1 and target_distance <= target_arrival_radius:
            return RegionalAnchorGuidance(
                mode="search_within_region",
                current_topology_node_id=current_id,
                target_topology_node_id=target_id,
                next_topology_node_id=target_id,
                route_node_ids=route,
                route_progress_index=progress_index,
                guidance_point_xyz_m=target.navigation_point_xyz_m,
                distance_to_next_node_m=round(target_distance, 3),
                distance_to_target_node_m=round(target_distance, 3),
                node_arrival_radius_m=round(target_arrival_radius, 3),
                target_region_reached=True,
                allowed_anchor_views=["left", "front", "right"],
                direction_constraint_active=False,
                reason=(
                    f"Robot is {target_distance:.2f} m from {target_id}, inside the "
                    f"{target_arrival_radius:.2f} m arrival radius; search all views for {plan.object_goal}."
                ),
            )

        next_index = min(progress_index + 1, len(route) - 1)
        next_id = route[next_index]
        next_node = nodes[next_id]
        distance_to_next = math.hypot(
            next_node.x - observation.pose.x,
            next_node.y - observation.pose.y,
        )
        next_arrival_radius = self._node_arrival_radius(plan, next_node)
        if next_index == progress_index:
            point = (target.x, target.y)
        else:
            point = self._topology_route_lookahead(
                topology,
                current_id,
                next_id,
                observation.pose.x,
                observation.pose.y,
            )
        dx = point[0] - observation.pose.x
        dy = point[1] - observation.pose.y
        forward = -math.sin(observation.pose.yaw) * dx - math.cos(observation.pose.yaw) * dy
        left = -math.cos(observation.pose.yaw) * dx + math.sin(observation.pose.yaw) * dy
        bearing = math.degrees(math.atan2(left, forward))
        view_centers = (("left", 90.0), ("front", 0.0), ("right", -90.0))
        allowed_views = [
            view
            for view, center in view_centers
            if abs(self._angle_delta_degrees(bearing, center)) <= 65.0
        ]
        viewpoint_action = None
        if not allowed_views:
            viewpoint_action = "turn_left" if bearing >= 0.0 else "turn_right"
        return RegionalAnchorGuidance(
            mode="route_to_region",
            current_topology_node_id=current_id,
            target_topology_node_id=target_id,
            next_topology_node_id=next_id,
            route_node_ids=route,
            route_progress_index=progress_index,
            guidance_point_xyz_m=[point[0], target.navigation_point_xyz_m[1], point[1]],
            distance_to_next_node_m=round(distance_to_next, 3),
            distance_to_target_node_m=round(target_distance, 3),
            node_arrival_radius_m=round(next_arrival_radius, 3),
            target_region_reached=False,
            relative_bearing_deg=round(bearing, 2),
            allowed_anchor_views=allowed_views,
            recommended_viewpoint_action=viewpoint_action,
            direction_constraint_active=True,
            reason=(
                f"Route segment {progress_index + 1}/{len(route)} follows {current_id} to {next_id}; "
                f"next-node distance is {distance_to_next:.2f} m."
            ),
        )

    @staticmethod
    def _node_arrival_radius(plan: SemanticRoutePlan, node: Any) -> float:
        map_radius = max(1.0, min(float(node.clearance_m), 2.5))
        return min(plan.node_local_radius_m, map_radius)

    @staticmethod
    def _shortest_topology_route(
        topology: BackboneTopology,
        source_id: str,
        target_id: str,
    ) -> list[str]:
        adjacency: dict[str, list[tuple[float, str]]] = {}
        for edge in topology.edges:
            adjacency.setdefault(edge.source, []).append((edge.distance_m, edge.target))
            adjacency.setdefault(edge.target, []).append((edge.distance_m, edge.source))
        queue: list[tuple[float, str]] = [(0.0, source_id)]
        distance = {source_id: 0.0}
        previous: dict[str, str] = {}
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost != distance.get(node_id):
                continue
            if node_id == target_id:
                break
            for edge_cost, neighbor in adjacency.get(node_id, []):
                candidate = cost + edge_cost
                if candidate >= distance.get(neighbor, float("inf")):
                    continue
                distance[neighbor] = candidate
                previous[neighbor] = node_id
                heapq.heappush(queue, (candidate, neighbor))
        if target_id not in distance:
            return []
        route = [target_id]
        while route[-1] != source_id:
            route.append(previous[route[-1]])
        route.reverse()
        return route

    @staticmethod
    def _topology_route_lookahead(
        topology: BackboneTopology,
        source_id: str,
        target_id: str,
        robot_x: float,
        robot_y: float,
        *,
        lookahead_m: float = 1.5,
    ) -> tuple[float, float]:
        nodes = {node.node_id: node for node in topology.nodes}
        source = nodes[source_id]
        target = nodes[target_id]
        edge = next(
            (
                item
                for item in topology.edges
                if {item.source, item.target} == {source_id, target_id}
            ),
            None,
        )
        points = list(edge.path_world_xy_m) if edge is not None else []
        if not points:
            points = [(source.x, source.y), (target.x, target.y)]
        if math.hypot(points[0][0] - source.x, points[0][1] - source.y) > math.hypot(
            points[-1][0] - source.x,
            points[-1][1] - source.y,
        ):
            points.reverse()
        nearest_index = min(
            range(len(points)),
            key=lambda index: math.hypot(
                points[index][0] - robot_x,
                points[index][1] - robot_y,
            ),
        )
        traveled = 0.0
        previous_point = points[nearest_index]
        for point in points[nearest_index + 1 :]:
            traveled += math.hypot(point[0] - previous_point[0], point[1] - previous_point[1])
            if traveled >= lookahead_m:
                return float(point[0]), float(point[1])
            previous_point = point
        return float(target.x), float(target.y)

    @staticmethod
    def _angle_delta_degrees(value: float, reference: float) -> float:
        return (value - reference + 180.0) % 360.0 - 180.0

    def advance_navigation(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = AdvanceNavigationInput.model_validate(payload)
        if request.planner_output is not None:
            request = request.model_copy(
                update={
                    "planner_output": self._complete_planner_output(
                        request.planner_output
                    )
                }
            )
        with self._pipeline_lock:
            decision_observation = self._require_observation(payload)
            protocol_normalization: dict[str, object] = {}
            if request.planner_output is not None and self._semantic_route_plan is not None:
                replanning = (
                    request.self_reflection is not None
                    and request.self_reflection.should_replan
                )
                if not replanning:
                    request = request.model_copy(update={"planner_output": None})
                    protocol_normalization.update(
                        {
                            "ignored_planner_output": True,
                            "planner_output_reason": (
                                "An active plan already exists and Self-Reflector did not "
                                "request replanning."
                            ),
                        }
                    )
            latest_id = ((self._latest_observation or {}).get("trace") or {}).get("observation_id")
            if request.observation_id != latest_id:
                raise ValueError(f"stale observation_id: {request.observation_id}; latest is {latest_id}")
            if (
                self.navigation_memory.last_slow_output is not None
                and request.self_reflection is None
            ):
                raise ValueError(
                    "Self-Reflector output is required after every completed macro execution"
                )
            if (
                request.self_reflection is not None
                and request.self_reflection.should_replan
                and request.planner_output is None
            ):
                raise ValueError(
                    "Self-Reflector requested replanning; replacement planner_output is required"
                )
            validate_task_type = (
                "navigation_task_type" in payload or request.planner_output is not None
            )
            request, reflection_normalization = self._normalize_point_goal_reflection(
                request,
                decision_observation,
            )
            if reflection_normalization is not None:
                protocol_normalization.update(reflection_normalization)
            request, task_type_normalization = self._normalize_transition_task_type(
                request
            )
            if task_type_normalization is not None:
                protocol_normalization.update(task_type_normalization)
            final_completion = self._preflight_advance_transition(
                request,
                decision_observation,
                validate_task_type=validate_task_type,
            )
            if final_completion and request.planner_output is None:
                return self._await_finish_verification(request, decision_observation)
            if request.planner_output is not None:
                self._create_semantic_route_plan(
                    request.planner_output, decision_observation
                )
            else:
                self._apply_subtask_reflection(
                    request.self_reflection,
                    decision_observation,
                )
            if self._semantic_route_plan is None and self._planner_required:
                raise ValueError(
                    "the first Slow decision must include planner_output built from "
                    "slow_context topology and current observation"
                )
            if self._semantic_route_plan is not None:
                request = request.model_copy(
                    update={"global_goal": self._semantic_route_plan.global_goal}
                )
            if validate_task_type:
                self._validate_active_task_type(request)
            topology_delta = None
            action, policy_override = self._apply_navigation_policy(request, decision_observation)
            if action is not request.action:
                request = request.model_copy(update={"action": action})
            self.navigation_memory.record_reasoning(request)
            slow_output = ABotSlowOutput(
                slow_step_id=f"slow-{uuid.uuid4().hex[:12]}",
                navigation_task_type=request.navigation_task_type,
                cot=request.cot,
                pixel_goal=None,
                reference_observation_id=request.observation_id,
            )
            slow_output_recorded = False

            def record_slow_output(pixel_goal: DualPixelGoal | None) -> None:
                nonlocal slow_output, slow_output_recorded
                if slow_output_recorded:
                    return
                slow_output = slow_output.model_copy(
                    update={"pixel_goal": pixel_goal}
                )
                self.navigation_memory.record_slow_output(
                    slow_output.model_dump(mode="json")
                )
                slow_output_recorded = True

            common = {
                "slow_step_id": slow_output.slow_step_id,
                "observation_id": request.observation_id,
                "semantic_map": request.semantic_map.model_dump(mode="json"),
            }
            if action.action_type == "visible_anchor":
                execution_candidate = next(
                    candidate
                    for candidate in request.candidates
                    if candidate.action_type == action.action_type
                    and candidate.target.casefold() == action.target.casefold()
                )
                result = self.navigate_to_anchor(
                    {
                        **common,
                        "anchor": action.target,
                        "anchor_role": action.anchor_role,
                        "view": action.view,
                        "low_confidence_recovery": action.low_confidence_recovery,
                        "recovery_viewpoint_action": action.recovery_viewpoint_action,
                        "task_relevance": execution_candidate.task_relevance,
                        "enforce_regional_direction": action.anchor_role == "affordance",
                        "max_control_steps": request.max_control_steps,
                        "goal_tolerance_m": request.goal_tolerance_m,
                        "anchor_max_depth_m": self._regional_anchor_depth_limit(
                            request.anchor_max_depth_m,
                        ),
                    },
                    pixel_guidance_callback=record_slow_output,
                )
            elif action.action_type == "known_location":
                navigation_target = (
                    {"topology_node_id": action.topology_node_id}
                    if action.topology_node_id is not None
                    else {"location_id": action.location_id}
                )
                result = self.navigate_to_known_location(
                    {
                        **common,
                        **navigation_target,
                        "max_control_steps": request.max_control_steps,
                        "goal_tolerance_m": request.goal_tolerance_m,
                    },
                    pixel_guidance_callback=record_slow_output,
                )
            else:
                record_slow_output(None)
                result = self.adjust_viewpoint(
                    {
                        **common,
                        "action": action.viewpoint_action,
                    }
                )
            if not slow_output_recorded:
                record_slow_output(None)
            execution_payload = result.get("result") or {}
            fast_execution = None
            if execution_payload.get("controller"):
                fast_execution = ABotFastExecution(
                    slow_step_id=slow_output.slow_step_id,
                    reference_observation_id=request.observation_id,
                    controller=str(execution_payload["controller"]),
                    reached=bool(execution_payload.get("reached")),
                    control_steps=int(execution_payload.get("control_steps", 0)),
                    distance_m=float(execution_payload.get("distance_m", 0.0)),
                    remaining_distance_m=float(
                        execution_payload.get("remaining_distance_m", 0.0)
                    ),
                    stalled=bool(execution_payload.get("stalled")),
                )
            result["decision"] = action.model_dump(mode="json")
            result["topology_update"] = (
                topology_delta.model_dump(mode="json") if topology_delta is not None else None
            )
            if policy_override is not None:
                result["policy_override"] = policy_override
            if protocol_normalization:
                result["protocol_normalization"] = protocol_normalization
            self._update_semantic_route_after_execution(action, result)
            result["semantic_route_plan"] = (
                self._semantic_route_plan.model_dump(mode="json")
                if self._semantic_route_plan is not None
                else None
            )
            result["agentic_plan"] = (
                self._semantic_route_plan.agentic_plan.model_dump(mode="json")
                if self._semantic_route_plan is not None
                else None
            )
            result["reasoning"] = {
                "previous_assessment": request.previous_assessment.model_dump(mode="json"),
                "route_hypothesis": request.route_hypothesis.model_dump(mode="json"),
                "candidates": [item.model_dump(mode="json") for item in request.candidates],
            }
            result["navigation_task_type"] = request.navigation_task_type
            result["cot"] = (
                request.cot.model_dump(mode="json") if request.cot is not None else None
            )
            result["slow_output"] = slow_output.model_dump(mode="json")
            result["fast_execution"] = (
                fast_execution.model_dump(mode="json")
                if fast_execution is not None
                else None
            )
            result["self_reflection"] = (
                request.self_reflection.model_dump(mode="json")
                if request.self_reflection is not None
                else None
            )
            result["planner_output"] = (
                request.planner_output.model_dump(mode="json")
                if request.planner_output is not None
                else None
            )
            post_observation_id = result.get("next_observation_id")
            post_observation = self._observations.get(str(post_observation_id))
            result["memory_view"] = self.navigation_memory.view(
                request.global_goal,
                current_observation=post_observation,
            ).model_dump(mode="json")
            if post_observation is not None:
                result["slow_context"] = self._slow_context(post_observation).model_dump(
                    mode="json"
                )
            result["active_target_evidence"] = (
                self._active_target_evidence.model_dump(mode="json")
                if self._active_target_evidence is not None
                else None
            )
            result["next_call_contract"] = self._next_call_contract(
                str(post_observation_id) if post_observation_id else None
            )
            result["topo_memory"] = self.navigation_memory.topo_memory_view().model_dump(
                mode="json"
            )
            self._latest_pipeline = dict(result)
            return result

    def _apply_navigation_policy(
        self,
        request: AdvanceNavigationInput,
        observation: Observation,
    ) -> tuple[SelectedNavigationAction, dict[str, object] | None]:
        self._refresh_regional_guidance(observation)
        semantic_action = self._semantic_route_policy(request, observation)
        if semantic_action is not None:
            return semantic_action
        return request.action, None

    def _semantic_route_policy(
        self,
        request: AdvanceNavigationInput,
        observation: Observation,
    ) -> tuple[SelectedNavigationAction, dict[str, object]] | None:
        plan = self._semantic_route_plan
        if plan is None or plan.selected_topology_node_id is None:
            return None
        guidance = plan.regional_guidance
        active_plan = plan.agentic_plan
        active_subtask = active_plan.subtasks[active_plan.active_subtask_index]
        if (
            active_subtask.task_type == "point_goal"
            and guidance is not None
            and guidance.mode == "route_to_region"
            and guidance.next_topology_node_id is not None
            and request.action.action_type != "viewpoint_adjustment"
            and (
                request.action.action_type == "known_location"
                or (
                    request.action.action_type == "visible_anchor"
                    and request.action.anchor_role == "affordance"
                )
            )
            and not (
                request.action.action_type == "visible_anchor"
                and request.action.anchor_role == "affordance"
                and guidance.direction_constraint_active
                and not guidance.allowed_anchor_views
                and guidance.recommended_viewpoint_action is not None
            )
            and not (
                request.action.action_type == "known_location"
                and request.action.topology_node_id == guidance.next_topology_node_id
            )
        ):
            route_target = self.navigation_memory.get_topology_node(
                guidance.next_topology_node_id
            )
            if route_target is None:
                raise ValueError(
                    f"regional route references unknown node: {guidance.next_topology_node_id}"
                )
            action = SelectedNavigationAction(
                action_type="known_location",
                intent="goal_pursuit",
                target=f"{guidance.next_topology_node_id} connected route waypoint",
                topology_node_id=guidance.next_topology_node_id,
                expected_evidence=(
                    f"Arrival at the next connected route node {guidance.next_topology_node_id}."
                ),
                success_condition=(
                    f"Runtime confirms arrival at {guidance.next_topology_node_id}."
                ),
                fallback="Reassess the connected topology route from the current node.",
            )
            return action, {
                "reason": "connected_topology_step_enforced",
                "constraint": "point_goal_topology_priority",
                "regional_guidance": guidance.model_dump(mode="json"),
                "replaced_action": request.action.model_dump(mode="json"),
            }
        if (
            request.action.action_type == "visible_anchor"
            and request.action.anchor_role == "affordance"
            and guidance is not None
            and guidance.direction_constraint_active
            and not guidance.allowed_anchor_views
            and guidance.recommended_viewpoint_action is not None
        ):
            action = SelectedNavigationAction(
                action_type="viewpoint_adjustment",
                intent="explore_route",
                target=f"face route toward {guidance.next_topology_node_id}",
                viewpoint_action=guidance.recommended_viewpoint_action,
                expected_evidence="A traversable anchor inside the topology-guided route direction.",
                success_condition="The next route segment enters at least one current camera view.",
                fallback="Use a historical observation or reassess the regional route.",
            )
            return action, {
                "reason": "regional_route_outside_current_views",
                "regional_guidance": guidance.model_dump(mode="json"),
                "replaced_action": request.action.model_dump(mode="json"),
            }
        if request.action.action_type != "known_location" or request.action.topology_node_id is None:
            return None
        if (
            guidance is not None
            and guidance.mode == "route_to_region"
            and guidance.next_topology_node_id is not None
            and request.action.topology_node_id != guidance.next_topology_node_id
        ):
            route_target = self.navigation_memory.get_topology_node(
                guidance.next_topology_node_id
            )
            if route_target is None:
                raise ValueError(
                    f"regional route references unknown node: {guidance.next_topology_node_id}"
                )
            action = request.action.model_copy(
                update={
                    "target": f"{guidance.next_topology_node_id} connected route waypoint",
                    "topology_node_id": guidance.next_topology_node_id,
                    "expected_evidence": (
                        f"Arrival at the next connected route node {guidance.next_topology_node_id}."
                    ),
                    "success_condition": (
                        f"Runtime confirms arrival at {guidance.next_topology_node_id}."
                    ),
                }
            )
            return action, {
                "reason": "connected_topology_step_enforced",
                "regional_guidance": guidance.model_dump(mode="json"),
                "replaced_action": request.action.model_dump(mode="json"),
            }
        next_target = self.navigation_memory.get_topology_node(request.action.topology_node_id)
        if next_target is None:
            raise ValueError("semantic node switching requires a fixed topology_node_id")
        if next_target.topology_node_id != plan.selected_topology_node_id:
            self._activate_semantic_node(
                plan,
                next_target,
                reason=(
                    f"Agent selected {next_target.topology_node_id} as a new regional hypothesis."
                ),
            )
        return None

    def _activate_semantic_node(
        self,
        plan: SemanticRoutePlan,
        target: Any,
        *,
        reason: str,
        match_score: float | None = None,
        matched_terms: list[str] | None = None,
    ) -> None:
        self._semantic_route_plan = plan.model_copy(
            update={
                "phase": "local_anchor_search",
                "selected_topology_node_id": target.topology_node_id,
                "evidence_location_id": None,
                "selected_node_semantics": target.node_semantics,
                "match_score": match_score,
                "matched_terms": matched_terms or [],
                "selection_reason": reason,
            }
        )

    def _update_semantic_route_after_execution(
        self,
        action: SelectedNavigationAction,
        result: dict[str, Any],
    ) -> None:
        plan = self._semantic_route_plan
        if plan is None:
            return
        execution = result.get("result") or {}
        generated = result.get("generated_point") or {}
        if (
            action.action_type == "visible_anchor"
            and action.anchor_role == "target"
            and bool(result.get("anchor_visible"))
            and float(result.get("grounding_confidence") or 0.0)
            >= self.target_confidence_threshold
        ):
            view = str(generated.get("view") or action.view)
            pixel_goal = execution.get("pixel_goal") or result.get("pixel_guidance") or {}
            target_pixel = generated.get("target_pixel") or (
                (pixel_goal.get("target_pixel") or {}).get(view)
            )
            affordance_pixel = (pixel_goal.get("affordance_pixel") or {}).get(view)
            source_observation_id = str(
                result.get("anchor_source_observation_id") or ""
            )
            post_observation_id = str(result.get("next_observation_id") or "")
            source_observation = self._observations.get(source_observation_id)
            if (
                view in {"left", "front", "right"}
                and isinstance(target_pixel, list)
                and len(target_pixel) == 2
                and isinstance(affordance_pixel, list)
                and len(affordance_pixel) == 2
                and source_observation_id
                and post_observation_id
            ):
                self._active_target_evidence = ABotTargetEvidence(
                    target=action.target,
                    confidence=float(result["grounding_confidence"]),
                    source_observation_id=source_observation_id,
                    post_observation_id=post_observation_id,
                    view=view,
                    target_pixel=[int(value) for value in target_pixel],
                    affordance_pixel=[int(value) for value in affordance_pixel],
                    source_pose=(
                        source_observation.pose.model_dump(mode="json")
                        if source_observation is not None
                        and source_observation.pose is not None
                        else None
                    ),
                    execution_reached=bool(execution.get("reached")),
                    remaining_distance_m=float(
                        execution.get("remaining_distance_m") or 0.0
                    ),
                    evidence_summary=str(
                        generated.get("evidence_summary")
                        or action.expected_evidence
                    )[:240],
                )
        if (
            action.action_type == "known_location"
            and action.topology_node_id == plan.selected_topology_node_id
        ):
            self._semantic_route_plan = plan.model_copy(
                update={"evidence_location_id": None}
            )
        elif (
            action.action_type == "known_location"
            and action.location_id is not None
            and plan.phase in {"approach_node", "local_anchor_search", "verify_goal"}
            and bool(execution.get("reached"))
        ):
            self._semantic_route_plan = plan.model_copy(
                update={
                    "phase": "local_anchor_search",
                    "evidence_location_id": action.location_id,
                }
            )
        post_observation = self._observations.get(str(result.get("next_observation_id") or ""))
        if post_observation is not None:
            self._refresh_regional_guidance(post_observation)

    def _capture_observation(self) -> Observation:
        observation = asyncio.run(self.adapter.observe(self._trace()))
        missing_depth = sorted(set(observation.rgb_refs) - set(observation.depth_refs))
        if missing_depth:
            raise RuntimeError(f"Hierarchical version requires matching depth views: {missing_depth}")
        self._cache_observation(observation)
        return observation

    def relay_live(self) -> dict[str, Any]:
        observation = asyncio.run(self.adapter.observe_live(self._trace()))
        return {
            "ok": True,
            "health": self.health(),
            "observation": observation.model_dump(mode="json"),
            "omninav": self._latest_omninav,
            "navigation_task": self._relay_navigation_task(),
            "route_graph": self.navigation_memory.route_graph(),
        }

    def relay_state(self) -> dict[str, Any]:
        return {
            "ok": True,
            "health": self.health(),
            "observation": self._latest_observation,
            "omninav": self._latest_omninav,
            "navigation_task": self._relay_navigation_task(),
            "route_graph": self.navigation_memory.route_graph(),
        }

    def _relay_navigation_task(self) -> dict[str, Any]:
        pipeline = self._latest_pipeline or {
            "status": "idle",
            "profile": "hierarchical",
        }
        agent_task = self.agent_task_status()
        if agent_task.get("status") == "idle":
            return dict(pipeline)
        return {**pipeline, **agent_task}

    def run_anchor_grounding(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._require_observation(payload)
        inputs = {
            "anchor": payload.get("anchor"),
            "view": payload.get("view"),
            "image_width": self._image_width(observation),
            "image_height": self._image_height(observation),
            "allowed_views": payload.get("allowed_views") or [],
            "direction_hint": payload.get("direction_hint") or "",
        }
        return self._run_stage("AnchorGroundingSkill", observation, inputs, timeout_s=60.0)

    def run_pixel_to_point(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._require_observation(payload)
        inputs = {
            "source_observation_id": observation.trace.observation_id,
            "view": payload.get("view"),
            "x_px": payload.get("x_px"),
            "y_px": payload.get("y_px"),
            "image_width": self._image_width(observation),
            "image_height": self._image_height(observation),
            "max_depth_m": payload.get("max_depth_m", 5.0),
        }
        return self._run_stage("PixelToPointSkill", observation, inputs)

    def run_path_planning(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._require_observation(payload)
        return self._run_stage(
            "PathPlanningSkill",
            observation,
            {"world_point_xyz_m": payload.get("world_point_xyz_m")},
        )

    def run_path_execution(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._require_observation(payload)
        with self._pipeline_lock:
            return self._run_stage(
                "PathExecutionSkill",
                observation,
                {
                    "world_path_xyz_m": payload.get("world_path_xyz_m"),
                    "max_control_steps": payload.get("max_control_steps", 64),
                    "goal_tolerance_m": payload.get("goal_tolerance_m", 0.3),
                    "control_period_s": payload.get("control_period_s", 0.1),
                },
                timeout_s=120.0,
            )

    def adjust_viewpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = ViewpointAdjustmentInput.model_validate(payload)
        with self._pipeline_lock:
            observation = self._require_observation(payload)
            latest_id = ((self._latest_observation or {}).get("trace") or {}).get("observation_id")
            if request.observation_id != latest_id:
                raise ValueError(f"stale observation_id: {request.observation_id}; latest is {latest_id}")
            memory_delta = self.navigation_memory.record_place(observation, request.semantic_map)
            stage = self._run_stage(
                "ViewpointAdjustmentSkill",
                observation,
                request.model_dump(mode="json"),
            )
            post_observation = self._capture_observation()
            post_observation = self._decorate_semantic_map(post_observation)
            state = {
                "ok": True,
                "status": "completed",
                "profile": "hierarchical",
                "phase": "viewpoint_adjusted",
                "action": request.action,
                "observation_id": request.observation_id,
                "result": stage["result"]["outputs"],
                "memory_delta": memory_delta.model_dump(mode="json"),
                "post_observation": post_observation.model_dump(mode="json"),
                "next_observation_id": post_observation.trace.observation_id,
                "finished_at": utc_now().isoformat(),
            }
            self._latest_pipeline = state
            return dict(state)

    def inspect_localization_map(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = LocalizationMapInput.model_validate(payload)
        with self._pipeline_lock:
            observation = self._capture_observation()
            memory_delta = self.navigation_memory.record_place(observation, request)
            observation = self._decorate_semantic_map(observation)
            stage = self._run_stage(
                "LocalizationMapSkill",
                observation,
                request.model_dump(mode="json"),
            )
            state = {
                "ok": True,
                "status": "completed",
                "profile": "hierarchical",
                "phase": "localization_map_observed",
                "result": stage["result"]["outputs"],
                "memory_delta": memory_delta.model_dump(mode="json"),
                "memory_view": self.navigation_memory.view().model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
                "next_observation_id": observation.trace.observation_id,
                "finished_at": utc_now().isoformat(),
            }
            self._latest_pipeline = state
            return dict(state)

    def navigate_to_known_location(
        self,
        payload: dict[str, Any],
        *,
        pixel_guidance_callback: Callable[[DualPixelGoal | None], None] | None = None,
    ) -> dict[str, Any]:
        request = KnownLocationNavigateInput.model_validate(payload)
        with self._pipeline_lock:
            observation = self._require_observation(payload)
            latest_id = ((self._latest_observation or {}).get("trace") or {}).get("observation_id")
            if request.observation_id != latest_id:
                raise ValueError(f"stale observation_id: {request.observation_id}; latest is {latest_id}")
            target = (
                self.navigation_memory.get_topology_node(request.topology_node_id)
                if request.topology_node_id is not None
                else self.navigation_memory.get_place(request.location_id or "")
            )
            if target is None:
                target_id = request.topology_node_id or request.location_id
                raise ValueError(f"unknown navigation target: {target_id}")
            navigation_id = request.topology_node_id or request.location_id
            target_kind = "topology_node" if request.topology_node_id is not None else "history_location"
            self.navigation_memory.record_place(observation, request.semantic_map)

            state: dict[str, Any] = {
                "ok": False,
                "status": "running",
                "profile": "hierarchical",
                "phase": "known_location_point_navigation",
                "navigation_target_id": navigation_id,
                "navigation_target_kind": target_kind,
                "topology_node_id": request.topology_node_id,
                "location_id": request.location_id,
                "target_location": {
                    "navigation_id": navigation_id,
                    "kind": target_kind,
                    "scene_summary": target.scene_summary,
                    "semantic_labels": target.semantic_labels[:6],
                },
                "observation_id": request.observation_id,
                "task_id": self.task_id,
                "started_at": utc_now().isoformat(),
                "stages": [],
            }
            self._latest_pipeline = state
            try:
                execution_started = time.monotonic()
                current_observation = observation
                trajectory_waypoints: list[dict[str, Any]] = []
                trajectory_candidates: list[dict[str, Any]] = []
                total_control_steps = 0
                total_distance_m = 0.0
                total_collisions = 0
                termination_reason = "continuation_limit"
                outputs: dict[str, Any] = {}
                while total_control_steps < self.point_navigation_total_step_limit:
                    remaining_time = self.point_navigation_timeout_s - (
                        time.monotonic() - execution_started
                    )
                    if remaining_time <= 0.0:
                        termination_reason = "timeout"
                        break
                    chunk_steps = min(
                        request.max_control_steps,
                        self.point_navigation_total_step_limit - total_control_steps,
                    )
                    execution = self._run_stage(
                        "PointNavigationSkill",
                        current_observation,
                        {
                            "slow_step_id": request.slow_step_id,
                            "target_id": navigation_id,
                            "world_point_xyz_m": target.navigation_point_xyz_m,
                            # Topology nodes are regions, so preserve the final path tangent.
                            # Historical locations retain their recorded observation heading.
                            "target_yaw": (
                                target.pose.yaw
                                if request.location_id is not None
                                else None
                            ),
                            "max_control_steps": chunk_steps,
                            "goal_tolerance_m": request.goal_tolerance_m,
                            "control_period_s": 0.1,
                            "source_observation_id": current_observation.trace.observation_id,
                        },
                        timeout_s=min(120.0, remaining_time),
                        progress_callback=(
                            self._pixel_guidance_progress_callback(
                                pixel_guidance_callback
                            )
                            if not state["stages"]
                            else None
                        ),
                    )
                    state["stages"].append(execution)
                    outputs = dict(execution["result"]["outputs"])
                    chunk_control_steps = int(outputs.get("control_steps", 0))
                    trajectory_waypoints.extend(
                        self._offset_trajectory_frames(
                            outputs.get("trajectory_waypoints") or [],
                            step_offset=total_control_steps,
                            skip_initial=bool(trajectory_waypoints),
                        )
                    )
                    trajectory_candidates.extend(
                        self._offset_trajectory_frames(
                            outputs.get("trajectory_keyframes") or [],
                            step_offset=total_control_steps,
                            skip_initial=bool(trajectory_candidates),
                        )
                    )
                    total_control_steps += chunk_control_steps
                    total_distance_m += float(outputs.get("distance_m", 0.0))
                    total_collisions += int(outputs.get("collisions", 0))
                    if bool(outputs.get("reached")):
                        termination_reason = "reached"
                        break
                    if bool(outputs.get("stalled")):
                        termination_reason = "stalled"
                        break
                    if chunk_control_steps < chunk_steps:
                        termination_reason = "execution_stopped"
                        break
                    current_observation = self._capture_observation()

                trajectory_keyframes = self._select_trajectory_keyframes(
                    trajectory_candidates
                )
                outputs.update(
                    control_steps=total_control_steps,
                    distance_m=total_distance_m,
                    collisions=total_collisions,
                    trajectory_waypoints=trajectory_waypoints,
                    trajectory_keyframes=trajectory_keyframes,
                    execution_chunks=len(state["stages"]),
                    termination_reason=termination_reason,
                )
                motion_performed = (
                    total_control_steps > 0 or total_distance_m > 1e-3
                )
                outputs["motion_performed"] = motion_performed
                outputs["completion_scope"] = "location_arrival_only"
                if bool(outputs.get("reached")) and not motion_performed:
                    outputs["no_motion_reason"] = "already_at_target_location"
                self.navigation_memory.record_route_waypoints(
                    trajectory_waypoints,
                    source_target=target.scene_summary,
                )
                trajectory_keyframes = self.navigation_memory.record_trajectory_keyframes(
                    trajectory_keyframes,
                    source_target=target.scene_summary,
                )
                outputs["trajectory_keyframes"] = trajectory_keyframes
                reached = bool(outputs["reached"])
                state["phase"] = "post_execution_observe"
                post_observation = self._capture_observation()
                memory_delta = self.navigation_memory.mark_known_visit(
                    post_observation,
                    target,
                    reach_tolerance_m=max(0.75, request.goal_tolerance_m),
                )
                post_observation = self._decorate_semantic_map(post_observation)
                state.update(
                    ok=reached,
                    status="completed" if reached else "incomplete",
                    phase="review_known_location",
                    result=outputs,
                    execution_telemetry=self._execution_telemetry(outputs),
                    trajectory_memory=self._trajectory_memory_index(trajectory_keyframes),
                    post_observation=post_observation.model_dump(mode="json"),
                    next_observation_id=post_observation.trace.observation_id,
                    memory_delta=memory_delta.model_dump(mode="json"),
                    finished_at=utc_now().isoformat(),
                )
            except Exception as exc:
                state.update(status="failed", error=str(exc), finished_at=utc_now().isoformat())
                raise
            finally:
                self._latest_pipeline = state
            return dict(state)

    @staticmethod
    def _offset_trajectory_frames(
        frames: list[dict[str, Any]],
        *,
        step_offset: int,
        skip_initial: bool,
    ) -> list[dict[str, Any]]:
        adjusted: list[dict[str, Any]] = []
        for frame in frames:
            frame_index = int(frame.get("frame_index", 0))
            if skip_initial and frame_index == 0:
                continue
            adjusted.append({**frame, "frame_index": step_offset + frame_index})
        return adjusted

    @staticmethod
    def _select_trajectory_keyframes(
        frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(frames) <= 4:
            return frames
        ordered = sorted(frames, key=lambda frame: int(frame.get("frame_index", 0)))
        last_frame = int(ordered[-1].get("frame_index", 0))
        selected: list[dict[str, Any]] = []
        for position in range(4):
            target = position * last_frame / 3
            frame = min(
                ordered,
                key=lambda item: abs(int(item.get("frame_index", 0)) - target),
            )
            if frame not in selected:
                selected.append(frame)
        return selected

    def navigate_to_anchor(
        self,
        payload: dict[str, Any],
        *,
        pixel_guidance_callback: Callable[[DualPixelGoal | None], None] | None = None,
        allow_stale_slow_reference: bool = False,
    ) -> dict[str, Any]:
        request = HierarchicalNavigateInput.model_validate(payload)
        with self._pipeline_lock:
            observation = (
                self._observations.get(request.observation_id)
                if request.observation_id is not None
                else None
            )
            if request.observation_id is not None and observation is None:
                raise ValueError(f"unknown or expired observation_id: {request.observation_id}")
            latest_id = ((self._latest_observation or {}).get("trace") or {}).get("observation_id")
            source_is_stale = (
                request.observation_id is not None
                and request.observation_id != latest_id
            )
            if source_is_stale and not allow_stale_slow_reference:
                raise ValueError(f"stale observation_id: {request.observation_id}; latest is {latest_id}")
            if observation is None:
                observed = self.observe()
                observation_id = observed["observation"]["trace"]["observation_id"]
                observation = self._observations[observation_id]

            state: dict[str, Any] = {
                "ok": False,
                "status": "running",
                "profile": "hierarchical",
                "anchor": request.anchor,
                "view": request.view,
                "observation_id": observation.trace.observation_id,
                "task_id": self.task_id,
                "scene_memory_enabled": self.scene_memory_enabled,
                "started_at": utc_now().isoformat(),
                "stages": [],
            }
            memory_update: dict[str, object] | None = None
            if self.scene_memory_enabled and request.scene_memory is not None:
                memory_update = self._scene_memory.record_visit(
                    observation,
                    request.scene_memory,
                    selected_anchor=f"{request.anchor} [{request.view}]",
                )
                state["scene_memory_update"] = memory_update
                state["scene_memory_index"] = self._scene_memory.compact_index()
            elif self.scene_memory_enabled:
                state["scene_memory_warning"] = "scene_memory was omitted; this stop was not recorded"
            memory_delta = self.navigation_memory.record_place(observation, request.semantic_map)
            self._latest_pipeline = state
            try:
                generated = (
                    self._project_slow_brain_target_pixel(request, observation)
                    if request.target_pixel is not None
                    else self._run_stage(
                        "AnchorGenerationSkill",
                        observation,
                        {
                            "source_observation_id": observation.trace.observation_id,
                            "anchor": request.anchor,
                            "anchor_role": request.anchor_role,
                            "view": request.view,
                            "image_width": self._image_width(observation),
                            "image_height": self._image_height(observation),
                            "max_depth_m": request.anchor_max_depth_m,
                            "allowed_views": self._allowed_anchor_views(request),
                            "direction_hint": self._anchor_direction_hint(request),
                        },
                        timeout_s=60.0,
                    )
                )
                state["stages"].append(generated)
                point = generated["result"]["outputs"]
                point, node_scope = self._anchor_region_context(point)
                generated["result"]["outputs"] = point
                confidence = float(point["confidence"])
                confidence_threshold = (
                    self.target_confidence_threshold
                    if request.anchor_role == "target"
                    else self.anchor_confidence_threshold
                )
                state.update(
                    phase="anchor_point_generated",
                    anchor_role=request.anchor_role,
                    requested_view=request.view,
                    view=point["view"],
                    anchor_source_observation_id=observation.trace.observation_id,
                    anchor_annotation_ref=point.get("annotated_ref"),
                    generated_point=point,
                    anchor_visible=point["anchor_visible"],
                    grounding_confidence=confidence,
                    grounding_confidence_threshold=confidence_threshold,
                    target_confidence_threshold=self.target_confidence_threshold,
                    anchor_max_depth_m=request.anchor_max_depth_m,
                    raw_anchor_depth_m=point.get("raw_depth_m"),
                    anchor_depth_limited=bool(point.get("depth_limited")),
                    **node_scope,
                )
                if confidence < confidence_threshold:
                    if pixel_guidance_callback is not None:
                        pixel_guidance_callback(None)
                    recovery_type, recovery_target, recovery = self._recover_low_confidence_anchor(
                        request,
                        observation,
                    )
                    if memory_update is not None:
                        self._scene_memory.record_result(
                            str(memory_update["scene_id"]),
                            status="incomplete",
                            reached=False,
                        )
                    state["stages"].extend(recovery.get("stages") or [])
                    state.update(
                        ok=bool(recovery.get("ok")),
                        status="incomplete",
                        phase="review_low_confidence_recovery",
                        anchor_rejected=True,
                        recovery={
                            "requested": request.low_confidence_recovery,
                            "executed": recovery_type,
                            "target": recovery_target,
                        },
                        result=recovery.get("result") or {},
                        execution_telemetry=self._execution_telemetry(
                            recovery.get("result") or {}
                        ),
                        trajectory_keyframes=(recovery.get("result") or {}).get(
                            "trajectory_keyframes", []
                        ),
                        trajectory_memory=recovery.get("trajectory_memory") or [],
                        post_observation=recovery.get("post_observation"),
                        next_observation_id=recovery.get("next_observation_id"),
                        memory_delta=recovery.get("memory_delta")
                        or memory_delta.model_dump(mode="json"),
                        scene_memory_index=(
                            self._scene_memory.compact_index()
                            if self.scene_memory_enabled
                            else []
                        ),
                        finished_at=utc_now().isoformat(),
                    )
                    return dict(state)
                execution_observation = observation
                if source_is_stale:
                    execution_observation = self._capture_observation()
                    state["slow_reference_observation_id"] = observation.trace.observation_id
                    state["execution_observation_id"] = (
                        execution_observation.trace.observation_id
                    )
                execution = self._run_stage(
                    "PointNavigationSkill",
                    execution_observation,
                    {
                        "slow_step_id": request.slow_step_id,
                        "target_id": point["target_id"],
                        "world_point_xyz_m": point["world_point_xyz_m"],
                        "max_control_steps": request.max_control_steps,
                        "goal_tolerance_m": request.goal_tolerance_m,
                        "control_period_s": 0.1,
                        "source_observation_id": execution_observation.trace.observation_id,
                        "target_view": request.n1_target_view
                        or (
                            point["view"]
                            if point.get("target_pixel") is not None
                            else None
                        ),
                        "target_pixel": request.n1_target_pixel
                        or point.get("target_pixel"),
                    },
                    timeout_s=120.0,
                    progress_callback=self._pixel_guidance_progress_callback(
                        pixel_guidance_callback
                    ),
                )
                state["stages"].append(execution)
                outputs = execution["result"]["outputs"]
                self.navigation_memory.record_route_waypoints(
                    outputs.get("trajectory_waypoints", []),
                    source_target=request.anchor,
                )
                trajectory_keyframes = self.navigation_memory.record_trajectory_keyframes(
                    outputs.get("trajectory_keyframes", []),
                    source_target=request.anchor,
                )
                outputs["trajectory_keyframes"] = trajectory_keyframes
                reached = bool(outputs["reached"])
                if memory_update is not None:
                    self._scene_memory.record_result(
                        str(memory_update["scene_id"]),
                        status="completed" if reached else "incomplete",
                        reached=reached,
                    )
                state.update(
                    phase="post_execution_observe",
                    execution_finished_at=utc_now().isoformat(),
                )
                post_observation = self._capture_observation()
                post_observation = self._decorate_semantic_map(post_observation)
                state.update(
                    ok=reached,
                    status="completed" if reached else "incomplete",
                    finished_at=utc_now().isoformat(),
                    result=execution["result"]["outputs"],
                    execution_telemetry=self._execution_telemetry(outputs),
                    pixel_guidance=outputs.get("pixel_goal"),
                    trajectory_keyframes=trajectory_keyframes,
                    trajectory_memory=self._trajectory_memory_index(trajectory_keyframes),
                    post_observation=post_observation.model_dump(mode="json"),
                    next_observation_id=post_observation.trace.observation_id,
                    memory_delta=memory_delta.model_dump(mode="json"),
                    topology_update=None,
                    semantic_topology_threshold=self.semantic_topology_threshold,
                    scene_memory_index=self._scene_memory.compact_index() if self.scene_memory_enabled else [],
                    phase="review_trajectory",
                )
            except Exception as exc:
                state.update(status="failed", error=str(exc), finished_at=utc_now().isoformat())
                raise
            finally:
                self._latest_pipeline = state
            return dict(state)

    def _project_slow_brain_target_pixel(
        self,
        request: HierarchicalNavigateInput,
        observation: Observation,
    ) -> dict[str, Any]:
        """Project an already grounded Slow-Brain pixel without another model call."""
        assert request.target_pixel is not None
        assert request.target_pixel_confidence is not None
        assert request.target_pixel_evidence is not None
        x_px, y_px = request.target_pixel
        projection = self._run_stage(
            "PixelToPointSkill",
            observation,
            {
                "source_observation_id": observation.trace.observation_id,
                "view": request.view,
                "x_px": x_px,
                "y_px": y_px,
                "image_width": self._image_width(observation),
                "image_height": self._image_height(observation),
                "max_depth_m": request.anchor_max_depth_m,
            },
        )
        spatial = projection["result"]["outputs"]
        pixel = AnchorPixel(
            anchor=request.anchor,
            view=request.view,
            x_px=x_px,
            y_px=y_px,
            anchor_visible=True,
            confidence=request.target_pixel_confidence,
            evidence_summary=request.target_pixel_evidence,
            bbox_px=request.target_bbox_px,
        )
        run_id = request.slow_step_id or uuid.uuid4().hex
        annotated_ref = AnchorGroundingSkill._annotate_anchor(
            observation.rgb_refs[pixel.view],
            pixel,
            run_id,
            primary_role=request.anchor_role,
        )
        annotation_view = pixel.view
        annotation_role = request.anchor_role
        point = GeneratedNavigationPoint(
            target_id=f"slow-pixel-{request.slow_step_id or uuid.uuid4().hex[:12]}",
            source_observation_id=observation.trace.observation_id,
            anchor=request.anchor,
            anchor_role=request.anchor_role,
            view=request.view,
            grounding_pixel=[x_px, y_px],
            target_pixel=request.n1_target_pixel,
            target_view=request.n1_target_view,
            target_label=request.n1_target_label,
            target_bbox_px=request.n1_target_bbox_px,
            target_confidence=request.n1_target_confidence,
            annotation_view=annotation_view,
            annotation_role=annotation_role,
            bbox_px=request.target_bbox_px,
            world_point_xyz_m=spatial["navigation_candidate_xyz_m"],
            anchor_visible=True,
            confidence=request.target_pixel_confidence,
            evidence_summary=request.target_pixel_evidence,
            annotated_ref=annotated_ref,
            depth_m=spatial["depth_m"],
            raw_depth_m=spatial.get("raw_depth_m"),
            depth_limited=bool(spatial.get("depth_limited")),
            max_depth_m=request.anchor_max_depth_m,
        )
        projection["stage"] = "SlowBrainTargetPixelProjection"
        projection["result"]["outputs"] = point.model_dump(mode="json")
        return projection

    def _anchor_region_context(
        self,
        point: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, object]]:
        plan = self._semantic_route_plan
        if (
            plan is None
            or plan.selected_topology_node_id is None
            or point.get("world_point_xyz_m") is None
        ):
            return point, {}
        guidance = plan.regional_guidance
        context_node_id = plan.selected_topology_node_id
        if (
            guidance is not None
            and guidance.mode == "route_to_region"
            and guidance.next_topology_node_id is not None
        ):
            context_node_id = guidance.next_topology_node_id
        node = self.navigation_memory.get_topology_node(context_node_id)
        if node is None:
            return point, {}

        candidate = [float(value) for value in point["world_point_xyz_m"]]
        center = [float(value) for value in node.navigation_point_xyz_m]
        distance = math.hypot(candidate[0] - center[0], candidate[2] - center[2])
        return point, {
            "active_semantic_node_id": plan.selected_topology_node_id,
            "regional_context_node_id": context_node_id,
            "active_evidence_location_id": plan.evidence_location_id,
            "anchor_distance_from_node_m": round(distance, 3),
            "node_scope_limited": False,
            "route_progress_index": (
                guidance.route_progress_index if guidance is not None else 0
            ),
            "target_region_reached": (
                guidance.target_region_reached if guidance is not None else False
            ),
        }

    def _regional_anchor_depth_limit(
        self,
        requested_depth_m: float,
    ) -> float:
        plan = self._semantic_route_plan
        if plan is None:
            return requested_depth_m
        guidance = plan.regional_guidance
        if (
            guidance is None
            or guidance.mode != "route_to_region"
            or guidance.distance_to_next_node_m is None
        ):
            return requested_depth_m
        return round(min(requested_depth_m, guidance.distance_to_next_node_m), 3)

    def _allowed_anchor_views(self, request: HierarchicalNavigateInput) -> list[str]:
        if not request.enforce_regional_direction or self._semantic_route_plan is None:
            return []
        guidance = self._semantic_route_plan.regional_guidance
        if guidance is None or not guidance.direction_constraint_active:
            return []
        return list(guidance.allowed_anchor_views)

    def _anchor_direction_hint(self, request: HierarchicalNavigateInput) -> str:
        allowed_views = self._allowed_anchor_views(request)
        if not allowed_views or self._semantic_route_plan is None:
            return ""
        guidance = self._semantic_route_plan.regional_guidance
        if guidance is None:
            return ""
        return (
            f"Active connected segment {guidance.current_topology_node_id} to "
            f"{guidance.next_topology_node_id}; only {', '.join(allowed_views)} view anchors are "
            f"allowed until the next node is within {guidance.node_arrival_radius_m} m."
        )

    def _recover_low_confidence_anchor(
        self,
        request: HierarchicalNavigateInput,
        observation: Observation,
    ) -> tuple[str, str, dict[str, Any]]:
        previous = self.navigation_memory.previous_place()
        if request.low_confidence_recovery == "previous_location" and previous is not None:
            recovery = self.navigate_to_known_location(
                {
                    "location_id": previous.place_id,
                    "observation_id": observation.trace.observation_id,
                    "semantic_map": request.semantic_map.model_dump(mode="json"),
                    "max_control_steps": request.max_control_steps,
                    "goal_tolerance_m": request.goal_tolerance_m,
                }
            )
            return "previous_location", previous.place_id, recovery

        viewpoint_action = request.recovery_viewpoint_action
        if viewpoint_action is None:
            viewpoint_action = {
                "left": "turn_left",
                "front": "turn_left",
                "right": "turn_right",
            }[request.view]
        recovery = self.adjust_viewpoint(
            {
                "observation_id": observation.trace.observation_id,
                "semantic_map": request.semantic_map.model_dump(mode="json"),
                "action": viewpoint_action,
            }
        )
        return "adjust_viewpoint", viewpoint_action, recovery

    @staticmethod
    def _trajectory_memory_index(
        keyframes: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            {
                "location_id": frame.get("location_id"),
                "frame_index": frame.get("frame_index"),
                "observation_id": frame.get("observation_id"),
                "timestamp": frame.get("timestamp"),
                "pose": frame.get("pose"),
                "map_id": frame.get("map_id"),
                "pose_confidence": frame.get("pose_confidence"),
                "navigation_point_xyz_m": frame.get("navigation_point_xyz_m"),
                "available_views": sorted((frame.get("rgb_refs") or {}).keys()),
            }
            for frame in keyframes
        ]

    def _verify_global_finish(
        self,
        *,
        requested_outcome: str,
        observation: Observation,
    ) -> FinishVerifierDecision | None:
        if self._finish_verifier_provider_factory is None:
            return None
        route_plan = self._semantic_route_plan
        task_context = None
        if route_plan is not None:
            task_context = {
                "global_goal": route_plan.global_goal,
                "destination_query": route_plan.destination_query,
                "object_goal": route_plan.object_goal,
                "destination_topology_node_id": route_plan.destination_topology_node_id,
                "subtasks": [
                    {
                        "stage": subtask.stage,
                        "target": subtask.target,
                        "topology_node_id": subtask.topology_node_id,
                        "success_condition": subtask.success_condition,
                        "status": subtask.status,
                    }
                    for subtask in route_plan.agentic_plan.subtasks
                ],
            }
        prompt = (
            f"Requested global outcome: {requested_outcome}\n"
            f"Original task and Agentic Plan: {task_context}\n"
            f"Latest Slow decision: {self.navigation_memory.last_slow_output}\n"
            f"Sticky active target evidence: "
            f"{self._active_target_evidence.model_dump(mode='json') if self._active_target_evidence else None}\n"
            f"Task memory summary: {self.navigation_memory.summary()}\n"
            "Independently verify the original global task against the latest synchronized observation. "
            "Judge the scene and task semantically. A local anchor reach, A* arrival, Skill success, or model "
            "STOP alone is not sufficient. Set evidence_satisfies_task only when current or preserved target "
            "evidence satisfies the original task. Set terminal_failure_supported only when the evidence proves "
            "the task cannot continue. Return short, concrete observed evidence."
        )

        async def complete() -> FinishVerifierDecision:
            provider = self._finish_verifier_provider_factory()
            try:
                view_audit = await audit_observation_views(provider, observation)
                response = await provider.complete(
                    [
                        ChatMessage(
                            role="system",
                            content=(
                                "You are the independent AgentOS Finish Verifier. Return strict JSON."
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=audited_observation_content(
                                prompt,
                                observation,
                                view_audit,
                            ),
                        ),
                    ],
                    response_schema=FinishVerifierAssessment.model_json_schema(),
                )
            finally:
                await provider.close()
            if not isinstance(response.parsed_json, dict):
                raise ValueError("Finish Verifier did not return a JSON object")
            try:
                assessment = FinishVerifierAssessment.model_validate(response.parsed_json)
            except ValidationError as exc:
                detail = _compact_error_message(exc)
                return FinishVerifierDecision(
                    requested_outcome=requested_outcome,
                    authorized=False,
                    recommended_action="replan",
                    summary="Finish Verifier did not provide valid structured evidence.",
                    observed_evidence=[
                        f"Finish Verifier protocol error: {detail}"
                    ],
                )
            if requested_outcome == "done":
                authorized = assessment.evidence_satisfies_task
                recommended_action = "finish" if authorized else "replan"
                summary = (
                    "Finish Verifier confirmed that the available evidence satisfies the global task."
                    if authorized
                    else "Finish Verifier did not confirm the global task from the available evidence."
                )
            else:
                authorized = assessment.terminal_failure_supported
                recommended_action = "fail" if authorized else "replan"
                summary = (
                    "Finish Verifier confirmed terminal failure from the available evidence."
                    if authorized
                    else "Finish Verifier did not confirm terminal failure from the available evidence."
                )
            return FinishVerifierDecision(
                requested_outcome=requested_outcome,
                authorized=authorized,
                recommended_action=recommended_action,
                summary=summary,
                observed_evidence=assessment.observed_evidence,
            )

        return asyncio.run(complete())

    def finish_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = HierarchicalTaskFinishInput.model_validate(payload)
        with self._pipeline_lock:
            verification = None
            if request.outcome in {"done", "failed"}:
                if request.observation_id is None:
                    raise ValueError(
                        "observation_id is required for independent global finish verification"
                    )
                observation = self._observations.get(request.observation_id)
                if observation is None:
                    raise ValueError(
                        f"unknown or expired observation_id: {request.observation_id}"
                    )
                managed_task_active = bool(
                    self._pending_global_goal
                    or self._planner_required
                    or self._semantic_route_plan is not None
                )
                finish_transition_ready = bool(
                    self._semantic_route_plan is not None
                    and self._semantic_route_plan.phase == "verify_goal"
                )
                if (
                    request.outcome == "done"
                    and managed_task_active
                    and not finish_transition_ready
                ):
                    return {
                        "ok": False,
                        "status": "finish_not_ready",
                        "outcome": request.outcome,
                        "memory_cleared": False,
                        "finish_verification": {
                            "status": "not_requested",
                            "reason": (
                                "The active Agentic Plan has not completed its final "
                                "subtask. Submit model Self-Reflector evidence through "
                                "advance_navigation before requesting global finish."
                            ),
                        },
                        "next_call_contract": self._next_call_contract(
                            request.observation_id
                        ),
                    }
                verification = self._verify_global_finish(
                    requested_outcome=request.outcome,
                    observation=observation,
                )
                if verification is not None and not verification.authorized:
                    return {
                        "ok": False,
                        "status": "finish_rejected",
                        "outcome": request.outcome,
                        "memory_cleared": False,
                        "finish_verification": verification.model_dump(mode="json"),
                    }
            finished_task_id = self.task_id
            finished_goal = (
                self._semantic_route_plan.global_goal
                if self._semantic_route_plan is not None
                else self._pending_global_goal or "Completed navigation task"
            )
            final_memory_update: dict[str, object] | None = None
            if self.scene_memory_enabled and request.scene_memory is not None:
                if request.observation_id is None:
                    raise ValueError("observation_id is required with final scene_memory")
                observation = self._observations.get(request.observation_id)
                if observation is None:
                    raise ValueError(f"unknown or expired observation_id: {request.observation_id}")
                final_memory_update = self._scene_memory.record_visit(
                    observation,
                    request.scene_memory,
                    selected_anchor="TASK_END",
                )
                self._scene_memory.record_result(
                    str(final_memory_update["scene_id"]),
                    status=request.outcome,
                    reached=request.outcome == "done",
                )
            summary = self._scene_memory.summary()
            explored_location_count = self.navigation_memory.place_count
            navigation_memory_summary = self.navigation_memory.summary()
            self.task_id = self._new_task_id()
            self._scene_memory = TaskSceneMemory(self.task_id)
            self.navigation_memory.start_task(task_id=self.task_id, episode_id=self.episode_id)
            self._topology_loaded_for_task = False
            self._topology_state = None
            self._semantic_route_plan = None
            self._active_target_evidence = None
            self._pending_global_goal = ""
            self._planner_required = False
            self._observations.clear()
            self._load_topology_for_task()
            result = {
                "ok": True,
                "outcome": request.outcome,
                "finished_task_id": finished_task_id,
                "memory_cleared": True,
                "memory_summary": summary,
                "navigation_memory_summary": navigation_memory_summary,
                "explored_location_count": explored_location_count,
                "final_memory_update": final_memory_update,
                "finish_verification": (
                    verification.model_dump(mode="json")
                    if verification is not None
                    else {"status": "not_configured"}
                ),
                "next_task_id": self.task_id,
                "next_call_contract": self._next_call_contract(None),
            }
            self._latest_pipeline = {
                "ok": True,
                "status": request.outcome,
                "profile": "hierarchical",
                "phase": request.outcome,
                "instruction": finished_goal,
                "finish_required": False,
                "finished_task_id": finished_task_id,
                "finish_verification": result["finish_verification"],
                "next_call_contract": self._next_call_contract(None),
            }
            return result

    def reset(self) -> dict[str, Any]:
        runtime = self._active_agent_runtime
        if runtime is not None:
            runtime.cancel("environment_reset")
        with self._pipeline_lock:
            self.adapter.close()
            self.adapter.start()
            self.episode_id = self._new_episode_id()
            self.task_id = self._new_task_id()
            self._scene_memory = TaskSceneMemory(self.task_id)
            self.navigation_memory.start_task(task_id=self.task_id, episode_id=self.episode_id)
            self._topology_loaded_for_task = False
            self._topology_state = None
            self._semantic_route_plan = None
            self._active_target_evidence = None
            self._pending_global_goal = ""
            self._planner_required = False
            self.observation_index = 0
            self._observations.clear()
            self._latest_observation = None
            self._latest_pipeline = None
            topology = self._load_topology_for_task()
            observation = self._capture_observation()
            route_graph = self.navigation_memory.route_graph()
        with self._agent_task_lock:
            self._agent_task_state = {"status": "idle"}
        return {
            "ok": True,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "observation_id": observation.trace.observation_id,
            "topology": topology,
            "route_graph": route_graph,
        }

    def _load_topology_for_task(self) -> dict[str, object]:
        if self._topology_loaded_for_task:
            return dict(self._topology_state or {})
        if self._topology_model is None and self._topology_provider is None:
            self._topology_state = {"status": "unsupported", "loaded": False}
            self._topology_loaded_for_task = True
            return dict(self._topology_state)
        if self._topology_model is None:
            if self._topology_cache is None or self._topology_provider is None:
                raise RuntimeError("topology provider requires a topology cache")
            grid = self._topology_provider.load_grid()
            topology, cache_hit = self._topology_cache.get_or_build(grid)
            self._topology_model = topology
        else:
            topology = self._topology_model
            cache_hit = True
        materialized = self.navigation_memory.record_backbone_topology(topology)
        semantic_map = self._apply_frozen_semantic_map(topology)
        self._topology_state = {
            "status": "loaded",
            "loaded": True,
            "frozen": True,
            "cache_hit": cache_hit,
            **materialized,
            "semantic_map": semantic_map,
        }
        self._topology_loaded_for_task = True
        return dict(self._topology_state)

    def semantic_map(self) -> dict[str, Any]:
        if self._semantic_map_store is None:
            return {"ok": True, "status": "disabled", "loaded": False, "nodes": {}}
        if not self._topology_loaded_for_task:
            self._load_topology_for_task()
        if self._topology_model is None:
            raise RuntimeError("semantic map requires a loaded topology")
        semantic_map = self._semantic_map_model
        if semantic_map is None:
            return {"ok": True, **(self._semantic_map_state or {})}
        return {
            "ok": True,
            "status": "loaded",
            "frozen": True,
            "path": str(self._semantic_map_store.path_for(semantic_map.map_id)),
            "semantic_map": semantic_map.model_dump(mode="json"),
            "node_locations": (self._topology_state or {}).get("node_locations", {}),
            "rebound": (self._semantic_map_state or {}).get("rebound"),
        }

    def reload_semantic_map(self) -> dict[str, Any]:
        with self._pipeline_lock:
            current = self.semantic_map()
            return {
                **current,
                "reloaded": False,
                "reason": "semantic backbone is frozen; restart the service to load map edits",
                "route_graph": self.navigation_memory.route_graph(),
            }

    def search_semantic_map(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = SemanticMapSearchInput.model_validate(payload)
        results = self.navigation_memory.search_semantic_nodes(
            request.query,
            limit=request.limit,
        )
        return {
            "ok": True,
            "query": request.query,
            "count": len(results),
            "results": results,
        }

    def _load_semantic_map(self, topology: BackboneTopology) -> dict[str, object]:
        self._semantic_map_load_attempted = True
        if self._semantic_map_store is None:
            self._semantic_map_state = {"status": "disabled", "loaded": False}
            return dict(self._semantic_map_state)
        path = self._semantic_map_store.path_for(topology.map_id)
        try:
            if self._topology_cache is None:
                raise RuntimeError("semantic map requires a topology cache")
            semantic_map, rebound = self._semantic_map_store.load_for_topology(
                topology,
                self._topology_cache,
                persist_rebind=False,
            )
            self._semantic_map_model = semantic_map
            applied = self.navigation_memory.apply_semantic_map(semantic_map)
            self._semantic_map_state = {
                "status": "loaded",
                "loaded": True,
                "frozen": True,
                "path": str(path),
                "rebound": rebound,
                **applied,
            }
        except ValueError as exc:
            self._semantic_map_state = {
                "status": "invalid",
                "loaded": False,
                "path": str(path),
                "error": str(exc),
            }
        return dict(self._semantic_map_state)

    def _apply_frozen_semantic_map(self, topology: BackboneTopology) -> dict[str, object]:
        if not self._semantic_map_load_attempted:
            return self._load_semantic_map(topology)
        if self._semantic_map_model is None:
            return dict(self._semantic_map_state or {})
        applied = self.navigation_memory.apply_semantic_map(self._semantic_map_model)
        self._semantic_map_state = {
            **(self._semantic_map_state or {}),
            "status": "loaded",
            "loaded": True,
            "frozen": True,
            **applied,
        }
        return dict(self._semantic_map_state)

    def dispatch(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        routes = {
            ("GET", "/health"): self.health,
            ("GET", "/capabilities"): self.capabilities,
            ("GET", "/tasks/agent/status"): self.agent_task_status,
            ("GET", "/relay/live"): self.relay_live,
            ("GET", "/relay/state"): self.relay_state,
            ("GET", "/semantic-map"): self.semantic_map,
            ("POST", "/observe"): self.observe,
            ("POST", "/decision/begin"): lambda: self.begin_navigation(payload),
            ("POST", "/decision/advance"): lambda: self.advance_navigation(payload),
            ("POST", "/tasks/agent/start"): lambda: self.start_agent_task(payload),
            ("POST", "/memory/visual-recall"): lambda: self.recall_visual_memory(payload),
            ("POST", "/semantic-map/reload"): self.reload_semantic_map,
            ("POST", "/semantic-map/search"): lambda: self.search_semantic_map(payload),
            ("POST", "/skills/anchor-grounding"): lambda: self.run_anchor_grounding(payload),
            ("POST", "/skills/pixel-to-point"): lambda: self.run_pixel_to_point(payload),
            ("POST", "/skills/path-planning"): lambda: self.run_path_planning(payload),
            ("POST", "/skills/path-execution"): lambda: self.run_path_execution(payload),
            ("POST", "/skills/viewpoint-adjustment"): lambda: self.adjust_viewpoint(payload),
            ("POST", "/skills/localization-map"): lambda: self.inspect_localization_map(payload),
            ("POST", "/skills/known-location-navigate"): lambda: self.navigate_to_known_location(payload),
            ("POST", "/skills/hierarchical-navigate"): lambda: self.navigate_to_anchor(payload),
            ("POST", "/task/finish"): lambda: self.finish_task(payload),
            ("POST", "/stop"): self.stop_agent_task,
            ("POST", "/reset"): self.reset,
        }
        handler = routes.get((method, path))
        if handler is None:
            raise LookupError(f"unknown endpoint: {method} {path}")
        return handler()

    def protocol_error_payload(self, error: Exception) -> dict[str, Any]:
        observation_id = ((self._latest_observation or {}).get("trace") or {}).get(
            "observation_id"
        )
        return {
            "ok": False,
            "error": _compact_error_message(error),
            "next_call_contract": self._next_call_contract(observation_id),
            "active_target_evidence": (
                self._active_target_evidence.model_dump(mode="json")
                if self._active_target_evidence is not None
                else None
            ),
        }

    def _run_stage(
        self,
        skill_name: str,
        observation: Observation,
        inputs: dict[str, Any],
        *,
        timeout_s: float = 30.0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        skill = self.skills.get(skill_name)
        trace = observation.trace.for_skill(
            skill_run_id=f"hierarchical-{skill_name.lower()}-{uuid.uuid4().hex}",
            deadline=utc_now() + timedelta(seconds=timeout_s),
        )
        request = SkillRequest(
            trace=trace,
            skill_name=skill_name,
            command=skill_name,
            inputs=inputs,
            timeout_s=timeout_s,
        )
        event_bus = InMemoryEventBus()
        if progress_callback is not None:
            event_bus.subscribe(
                lambda event: progress_callback(dict(event.payload))
            )
        context = SkillContext(event_bus=event_bus, cancellation=CancellationToken())

        async def run() -> SkillResult:
            return await asyncio.wait_for(skill.run(request, observation, context), timeout=timeout_s)

        result = asyncio.run(run())
        return {
            "ok": result.status.value == "success",
            "skill": skill_name,
            "observation_id": observation.trace.observation_id,
            "result": result.model_dump(mode="json"),
        }

    @staticmethod
    def _pixel_guidance_progress_callback(
        callback: Callable[[DualPixelGoal | None], None] | None,
    ) -> Callable[[dict[str, Any]], None] | None:
        if callback is None:
            return None

        def handle(payload: dict[str, Any]) -> None:
            if payload.get("phase") != "pixel_guidance_ready":
                return
            pixel_goal = payload.get("pixel_goal")
            callback(
                DualPixelGoal.model_validate(pixel_goal)
                if pixel_goal is not None
                else None
            )

        return handle

    def _require_observation(self, payload: dict[str, Any]) -> Observation:
        observation_id = str(payload.get("observation_id", "")).strip()
        if not observation_id:
            raise ValueError("observation_id is required; call /observe first")
        observation = self._observations.get(observation_id)
        if observation is None:
            raise ValueError(f"unknown or expired observation_id: {observation_id}")
        return observation

    def _cache_observation(self, observation: Observation) -> None:
        payload = observation.model_dump(mode="json")
        self._latest_observation = payload
        self._observations[observation.trace.observation_id] = observation
        while len(self._observations) > 32:
            self._observations.pop(next(iter(self._observations)))

    def _decorate_semantic_map(self, observation: Observation) -> Observation:
        semantic_map_ref = self._render_semantic_map(observation)
        if semantic_map_ref is None:
            return observation
        decorated = observation.model_copy(update={"topdown_ref": semantic_map_ref})
        self._cache_observation(decorated)
        return decorated

    def _render_semantic_map(self, observation: Observation) -> str | None:
        if observation.topdown_ref is None:
            return None
        metadata = observation.metadata.get("topdown_map") or {}
        origin = metadata.get("origin_world_xz_m")
        meters_per_pixel = metadata.get("meters_per_pixel")
        if not isinstance(origin, list) or len(origin) != 2 or not meters_per_pixel:
            return observation.topdown_ref

        from PIL import Image, ImageDraw, ImageFont

        source = Path(observation.topdown_ref)
        output = source.with_name("semantic_topdown.png")
        map_image = Image.open(source).convert("RGB")
        sidebar_width = 280
        image = Image.new("RGB", (map_image.width + sidebar_width, map_image.height), (12, 18, 18))
        image.paste(map_image, (0, 0))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            font = ImageFont.load_default()
        colors = {"left": (80, 180, 255), "front": (255, 210, 70), "right": (110, 230, 140)}
        relative_yaw = {"left": math.pi / 2.0, "front": 0.0, "right": -math.pi / 2.0}
        scale = float(meters_per_pixel)
        origin_x, origin_z = float(origin[0]), float(origin[1])

        places = self.navigation_memory.places()
        for location in places[-24:]:
            px = int((location.pose.x - origin_x) / scale)
            py = int((location.pose.y - origin_z) / scale)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(50, 150, 245), outline=(255, 255, 255))
            draw.text((px + 5, py - 5), location.place_id.replace("location-", "L"), fill=(255, 255, 255), font=font)
            observations = location.semantic_observations
            if not observations:
                continue
            semantic = observations[-1]
            yaw = semantic.yaw
            for view in ("left", "front", "right"):
                labels = getattr(semantic, view)
                if not labels:
                    continue
                angle = yaw + relative_yaw[view]
                endpoint_x = int(px - math.sin(angle) * 24)
                endpoint_y = int(py - math.cos(angle) * 24)
                color = colors[view]
                draw.line((px, py, endpoint_x, endpoint_y), fill=color, width=2)
                label = view[0].upper()
                draw.ellipse(
                    (endpoint_x - 7, endpoint_y - 7, endpoint_x + 7, endpoint_y + 7),
                    fill=(12, 18, 18),
                    outline=color,
                )
                draw.text((endpoint_x - 3, endpoint_y - 5), label, fill=color, font=font)

        sidebar_x = map_image.width
        draw.line((sidebar_x, 0, sidebar_x, image.height), fill=(50, 65, 64), width=1)
        draw.text((sidebar_x + 10, 10), "SEMANTIC LOCATIONS", fill=(235, 242, 239), font=font)
        row_y = 30
        for location in places[-8:]:
            semantic = location.semantic_observations[-1] if location.semantic_observations else None
            location_id = location.place_id.replace("location-", "L")
            header = (
                f"{location_id}  x={location.pose.x:.1f} z={location.pose.y:.1f}  "
                f"visits={location.visit_count}"
            )
            draw.text((sidebar_x + 10, row_y), header, fill=(110, 195, 255), font=font)
            row_y += 13
            for view in ("left", "front", "right"):
                labels = "/".join(getattr(semantic, view)[:2]) if semantic is not None else "-"
                labels = labels or "-"
                line = f"{view[0].upper()}: {labels}"[:42]
                draw.text((sidebar_x + 16, row_y), line, fill=colors[view], font=font)
                row_y += 12
            draw.line((sidebar_x + 10, row_y + 2, image.width - 10, row_y + 2), fill=(35, 48, 47))
            row_y += 8
            if row_y + 52 > image.height:
                break
        image.save(output)
        return str(output)

    @staticmethod
    def _image_width(observation: Observation) -> int:
        return int((observation.metadata.get("camera") or {}).get("output_resolution", [640, 480])[0])

    @staticmethod
    def _image_height(observation: Observation) -> int:
        return int((observation.metadata.get("camera") or {}).get("output_resolution", [640, 480])[1])

    def _trace(self) -> TraceContext:
        now = utc_now()
        index = self.observation_index
        self.observation_index += 1
        return TraceContext(
            episode_id=self.episode_id,
            task_id=self.task_id,
            plan_revision=0,
            skill_run_id=f"hierarchical-observe-{index}",
            observation_id=f"hierarchical-obs-{index}",
            frame_timestamp=now,
            pose_timestamp=now,
            deadline=now + timedelta(seconds=30),
        )

    @staticmethod
    def _new_episode_id() -> str:
        return f"hierarchical-{utc_now().strftime('%Y%m%dT%H%M%S%f')}"

    @staticmethod
    def _new_task_id() -> str:
        return f"hierarchical-task-{utc_now().strftime('%Y%m%dT%H%M%S%f')}"


def make_handler(service: HierarchicalNavigationService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def log_message(self, format: str, *args: object) -> None:
            if self.path not in {"/relay/live", "/relay/state"}:
                print(f"[hierarchical] {self.address_string()} {format % args}")

        def _handle(self, method: str) -> None:
            try:
                payload = self._read_json() if method == "POST" else {}
                self._write_json(200, service.dispatch(method, self.path, payload))
            except ValueError as exc:
                self._write_json(400, service.protocol_error_payload(exc))
            except LookupError as exc:
                self._write_json(404, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": str(exc)})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical anchor navigation service for Habitat-GS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--scene-id", default="1st_floor_exh")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/jianghaoge/habitat-gs/scenes/datasets/1st_floor_exh_collision_v1"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--start-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--start-yaw-deg", type=float)
    parser.add_argument("--agentos-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--omninav-url", default=os.environ.get("OMNINAV_URL"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=120.0)
    parser.add_argument("--vfov", type=float, default=90.0)
    parser.add_argument("--scene-memory", action="store_true")
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=Path("runtime/agent_memory/hierarchical.sqlite3"),
    )
    parser.add_argument("--environment-id")
    parser.add_argument(
        "--topology-cache-dir",
        type=Path,
        default=Path("runtime/topology_cache"),
    )
    parser.add_argument(
        "--semantic-map-dir",
        type=Path,
        default=Path("runtime/semantic_maps"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path("logs/hierarchical") / timestamp).resolve()
    raw_adapter = HabitatGSAdapter(
        HabitatGSConfig(
            dataset_root=args.dataset_root,
            scene_id=args.scene_id,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            hfov=args.hfov,
            vfov=args.vfov,
            enable_depth=True,
            sensor_views=HabitatGSAdapter.SENSOR_ORDER,
            gpu_device_id=args.gpu_device_id,
            start_position=tuple(args.start_position) if args.start_position is not None else None,
            start_yaw_deg=args.start_yaw_deg,
        )
    )
    adapter = ThreadBoundHabitatAdapter(raw_adapter)
    adapter.start()
    llm_config = load_llm_config(args.agentos_config)
    skills = SkillRegistry()
    provider_factory = lambda: build_provider_from_config(llm_config, active_provider="qwen")
    skills.register(AnchorGroundingSkill(provider_factory))
    skills.register(
        VisualGroundingSkill(
            lambda: QwenVisualGroundingProvider(provider_factory())
        )
    )
    skills.register(PixelToPointSkill(adapter))
    skills.register(PathPlanningSkill(adapter))
    skills.register(PathExecutionSkill(adapter))
    skills.register(AnchorGenerationSkill(provider_factory, adapter))
    skills.register(PointNavigationSkill(adapter))
    skills.register(ViewpointAdjustmentSkill(adapter))
    skills.register(LocalizationMapSkill())
    environment_id = args.environment_id or f"habitat:{args.scene_id}"
    memory_store = SQLiteGraphMemory(args.memory_db)
    topology_provider = HabitatNavmeshMapProvider(
        run_serialized=adapter.run_serialized,
        cache_dir=args.topology_cache_dir.resolve(),
        resolution_px=raw_adapter.topdown_resolution,
    )
    semantic_map_store = SemanticMapStore(args.semantic_map_dir.resolve())
    service = HierarchicalNavigationService(
        adapter,
        skills,
        scene_memory_enabled=args.scene_memory,
        environment_id=environment_id,
        topology_provider=topology_provider,
        semantic_map_store=semantic_map_store,
        semantic_node_provider_factory=provider_factory,
        finish_verifier_provider_factory=provider_factory,
        agent_provider_factory=provider_factory,
        omninav_endpoint_url=args.omninav_url,
    )
    service.navigation_memory = AgentNavigationMemory(
        memory_store,
        environment_id=environment_id,
        task_id=service.task_id,
        episode_id=service.episode_id,
    )
    topology = service.load_environment_map()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"CEE-AgentOS hierarchical navigation: http://{args.host}:{args.port}")
    print(f"Artifacts: {output_dir}")
    print(
        "Frozen semantic backbone: "
        f"{topology.get('node_count', 0)} nodes / {topology.get('edge_count', 0)} edges"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        adapter.shutdown()
        memory_store.close()


if __name__ == "__main__":
    main()
