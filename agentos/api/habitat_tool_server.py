from __future__ import annotations

import argparse
import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from agentos.context.manager import BoundedContextManager
from agentos.harness.verifier import Verifier
from agentos.core.runtime import AgentRuntime, RuntimeResult
from agentos.core.events import InMemoryEventBus
from agentos.evolution import TraceDrivenSkillEvolvement
from agentos.embodiments.habitat_gs import HabitatGSAdapter, HabitatGSConfig
from agentos.memory import GraphMemoryStore, MemoryWriter, SQLiteGraphMemory, observation_node_id
from agentos.planning import (
    BoundedTrajectoryMemory,
    OmniNavSupervisor,
    SceneConditionedPlanner,
    TrajectoryMemoryFrame,
    persist_trajectory_memory,
)
from agentos.providers.config import build_provider_from_config, load_llm_config
from agentos.schemas.base import utc_now
from agentos.schemas.enums import EventType, RuntimeState
from agentos.schemas.events import Event
from agentos.schemas.memory import MemoryEdge, MemoryEdgeType, MemoryNode, MemoryNodeType, MemoryQuery
from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.trace import TraceContext
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.habitat_navigation import HabitatNavigationSkill
from agentos.skills.navigation import OmniNavSkill, PersistentOmniNavSkill
from agentos.skills.registry import SkillRegistry
from agentos.verification import LLMFinishVerifier, LLMSkillVerifier, ProgressRuntimeVerifier


class ThreadBoundHabitatAdapter:
    """Keep EGL and Habitat-Sim calls on the thread that created them."""

    def __init__(self, adapter: HabitatGSAdapter) -> None:
        self._adapter = adapter
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="habitat-sim")

    @property
    def name(self) -> str:
        return self._adapter.name

    @property
    def sim(self) -> Any:
        return self._adapter.sim

    def start(self) -> None:
        self._executor.submit(self._adapter.start).result()

    async def observe(self, trace: TraceContext) -> Observation:
        future = self._executor.submit(asyncio.run, self._adapter.observe(trace))
        return await asyncio.wrap_future(future)

    async def observe_live(self, trace: TraceContext) -> Observation:
        future = self._executor.submit(asyncio.run, self._adapter.observe_live(trace))
        return await asyncio.wrap_future(future)

    async def apply_skill_result(self, result: SkillResult) -> dict[str, object]:
        future = self._executor.submit(asyncio.run, self._adapter.apply_skill_result(result))
        return await asyncio.wrap_future(future)

    async def pixel_to_world(self, **kwargs: Any) -> dict[str, object]:
        future = self._executor.submit(asyncio.run, self._adapter.pixel_to_world(**kwargs))
        return await asyncio.wrap_future(future)

    async def world_to_pixel(self, **kwargs: Any) -> dict[str, object] | None:
        future = self._executor.submit(asyncio.run, self._adapter.world_to_pixel(**kwargs))
        return await asyncio.wrap_future(future)

    async def plan_path(self, world_point: list[float]) -> dict[str, object]:
        future = self._executor.submit(asyncio.run, self._adapter.plan_path(world_point))
        return await asyncio.wrap_future(future)

    async def execute_world_path(self, world_path: list[list[float]], **kwargs: Any) -> dict[str, object]:
        future = self._executor.submit(asyncio.run, self._adapter.execute_world_path(world_path, **kwargs))
        return await asyncio.wrap_future(future)

    async def follow_world_path_step(self, world_path: list[list[float]], **kwargs: Any) -> dict[str, object]:
        future = self._executor.submit(asyncio.run, self._adapter.follow_world_path_step(world_path, **kwargs))
        return await asyncio.wrap_future(future)

    async def follow_navmesh_path_step(self, world_path: list[list[float]], **kwargs: Any) -> dict[str, object]:
        future = self._executor.submit(asyncio.run, self._adapter.follow_navmesh_path_step(world_path, **kwargs))
        return await asyncio.wrap_future(future)

    async def observe_keyframe(self, trace: TraceContext, run_id: str, frame_index: int) -> Observation:
        future = self._executor.submit(asyncio.run, self._adapter.observe_keyframe(trace, run_id, frame_index))
        return await asyncio.wrap_future(future)

    def step(self, action: str) -> dict[str, object]:
        return self._executor.submit(self._adapter.step, action).result()

    def run_serialized(self, operation: Callable[[HabitatGSAdapter], Any]) -> Any:
        """Run an injected integration operation on Habitat's owning thread."""
        return self._executor.submit(operation, self._adapter).result()

    def close(self) -> None:
        self._executor.submit(self._adapter.close).result()

    def shutdown(self) -> None:
        self.close()
        self._executor.shutdown(wait=True)


class HabitatToolService:
    """Small, serialized HTTP boundary around one persistent Habitat instance."""

    ALLOWED_ACTIONS = frozenset({"forward", "back", "left", "right", "stop"})

    def __init__(
        self,
        adapter: HabitatGSAdapter,
        *,
        max_actions: int = 8,
        skills: SkillRegistry | None = None,
        memory: GraphMemoryStore | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.max_actions = max_actions
        self.skills = skills or SkillRegistry()
        self.memory = memory
        self.agent_config = agent_config
        self.episode_id = "hermes-habitat-episode"
        self.observation_index = 0
        self._observations: dict[str, Observation] = {}
        self._latest_observation: dict[str, Any] | None = None
        self._latest_omninav: dict[str, Any] | None = None
        self._task_lock = threading.Lock()
        self._task_cancel: threading.Event | None = None
        self._active_runtime: AgentRuntime | None = None
        self._task_state: dict[str, Any] = {"status": "idle"}

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "cee-agentos-habitat",
            "embodiment": self.adapter.name,
            "simulator_ready": self.adapter.sim is not None,
            "allowed_actions": sorted(self.ALLOWED_ACTIONS),
            "max_actions_per_call": self.max_actions,
            "memory_ready": self.memory is not None,
            "skill_count": len(self.skills.names()),
            "agentos_ready": self.agent_config is not None and self.memory is not None,
            "agent_harness": {
                "main_agent": "server_side",
                "skill_runner": "isolated_context",
                "verifier": {
                    "component": "Verifier",
                    "stages": ["runtime", "skill_level", "finish_time"],
                },
                "context_management": "bounded_retrieval_and_compression",
                "skill_evolvement": "trace_driven_gated_assets",
                "hermes_role": "task_client",
            },
        }

    def observe(self) -> dict[str, Any]:
        observation = asyncio.run(self.adapter.observe(self._trace()))
        payload = observation.model_dump(mode="json")
        self._cache_observation(observation, payload)
        self._record_observation(payload)
        return {"ok": True, "observation": payload}

    def capabilities(self) -> dict[str, Any]:
        return {
            "ok": True,
            "embodiment": self.adapter.name,
            "skills": [descriptor.model_dump(mode="json") for descriptor in self.skills.catalog()],
        }

    def relay_state(self) -> dict[str, Any]:
        """Return the latest snapshots without advancing the simulator."""
        return {
            "ok": True,
            "health": self.health(),
            "observation": self._latest_observation,
            "omninav": self._latest_omninav,
            "navigation_task": self.navigation_task_status(),
        }

    def relay_live(self) -> dict[str, Any]:
        """Capture the current Habitat frame independently of model inference."""
        observation = asyncio.run(self.adapter.observe_live(self._trace()))
        return {
            "ok": True,
            "health": self.health(),
            "observation": observation.model_dump(mode="json"),
            "omninav": self._latest_omninav,
            "navigation_task": self.navigation_task_status(),
        }

    def navigation_task_status(self) -> dict[str, Any]:
        with self._task_lock:
            return dict(self._task_state)

    def start_agent_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("instruction is required")
        if self.agent_config is None or self.memory is None:
            raise ValueError("AgentOS is not configured; start with --agentos-config and --memory-db")
        max_steps = int(payload.get("max_steps", 1000))
        if max_steps < 1 or max_steps > 10000:
            raise ValueError("max_steps must be between 1 and 10000")
        timeout_s = float(payload.get("timeout_s", 1800.0))
        if timeout_s <= 0.0 or timeout_s > 7200.0:
            raise ValueError("timeout_s must be between 0 and 7200 seconds")

        with self._task_lock:
            if self._task_state.get("status") in {"starting", "running"}:
                raise ValueError("an AgentOS task is already running")
            task_id = f"agent-task-{uuid.uuid4().hex}"
            cancel = threading.Event()
            self._task_cancel = cancel
            self._task_state = {
                "ok": True,
                "task_id": task_id,
                "instruction": instruction,
                "status": "starting",
                "agent_state": RuntimeState.PLAN.value,
                "step": 0,
                "max_steps": max_steps,
                "distance_m": 0.0,
                "plan_revision": 0,
                "agent_harness": {
                    "main_agent": "server_side",
                    "skill_runner": "isolated_context",
                    "verifier": {
                        "component": "Verifier",
                        "stages": ["runtime", "skill_level", "finish_time"],
                    },
                    "context_management": "bounded_retrieval_and_compression",
                    "skill_evolvement": "trace_driven_gated_assets",
                    "hermes_role": "task_client",
                },
                "started_at": utc_now().isoformat(),
            }

        worker = threading.Thread(
            target=self._run_agent_task,
            args=(task_id, instruction, max_steps, timeout_s, cancel, payload),
            name=f"agent-task-{task_id[-8:]}",
            daemon=True,
        )
        worker.start()
        return self.navigation_task_status()

    def _run_agent_task(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        timeout_s: float,
        cancel: threading.Event,
        payload: dict[str, Any],
    ) -> None:
        try:
            asyncio.run(
                self._run_agent_task_async(
                    task_id,
                    instruction,
                    max_steps,
                    timeout_s,
                    cancel,
                    payload,
                )
            )
        except Exception as exc:
            self._finish_task(task_id, "failed", f"{type(exc).__name__}: {exc}")

    async def _run_agent_task_async(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        timeout_s: float,
        cancel: threading.Event,
        payload: dict[str, Any],
    ) -> None:
        assert self.agent_config is not None
        assert self.memory is not None
        try:
            inner = self.skills.get("NavigationSkill")
        except KeyError as exc:
            raise ValueError("OmniNav NavigationSkill is not configured") from exc
        if not isinstance(inner, OmniNavSkill):
            raise ValueError("registered NavigationSkill is not the OmniNav HTTP skill")

        provider = build_provider_from_config(self.agent_config)
        event_bus = InMemoryEventBus()
        event_bus.subscribe(lambda event: self._on_agent_event(task_id, event))

        async def observe(trace: TraceContext) -> Observation:
            return await self._observe_for_agent(trace)

        async def progress(update: dict[str, Any]) -> None:
            if cancel.is_set() and self._active_runtime is not None:
                self._active_runtime.cancel("user_stop")
            self._latest_omninav = {"ok": True, "instruction": instruction, "result": {"outputs": update.get("outputs", {})}}
            self._update_task(task_id, **{key: value for key, value in update.items() if key != "outputs"})

        persistent = PersistentOmniNavSkill(
            inner=inner,
            observe=observe,
            max_steps=max_steps,
            on_progress=progress,
        )
        registry = SkillRegistry()
        registry.register(persistent)
        planner = SceneConditionedPlanner(provider, registry.catalog(), memory=self.memory)
        runtime = AgentRuntime(
            planner=planner,
            replanner=planner,
            skills=registry,
            verifier=Verifier(
                runtime=ProgressRuntimeVerifier(),
                skill=LLMSkillVerifier(provider),
                finish=LLMFinishVerifier(provider),
            ),
            context_manager=BoundedContextManager(self.memory),
            memory_writer=MemoryWriter(self.memory),
            skill_evolvement=TraceDrivenSkillEvolvement("logs/agentos_evolution/traces"),
            max_plan_revisions=int(payload.get("max_plan_revisions", 3)),
            event_bus=event_bus,
            observation_provider=observe,
        )
        self._active_runtime = runtime
        self._update_task(task_id, status="running")
        now = utc_now()
        task_trace = TraceContext(
            episode_id=self.episode_id,
            task_id=task_id,
            plan_revision=0,
            skill_run_id="agent-root",
            observation_id="agent-initial",
            frame_timestamp=now,
            pose_timestamp=now,
            deadline=now + timedelta(seconds=timeout_s),
        )
        initial = await observe(task_trace)
        task = TaskRequest(
            trace=task_trace,
            instruction=instruction,
            success_conditions=[str(item) for item in payload.get("success_conditions", [])],
            metadata={"embodiment": self.adapter.name, "agentos_entrypoint": True},
        )
        try:
            result = await runtime.run_task(task, initial)
            self._complete_agent_result(task_id, result)
        finally:
            await provider.close()
            self._active_runtime = None

    async def _observe_for_agent(self, base_trace: TraceContext) -> Observation:
        now = utc_now()
        index = self.observation_index
        self.observation_index += 1
        trace = base_trace.model_copy(
            update={
                "observation_id": f"agent-obs-{index}",
                "frame_timestamp": now,
                "pose_timestamp": now,
            }
        )
        observation = await self.adapter.observe(trace)
        payload = observation.model_dump(mode="json")
        self._cache_observation(observation, payload)
        self._record_observation(payload)
        return observation

    def _on_agent_event(self, task_id: str, event: Event) -> None:
        updates: dict[str, Any] = {
            "agent_state": event.state.value,
            "last_event": event.event_type.value,
        }
        if event.event_type in {EventType.PLAN_CREATED, EventType.PLAN_REVISED}:
            plan = event.payload.get("plan") or {}
            updates["plan"] = plan
            updates["plan_revision"] = plan.get("plan_revision", 0)
        elif event.event_type == EventType.SKILL_STARTED:
            updates["active_skill"] = event.payload.get("skill_name")
        elif event.event_type == EventType.SKILL_CONTEXT_OPENED:
            updates["skill_runner"] = {
                "status": "running",
                "session": event.payload.get("session"),
            }
        elif event.event_type == EventType.SKILL_CONTEXT_CLOSED:
            updates["skill_runner"] = {
                "status": "returned",
                "session": event.payload.get("session"),
                "outcome": event.payload.get("outcome"),
            }
        elif event.event_type in {
            EventType.RUNTIME_VERIFIED,
            EventType.SKILL_VERIFIED,
            EventType.FINISH_VERIFIED,
        }:
            updates["verification"] = event.payload.get("verification")
        self._update_task(task_id, **updates)

    def _complete_agent_result(self, task_id: str, result: RuntimeResult) -> None:
        status_by_state = {
            RuntimeState.DONE: "completed",
            RuntimeState.CANCELLED: "cancelled",
            RuntimeState.FAILED: "failed",
        }
        finish = result.finish_verification.model_dump(mode="json") if result.finish_verification else None
        reason = result.finish_verification.summary if result.finish_verification else result.state.value
        self._update_task(
            task_id,
            status=status_by_state.get(result.state, "failed"),
            agent_state=result.state.value,
            reason=reason,
            plan=result.plan.model_dump(mode="json") if result.plan else None,
            skill_results=[item.model_dump(mode="json") for item in result.skill_results],
            main_agent_results=[item.model_dump(mode="json") for item in result.main_agent_results],
            finish_verification=finish,
            finished_at=utc_now().isoformat(),
        )

    def start_omninav_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("instruction is required")
        if self.agent_config is None or self.memory is None:
            raise ValueError("Agent-supervised OmniNav requires --agentos-config and --memory-db")
        max_steps = int(payload.get("max_steps", 1000))
        if max_steps < 1 or max_steps > 10000:
            raise ValueError("max_steps must be between 1 and 10000")
        timeout_s = float(payload.get("timeout_s", 30.0))
        if timeout_s <= 0.0 or timeout_s > 120.0:
            raise ValueError("timeout_s must be between 0 and 120 seconds")
        max_skill_steps = int(payload.get("max_omninav_steps_per_skill", 64))
        if max_skill_steps < 1 or max_skill_steps > 1000:
            raise ValueError("max_omninav_steps_per_skill must be between 1 and 1000")

        with self._task_lock:
            if self._task_state.get("status") in {"starting", "running"}:
                raise ValueError("an OmniNav navigation task is already running")
            task_id = f"omninav-task-{uuid.uuid4().hex}"
            cancel = threading.Event()
            self._task_cancel = cancel
            self._task_state = {
                "ok": True,
                "task_id": task_id,
                "instruction": instruction,
                "status": "starting",
                "agent_phase": "initial_planning",
                "agent_calls": 0,
                "step": 0,
                "max_steps": max_steps,
                "max_omninav_steps_per_skill": max_skill_steps,
                "distance_m": 0.0,
                "trajectory_memory": [],
                "started_at": utc_now().isoformat(),
            }

        worker = threading.Thread(
            target=self._run_omninav_task,
            args=(task_id, instruction, max_steps, timeout_s, cancel, payload),
            name=f"omninav-task-{task_id[-8:]}",
            daemon=True,
        )
        worker.start()
        return self.navigation_task_status()

    def _run_omninav_task(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        timeout_s: float,
        cancel: threading.Event,
        payload: dict[str, Any],
    ) -> None:
        try:
            asyncio.run(
                self._run_omninav_task_async(
                    task_id,
                    instruction,
                    max_steps,
                    timeout_s,
                    cancel,
                    payload,
                )
            )
        except Exception as exc:
            self.adapter.step("stop")
            self._finish_task(task_id, "failed", f"{type(exc).__name__}: {exc}")

    async def _run_omninav_task_async(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        timeout_s: float,
        cancel: threading.Event,
        payload: dict[str, Any],
    ) -> None:
        assert self.agent_config is not None
        assert self.memory is not None
        provider = build_provider_from_config(self.agent_config)
        supervisor = OmniNavSupervisor(provider)
        trajectory_memory = BoundedTrajectoryMemory()
        distance_m = 0.0
        agent_calls = 0
        total_frame_index = 0
        skill_run_index = 0
        max_skill_steps = int(payload.get("max_omninav_steps_per_skill", 64))
        try:
            self._update_task(task_id, status="running", agent_phase="initial_planning")
            observation = await self._observe_for_omni_task(task_id, 0, timeout_s)
            plan = await supervisor.initialize(
                instruction,
                observation,
                [str(item) for item in payload.get("success_conditions", [])],
            )
            agent_calls += 1
            self._update_task(
                task_id,
                agent_calls=agent_calls,
                plan=plan.model_dump(mode="json"),
            )
            if plan.decision == "stop":
                self.adapter.step("stop")
                self._finish_task(task_id, "completed", "agent_initial_stop")
                return
            assert plan.initial_subtask is not None
            assert plan.initial_visible_anchor is not None
            assert plan.initial_anchor_view is not None
            current_subtask = plan.initial_subtask
            current_conditions = plan.initial_subtask_success_conditions
            current_anchor = plan.initial_visible_anchor
            current_anchor_view = plan.initial_anchor_view
            self._update_task(
                task_id,
                active_subtask=current_subtask,
                visible_anchor=current_anchor,
                anchor_view=current_anchor_view,
                subtask_success_conditions=current_conditions,
            )

            while total_frame_index < max_steps:
                if cancel.is_set():
                    self.adapter.step("stop")
                    self._finish_task(task_id, "cancelled", "user_stop")
                    return

                self._update_task(
                    task_id,
                    agent_phase="omninav",
                    skill_run_index=skill_run_index,
                )
                local_observations: list[tuple[int, Observation]] = [(total_frame_index, observation)]
                local_actions: list[str] = []
                local_distance_m = 0.0
                local_collisions = 0
                local_stop = False
                last_outputs: dict[str, Any] = {}
                local_limit = min(max_skill_steps, max_steps - total_frame_index)

                for local_frame_index in range(local_limit):
                    if cancel.is_set():
                        self.adapter.step("stop")
                        self._finish_task(task_id, "cancelled", "user_stop")
                        return
                    response: dict[str, Any] = {}
                    for attempt in range(1, 4):
                        try:
                            response = await self._run_omninav_async(
                                {
                                    "instruction": current_subtask,
                                    "observation_id": observation.trace.observation_id,
                                    "reset": local_frame_index == 0,
                                    "frame_index": total_frame_index,
                                    "timeout_s": timeout_s,
                                },
                                observation=observation,
                            )
                            if not response.get("ok"):
                                error = str((response.get("result") or {}).get("error") or "OmniNav inference failed")
                                raise RuntimeError(error)
                            self._update_task(task_id, transient_error=None, retry=None)
                            break
                        except Exception as exc:
                            self._update_task(task_id, transient_error=str(exc), retry=f"{attempt}/3")
                            if attempt == 3:
                                raise

                    last_outputs = (response.get("result") or {}).get("outputs") or {}
                    execution = last_outputs.get("execution") or {}
                    action = str(execution.get("action") or "unknown")
                    step_distance = max(0.0, float(execution.get("distance_m", 0.0)))
                    collided = bool(execution.get("collided", False))
                    local_actions.append(action)
                    local_distance_m += step_distance
                    local_collisions += int(collided)
                    distance_m += step_distance
                    total_frame_index += 1
                    observation = await self._observe_for_omni_task(task_id, total_frame_index, timeout_s)
                    local_observations.append((total_frame_index, observation))
                    local_stop = bool(last_outputs.get("omninav_stop"))
                    self._update_task(
                        task_id,
                        step=total_frame_index,
                        distance_m=round(distance_m, 4),
                        observation_id=observation.trace.observation_id,
                        action=action,
                        collided=collided,
                        omninav_stop=local_stop,
                        omninav_stop_scope="local_evidence_only",
                        local_skill_step=local_frame_index + 1,
                    )
                    if local_stop:
                        break

                stop_reason = (
                    "omninav_local_stop"
                    if local_stop
                    else ("global_step_limit" if total_frame_index >= max_steps else "local_step_limit")
                )
                keyframes = self._select_trajectory_keyframes(local_observations)
                trajectory = last_outputs.get("trajectory") or []
                last_execution = last_outputs.get("execution") or {}
                action_counts = {action: local_actions.count(action) for action in sorted(set(local_actions))}
                run_summary = {
                    "skill_run_index": skill_run_index,
                    "subtask": current_subtask,
                    "visible_anchor": current_anchor,
                    "anchor_view": current_anchor_view,
                    "steps": len(local_actions),
                    "stop_reason": stop_reason,
                    "omninav_local_stop": local_stop,
                    "distance_m": round(local_distance_m, 4),
                    "collision_count": local_collisions,
                    "action_counts": action_counts,
                    "first_action": local_actions[0] if local_actions else None,
                    "last_action": local_actions[-1] if local_actions else None,
                    "last_waypoint_count": len(trajectory) if isinstance(trajectory, list) else 0,
                    "last_execution": last_execution,
                    "last_model_infer_sec": last_outputs.get("model_infer_sec"),
                }
                self._update_task(task_id, agent_phase="after_omninav")
                post = await supervisor.after_omninav(
                    instruction,
                    plan,
                    current_subtask,
                    current_conditions,
                    run_summary,
                    trajectory_memory,
                    keyframes,
                    observation,
                )
                agent_calls += 1
                keyframes_by_id = {str(frame["observation_id"]): frame for frame in keyframes}
                for description in post.trajectory_descriptions:
                    frame = keyframes_by_id[description.observation_id]
                    trajectory_memory.add(
                        TrajectoryMemoryFrame(
                            frame_index=description.frame_index,
                            observation_id=description.observation_id,
                            timestamp=str(frame["timestamp"]),
                            scene_description=description.description,
                            omni_action_description=post.omni_action_description,
                            subtask=current_subtask,
                            subtask_status=post.previous_subtask_status,
                            pose=frame.get("pose"),
                        )
                    )
                persist_trajectory_memory(
                    self.memory,
                    episode_id=self.episode_id,
                    task_id=task_id,
                    trajectory_memory=trajectory_memory,
                )
                self._update_task(
                    task_id,
                    agent_calls=agent_calls,
                    skill_run_index=skill_run_index,
                    distance_m=round(distance_m, 4),
                    omninav_stop=local_stop,
                    omninav_stop_scope="local_evidence_only",
                    previous_subtask=current_subtask,
                    previous_subtask_status=post.previous_subtask_status,
                    post_decision=post.model_dump(mode="json"),
                    last_skill_run=run_summary,
                    trajectory_memory=trajectory_memory.as_json(),
                )
                if post.decision == "stop":
                    self._update_task(task_id, agent_phase="finish_verification")
                    finish = await supervisor.verify_finish(
                        instruction,
                        plan.task_success_conditions,
                        post,
                        observation,
                    )
                    agent_calls += 1
                    self._update_task(
                        task_id,
                        agent_calls=agent_calls,
                        finish_verification=finish.model_dump(mode="json"),
                    )
                    if finish.verdict == "finish":
                        self.adapter.step("stop")
                        self._finish_task(task_id, "completed", "finish_verified")
                        return
                    assert finish.next_subtask is not None
                    assert finish.next_visible_anchor is not None
                    assert finish.next_anchor_view is not None
                    current_subtask = finish.next_subtask
                    current_conditions = finish.next_subtask_success_conditions
                    current_anchor = finish.next_visible_anchor
                    current_anchor_view = finish.next_anchor_view
                elif post.decision == "failed":
                    self.adapter.step("stop")
                    self._finish_task(task_id, "failed", post.rationale_summary)
                    return
                else:
                    assert post.next_subtask is not None
                    assert post.next_visible_anchor is not None
                    assert post.next_anchor_view is not None
                    current_subtask = post.next_subtask
                    current_conditions = post.next_subtask_success_conditions
                    current_anchor = post.next_visible_anchor
                    current_anchor_view = post.next_anchor_view
                self._update_task(
                    task_id,
                    active_subtask=current_subtask,
                    visible_anchor=current_anchor,
                    anchor_view=current_anchor_view,
                    subtask_success_conditions=current_conditions,
                )
                skill_run_index += 1

            self.adapter.step("stop")
            self._finish_task(task_id, "failed", "max_steps_exhausted_without_agent_stop")
        finally:
            await provider.close()

    async def _observe_for_omni_task(self, task_id: str, frame_index: int, timeout_s: float) -> Observation:
        now = utc_now()
        index = self.observation_index
        self.observation_index += 1
        trace = TraceContext(
            episode_id=self.episode_id,
            task_id=task_id,
            plan_revision=frame_index,
            skill_run_id=f"omninav-agent-{frame_index}",
            observation_id=f"omninav-agent-obs-{index}",
            frame_timestamp=now,
            pose_timestamp=now,
            deadline=now + timedelta(seconds=timeout_s),
        )
        observation = await self.adapter.observe(trace)
        observation_payload = observation.model_dump(mode="json")
        self._cache_observation(observation, observation_payload)
        self._record_observation(observation_payload)
        return observation

    @staticmethod
    def _select_trajectory_keyframes(
        observations: list[tuple[int, Observation]],
    ) -> list[dict[str, Any]]:
        if len(observations) <= 4:
            selected = observations
        else:
            last = len(observations) - 1
            selected = [observations[round(position * last / 3)] for position in range(4)]
        return [
            {
                "frame_index": frame_index,
                "observation_id": observation.trace.observation_id,
                "timestamp": observation.trace.frame_timestamp.isoformat(),
                "pose": observation.pose.model_dump(mode="json") if observation.pose is not None else None,
                "front_ref": observation.rgb_refs["front"],
            }
            for frame_index, observation in selected
        ]

    def _update_task(self, task_id: str, **updates: Any) -> None:
        with self._task_lock:
            if self._task_state.get("task_id") != task_id:
                return
            self._task_state.update(updates)

    def _finish_task(self, task_id: str, status: str, reason: str) -> None:
        self._update_task(
            task_id,
            status=status,
            reason=reason,
            finished_at=utc_now().isoformat(),
        )

    def search_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.memory is None:
            raise ValueError("embodied memory is not configured")
        query = MemoryQuery.model_validate(payload)
        return {
            "ok": True,
            "hits": [hit.model_dump(mode="json") for hit in self.memory.search(query)],
        }

    def remember(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.memory is None:
            raise ValueError("embodied memory is not configured")
        source_observation_id = str(payload.get("source_observation_id", "")).strip()
        title = str(payload.get("title", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not source_observation_id or not title or not content:
            raise ValueError("source_observation_id, title, and content are required")
        source_id = observation_node_id(self.episode_id, source_observation_id)
        source = self.memory.get_node(source_id)
        if source is None:
            raise ValueError(f"unknown source observation: {source_observation_id}")
        requested_type = str(payload.get("node_type", MemoryNodeType.EVIDENCE.value))
        node_type = MemoryNodeType(requested_type)
        if node_type not in {MemoryNodeType.EVIDENCE, MemoryNodeType.ENTITY, MemoryNodeType.PLACE, MemoryNodeType.SEMANTIC_EVENT}:
            raise ValueError("node_type must be evidence, entity, place, or semantic_event")
        node = MemoryNode(
            node_id=f"{node_type.value}:{uuid.uuid4().hex}",
            node_type=node_type,
            title=title,
            content=content,
            source_ref=source.source_ref,
            episode_id=self.episode_id,
            metadata={"confidence": payload.get("confidence", 1.0), **dict(payload.get("metadata") or {})},
        )
        self.memory.upsert_node(node)
        self.memory.add_edge(
            MemoryEdge(
                edge_id=f"edge:{uuid.uuid4().hex}",
                source_id=node.node_id,
                target_id=source_id,
                edge_type=MemoryEdgeType.PROVENANCE,
                source_ref=source.source_ref,
            )
        )
        return {"ok": True, "node": node.model_dump(mode="json")}

    def execute_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty list")
        if len(actions) > self.max_actions:
            raise ValueError(f"at most {self.max_actions} actions are allowed per call")

        normalized = [str(action).strip().lower() for action in actions]
        unsupported = [action for action in normalized if action not in self.ALLOWED_ACTIONS]
        if unsupported:
            raise ValueError(f"unsupported actions: {unsupported}")

        steps = [self.adapter.step(action) for action in normalized]
        observation = asyncio.run(self.adapter.observe(self._trace()))
        observation_payload = observation.model_dump(mode="json")
        self._cache_observation(observation, observation_payload)
        self._record_observation(observation_payload)
        return {
            "ok": True,
            "executed_actions": normalized,
            "steps": steps,
            "observation": observation_payload,
        }

    def run_omninav(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._run_omninav_async(payload))

    async def _run_omninav_async(
        self,
        payload: dict[str, Any],
        *,
        observation: Observation | None = None,
    ) -> dict[str, Any]:
        instruction = str(payload.get("instruction", "")).strip()
        observation_id = str(payload.get("observation_id", "")).strip()
        if not instruction or not observation_id:
            raise ValueError("instruction and observation_id are required; call /observe first")
        observation = observation or self._observations.get(observation_id)
        if observation is None:
            raise ValueError(f"unknown or expired observation_id: {observation_id}")
        if observation.trace.observation_id != observation_id:
            raise ValueError("observation_id does not match the supplied observation")
        try:
            skill = self.skills.get("NavigationSkill")
        except KeyError as exc:
            raise ValueError("OmniNav NavigationSkill is not configured") from exc
        if not isinstance(skill, OmniNavSkill):
            raise ValueError("registered NavigationSkill is not the OmniNav HTTP skill")

        timeout_s = float(payload.get("timeout_s", 30.0))
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        trace = observation.trace.for_skill(
            skill_run_id=f"omninav-{uuid.uuid4().hex}",
            deadline=utc_now() + timedelta(seconds=timeout_s),
        )
        request = SkillRequest(
            trace=trace,
            skill_name=skill.name,
            command=instruction,
            inputs={
                "instruction": instruction,
                "reset": bool(payload.get("reset", False)),
                "frame_index": int(payload.get("frame_index", 0)),
            },
            timeout_s=timeout_s,
        )
        context = SkillContext(event_bus=InMemoryEventBus(), cancellation=CancellationToken())
        result = await asyncio.wait_for(skill.run(request, observation, context), timeout=timeout_s)
        response = {
            "ok": result.status.value == "success",
            "instruction": instruction,
            "observation_id": observation_id,
            "result": result.model_dump(mode="json"),
        }
        self._latest_omninav = response
        return response

    def stop(self) -> dict[str, Any]:
        with self._task_lock:
            cancel = self._task_cancel
            runtime = self._active_runtime
        if cancel is not None:
            cancel.set()
        if runtime is not None:
            runtime.cancel("user_stop")
        return {"ok": True, "step": self.adapter.step("stop"), "navigation_task": self.navigation_task_status()}

    def reset(self) -> dict[str, Any]:
        with self._task_lock:
            cancel = self._task_cancel
            runtime = self._active_runtime
        if cancel is not None:
            cancel.set()
        if runtime is not None:
            runtime.cancel("user_reset")
        self.adapter.close()
        self.adapter.start()
        self.episode_id = f"hermes-habitat-{utc_now().strftime('%Y%m%dT%H%M%S')}"
        self.observation_index = 0
        self._observations.clear()
        self._latest_observation = None
        self._latest_omninav = None
        return {"ok": True, "episode_id": self.episode_id, **self.observe()}

    def dispatch(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "GET" and path == "/health":
            return self.health()
        if method == "GET" and path == "/capabilities":
            return self.capabilities()
        if method == "GET" and path == "/relay/state":
            return self.relay_state()
        if method == "GET" and path == "/relay/live":
            return self.relay_live()
        if method == "GET" and path == "/tasks/omninav/status":
            return self.navigation_task_status()
        if method == "GET" and path == "/tasks/agent/status":
            return self.navigation_task_status()
        if method == "POST" and path == "/observe":
            return self.observe()
        if method == "POST" and path == "/execute_actions":
            return self.execute_actions(payload)
        if method == "POST" and path == "/skills/omninav":
            return self.run_omninav(payload)
        if method == "POST" and path == "/tasks/omninav/start":
            return self.start_omninav_task(payload)
        if method == "POST" and path == "/tasks/agent/start":
            return self.start_agent_task(payload)
        if method == "POST" and path == "/stop":
            return self.stop()
        if method == "POST" and path == "/reset":
            return self.reset()
        if method == "POST" and path == "/memory/search":
            return self.search_memory(payload)
        if method == "POST" and path == "/memory/remember":
            return self.remember(payload)
        raise LookupError(f"unknown endpoint: {method} {path}")

    def _cache_observation(self, observation: Observation, payload: dict[str, Any]) -> None:
        self._latest_observation = payload
        observation_id = str((payload.get("trace") or {}).get("observation_id") or "").strip()
        if not observation_id:
            return
        self._observations[observation_id] = observation
        while len(self._observations) > 16:
            self._observations.pop(next(iter(self._observations)))

    def _record_observation(self, payload: dict[str, Any]) -> None:
        if self.memory is None:
            return
        trace = payload.get("trace") or {}
        observation_id = str(trace.get("observation_id") or "").strip()
        if not observation_id:
            return
        pose = payload.get("pose")
        views = sorted((payload.get("rgb_refs") or {}).keys())
        node = MemoryNode(
            node_id=observation_node_id(self.episode_id, observation_id),
            node_type=MemoryNodeType.SOURCE,
            title=f"Observation {observation_id}",
            content=f"views={views}; pose={pose}",
            source_ref=f"observation:{observation_id}",
            episode_id=self.episode_id,
            metadata={
                "rgb_refs": payload.get("rgb_refs") or {},
                "depth_refs": payload.get("depth_refs") or {},
                "topdown_ref": payload.get("topdown_ref"),
                "pose": pose,
            },
        )
        self.memory.upsert_node(node)

    def _trace(self) -> TraceContext:
        now = utc_now()
        index = self.observation_index
        self.observation_index += 1
        return TraceContext(
            episode_id=self.episode_id,
            task_id="hermes-navigation",
            plan_revision=0,
            skill_run_id=f"hermes-tool-{index}",
            observation_id=f"obs-{index}",
            frame_timestamp=now,
            pose_timestamp=now,
            deadline=now + timedelta(seconds=30),
        )


def make_handler(service: HabitatToolService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def log_message(self, format: str, *args: object) -> None:
            if self.path in {"/relay/state", "/relay/live"}:
                return
            print(f"[habitat-tool] {self.address_string()} {format % args}")

        def _handle(self, method: str) -> None:
            try:
                payload = self._read_json() if method == "POST" else {}
                response = service.dispatch(method, self.path, payload)
                self._write_json(200, response)
            except ValueError as exc:
                self._write_json(400, {"ok": False, "error": str(exc)})
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
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose bounded CEE-AgentOS Habitat tools to Hermes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
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
    parser.add_argument(
        "--sensor-views",
        nargs="+",
        choices=HabitatGSAdapter.SENSOR_ORDER,
        default=HabitatGSAdapter.SENSOR_ORDER,
        help="RGB sensors to create (default: left front right).",
    )
    parser.add_argument("--enable-depth", action="store_true", help="Enable the optional front depth sensor.")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--memory-db", type=Path)
    parser.add_argument("--agentos-config", type=Path, help="LLM config for the unified AgentOS task endpoint.")
    parser.add_argument("--omninav-url", help="OmniNav /eval URL; enables cee_omninav_step when set.")
    parser.add_argument("--omninav-single-view", action="store_true", help="Send only the front RGB view.")
    parser.add_argument("--omninav-no-verify-tls", action="store_true", help="Disable TLS verification for testing only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path("logs/hermes_habitat") / timestamp).resolve()
    raw_adapter = HabitatGSAdapter(
        HabitatGSConfig(
            dataset_root=args.dataset_root,
            scene_id=args.scene_id,
            output_dir=output_dir,
            gpu_device_id=args.gpu_device_id,
            enable_depth=args.enable_depth,
            sensor_views=tuple(args.sensor_views),
            start_position=tuple(args.start_position) if args.start_position is not None else None,
            start_yaw_deg=args.start_yaw_deg,
        )
    )
    adapter = ThreadBoundHabitatAdapter(raw_adapter)
    adapter.start()
    skills = SkillRegistry()
    if args.omninav_url:
        skills.register(
            OmniNavSkill(
                endpoint_url=args.omninav_url,
                embodiment=adapter,
                verify_tls=not args.omninav_no_verify_tls,
                require_three_views=not args.omninav_single_view,
            )
        )
    else:
        skills.register(HabitatNavigationSkill(adapter))
    memory_path = (args.memory_db or output_dir.parent / "agentos_memory.sqlite3").resolve()
    memory = SQLiteGraphMemory(memory_path)
    agent_config = load_llm_config(args.agentos_config) if args.agentos_config else None
    service = HabitatToolService(
        adapter,
        max_actions=args.max_actions,
        skills=skills,
        memory=memory,
        agent_config=agent_config,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"CEE-AgentOS Habitat tool service: http://{args.host}:{args.port}")
    print(f"Artifacts: {output_dir.resolve()}")
    print(f"Embodied memory: {memory_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        memory.close()
        adapter.shutdown()


if __name__ == "__main__":
    main()
