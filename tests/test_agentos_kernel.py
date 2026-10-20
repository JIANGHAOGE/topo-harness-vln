from __future__ import annotations

import asyncio

from agentos.context.manager import BoundedContextManager
from agentos.core.runtime import AgentRuntime
from agentos.evolution import (
    EvoAsset,
    EvoAssetGate,
    EvoAssetStatus,
    JsonEvoAssetStore,
    TraceDrivenSkillEvolvement,
)
from agentos.memory import MemoryWriter, SQLiteGraphMemory
from agentos.harness.navigation_session import AgentOSMainAgentSession
from agentos.planning.agentos_hierarchical import AgentOSHierarchicalPlanner
from agentos.planning.static import StaticPlanner
from agentos.routing import ModelRoute, ModelRouteRequest, RuleBasedModelRouter
from agentos.schemas.enums import (
    EventType,
    RuntimeState,
    SkillStatus,
    VerificationAction,
    VerificationLevel,
    VerificationStatus,
)
from agentos.schemas.memory import MemoryNodeType, MemoryQuery
from agentos.schemas.navigation import AnchorType, NavigationIntent, NavigationMode, SpatialAnchor
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult
from agentos.skills.mock import MockNavigationSkill
from agentos.skills.registry import SkillRegistry
from agentos.verification.simple import SimpleFinishVerifier, SimpleSkillVerifier


def _task_and_observation(make_trace, instruction: str = "Go to the kitchen and stop."):
    trace = make_trace(skill_run_id="task-root")
    observation_trace = trace.for_skill(skill_run_id="observation-root", observation_id="obs-kernel")
    return (
        TaskRequest(trace=trace, instruction=instruction),
        Observation(
            trace=observation_trace,
            rgb_refs={"front": "mock://front.jpg"},
            pose=Pose2D(x=1.0, y=2.0, yaw=0.2, source="mock", frame="mock_odom"),
            text="A kitchen doorway is visible ahead.",
        ),
    )


def test_persistent_memory_and_three_stage_verification(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "agent_memory.sqlite3")
    journal_mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode.casefold() == "delete"
    writer = MemoryWriter(store)
    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    runtime = AgentRuntime(
        planner=StaticPlanner(),
        skills=registry,
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=SimpleFinishVerifier(),
        memory_writer=writer,
    )
    task, observation = _task_and_observation(make_trace)

    result = asyncio.run(runtime.run_task(task, observation))

    assert result.state == RuntimeState.DONE
    event_types = [event.event_type for event in result.events]
    assert event_types.index(EventType.RUNTIME_VERIFIED) < event_types.index(EventType.SKILL_VERIFIED)
    assert event_types.index(EventType.SKILL_VERIFIED) < event_types.index(EventType.FINISH_VERIFIED)
    assert EventType.MEMORY_WRITE_FAILED not in event_types

    task_hits = store.search(MemoryQuery(text="kitchen", node_types=[MemoryNodeType.TASK]))
    skill_hits = store.search(MemoryQuery(node_types=[MemoryNodeType.SKILL_RUN]))
    assert task_hits[0].node.source_ref.startswith("trace:")
    assert skill_hits[0].node.metadata["status"] == "success"
    assert store.neighbors(skill_hits[0].node.node_id)
    store.close()


def test_context_is_bounded_and_retrieves_source_grounded_evidence(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "context.sqlite3")
    writer = MemoryWriter(store)
    task, observation = _task_and_observation(make_trace)
    writer.record_task(task)
    writer.record_observation(task, observation)
    manager = BoundedContextManager(store, max_events=2, max_results=1, max_evidence=3)
    plan = asyncio.run(StaticPlanner().plan(task, observation))

    context = manager.build(task=task, observation=observation, plan=plan, skill_results=[], events=[])

    assert "mock://front.jpg" not in context.render()
    assert context.robot_state["frame"] == "mock_odom"
    assert context.observation_summary.startswith("A kitchen doorway")
    store.close()


def test_high_confidence_grounding_is_promoted_to_cross_task_place_memory(
    tmp_path,
    make_trace,
):
    store = SQLiteGraphMemory(tmp_path / "long-term.sqlite3")
    writer = MemoryWriter(store)
    task, _ = _task_and_observation(make_trace, "Find the bed in the bedroom.")
    result = SkillResult(
        trace=task.trace.for_skill(skill_run_id="grounded-run"),
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="Target grounded and reached.",
        outputs={
            "slow_guidance_source_observation_id": "obs-bedroom",
            "slow_brain_input": {
                "topology_context": {
                    "map_id": "first-floor",
                    "current_topology_node_id": "backbone-003",
                    "nodes": [
                        {
                            "topology_node_id": "backbone-003",
                            "place_type": "bedroom",
                            "scene_summary": "A sleeping area.",
                            "objects": ["bed"],
                        }
                    ],
                }
            },
            "slow_brain_grounding": {
                "target_grounding": {
                    "label": "bed",
                    "view": "left",
                    "bbox_normalized_1000": [100, 200, 500, 700],
                    "confidence": 0.96,
                    "evidence_summary": "A bed is clearly visible.",
                },
                "affordance_grounding": {
                    "label": "doorway",
                    "view": "front",
                    "bbox_normalized_1000": [300, 100, 600, 900],
                    "confidence": 0.82,
                    "evidence_summary": "A doorway is visible.",
                },
            },
        },
    )

    writer.record_skill_result(task, result)
    for _ in range(8):
        writer.record_verification(
            task,
            VerificationResult(
                trace=result.trace,
                level=VerificationLevel.SKILL,
                status=VerificationStatus.FAILURE,
                summary="Find the bed in the bedroom requires another verifier review.",
                recommended_action=VerificationAction.REPLAN,
            ),
        )

    entity_hits = store.search(
        MemoryQuery(text="bed", node_types=[MemoryNodeType.ENTITY])
    )
    assert len(entity_hits) == 1
    assert entity_hits[0].node.task_id is None
    assert entity_hits[0].node.metadata["topology_node_id"] == "backbone-003"
    assert any(
        node.node_type == MemoryNodeType.PLACE
        for node in store.neighbors(entity_hits[0].node.node_id)
    )
    assert not store.search(
        MemoryQuery(text="doorway", node_types=[MemoryNodeType.ENTITY])
    )
    _, observation = _task_and_observation(make_trace, "Find the bed again.")
    followup = TaskRequest(
        trace=make_trace(task_id="followup-task"),
        instruction="Find the bed again.",
    )
    context = BoundedContextManager(store, max_evidence=4).build(
        task=followup,
        observation=observation,
        plan=asyncio.run(StaticPlanner().plan(followup, observation)),
        skill_results=[],
        events=[],
    )
    assert {item.node_type for item in context.retrieved_evidence} >= {
        "entity",
        "place",
    }
    assert "semantic_event" not in {
        item.node_type for item in context.retrieved_evidence
    }
    planner = AgentOSHierarchicalPlanner(
        provider=None,
        service=None,
        session=AgentOSMainAgentSession(global_goal=followup.instruction),
        memory=store,
    )
    planner_memory = planner._retrieve_long_term_memory(
        followup.instruction,
        map_id="first-floor",
    )
    assert [item["node_type"] for item in planner_memory[:2]] == [
        "entity",
        "place",
    ]
    store.close()


def test_skill_catalog_and_edge_cloud_router():
    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    descriptor = registry.describe("NavigationSkill")
    assert descriptor.required_modalities == ["rgb", "pose"]
    assert registry.catalog(embodiment="mock")[0].name == "NavigationSkill"

    router = RuleBasedModelRouter(cloud_complexity_threshold=6)
    assert router.route(ModelRouteRequest(purpose="reflex", latency_budget_ms=200)).route == ModelRoute.EDGE
    assert router.route(ModelRouteRequest(purpose="long-horizon plan", complexity=9)).route == ModelRoute.CLOUD


class _ReplanVerifier:
    async def verify_runtime(self, request, result, observation):
        del request, observation
        if result.status == SkillStatus.SUCCESS:
            return VerificationResult(
                trace=result.trace,
                level=VerificationLevel.RUNTIME,
                status=VerificationStatus.SUCCESS,
                summary="runtime accepted",
            )
        return VerificationResult(
            trace=result.trace,
            level=VerificationLevel.RUNTIME,
            status=VerificationStatus.FAILURE,
            summary="blocked route requires recovery",
            recommended_action=VerificationAction.REPLAN,
        )


class _FailThenSucceedSkill(MockNavigationSkill):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run(self, request, observation, context):
        self.calls += 1
        self.force_status = SkillStatus.FAILURE if self.calls == 1 else SkillStatus.SUCCESS
        return await super().run(request, observation, context)


class _RecoveryPlanner:
    async def replan(self, task, observation, previous_plan, skill_results, *, reason, next_revision):
        del observation, previous_plan, skill_results, reason
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=next_revision,
            rationale_summary="Recovery plan avoids the blocked route.",
            steps=[
                PlanStep(
                    step_id="recovery-001",
                    skill_name="NavigationSkill",
                    command=task.instruction,
                )
            ],
        )


def test_runtime_can_replan_without_losing_failure_trace(make_trace):
    registry = SkillRegistry()
    skill = _FailThenSucceedSkill()
    registry.register(skill)
    runtime = AgentRuntime(
        planner=StaticPlanner(),
        replanner=_RecoveryPlanner(),
        runtime_verifier=_ReplanVerifier(),
        skills=registry,
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=SimpleFinishVerifier(),
    )
    task, observation = _task_and_observation(make_trace, "Go around the blocked doorway.")

    result = asyncio.run(runtime.run_task(task, observation))

    assert result.state == RuntimeState.DONE
    assert [item.status for item in result.skill_results] == [SkillStatus.FAILURE, SkillStatus.SUCCESS]
    assert result.plan is not None and result.plan.plan_revision == 1
    assert EventType.PLAN_REVISED in [event.event_type for event in result.events]


def test_navigation_intent_and_evolution_rollback_are_typed(tmp_path):
    intent = NavigationIntent(
        mode=NavigationMode.POINT_GOAL,
        semantic_goal="reach the open doorway",
        success_conditions=["doorway crossed"],
        anchors=[
            SpatialAnchor(
                anchor_type=AnchorType.WORLD_POINT,
                value={"x": 2.0, "y": 1.0},
                frame="map",
                source_observation_id="obs-1",
                source="slow_planner",
            )
        ],
    )
    assert intent.anchors[0].frame == "map"

    store = JsonEvoAssetStore(tmp_path / "assets")
    gate = EvoAssetGate(name="regression", passed=True, summary="safe")
    for version in (1, 2):
        store.save(
            EvoAsset(
                asset_id="door-recovery",
                asset_type="recovery_policy",
                version=version,
                trigger_failure="door traversal stalled",
                operations=[{"op": "reobserve"}],
                gates=[gate],
            )
        )
        store.promote("door-recovery", version)

    restored = store.rollback("door-recovery", 1)
    assert restored.version == 1
    assert restored.status == EvoAssetStatus.PROMOTED
    assert store.load("door-recovery", 2).status == EvoAssetStatus.ROLLED_BACK


def test_runtime_records_bounded_trace_for_offline_skill_evolvement(tmp_path, make_trace):
    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    runtime = AgentRuntime(
        planner=StaticPlanner(),
        skills=registry,
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=SimpleFinishVerifier(),
        skill_evolvement=TraceDrivenSkillEvolvement(tmp_path / "traces"),
    )
    task, observation = _task_and_observation(make_trace)

    result = asyncio.run(runtime.run_task(task, observation))
    trace_path = next((tmp_path / "traces").glob("*.json"))
    trace_text = trace_path.read_text(encoding="utf-8")

    assert result.state == RuntimeState.DONE
    assert '"terminal_state": "DONE"' in trace_text
    assert '"skill_results"' in trace_text
    assert '"outputs"' not in trace_text
