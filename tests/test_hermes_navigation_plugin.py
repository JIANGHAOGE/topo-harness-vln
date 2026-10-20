from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tools_module():
    path = Path(__file__).parents[1] / ".hermes/plugins/cee-navigation/tools.py"
    spec = importlib.util.spec_from_file_location("cee_navigation_tools_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hierarchical_launcher_uses_agentos_task_facade():
    root = Path(__file__).parents[1]
    launcher = (root / "scripts/start_hierarchical_agent.sh").read_text()

    assert "configs/navigation_SOUL.md" in launcher
    assert "--toolsets cee-navigation" in launcher
    assert "CEE_AGENTOS_TOOL_URL" in launcher
    assert "--toolsets cee-hierarchical" not in launcher


def test_wait_for_task_returns_compact_reasoning_timeline(monkeypatch):
    tools = _load_tools_module()
    task_id = "task-1"
    base = {
        "ok": True,
        "task_id": task_id,
        "status": "running",
        "agent_phase": "omninav",
        "agent_calls": 1,
        "skill_run_index": 0,
        "step": 1,
        "distance_m": 0.2,
        "active_subtask": "Reach the doorway",
        "plan": {
            "task_summary": "Find a bed.",
            "subtasks": ["Reach the doorway", "Find the bedroom"],
            "rationale_summary": "The doorway is the best immediate target.",
        },
        "trajectory_memory": [{"raw": "must not be copied"}],
    }
    statuses = [
        base,
        {
            **base,
            "agent_phase": "after_omninav",
            "agent_calls": 2,
            "step": 16,
            "distance_m": 3.2,
            "post_decision": {
                "previous_subtask_status": "success",
                "evidence_summary": "The doorway was reached.",
                "decision": "next_subtask",
                "next_subtask": "Find the bedroom",
                "rationale_summary": "Continue through the doorway.",
            },
        },
        {
            **base,
            "status": "completed",
            "agent_phase": "finish_verification",
            "agent_calls": 4,
            "step": 31,
            "distance_m": 6.2,
            "active_subtask": "Find the bedroom",
            "reason": "finish_verified",
            "finish_verification": {
                "verdict": "finish",
                "evidence_view": "front",
                "evidence_summary": "A bed is unambiguously visible.",
            },
        },
    ]

    monkeypatch.setattr(tools, "_request", lambda _path: json.dumps(statuses.pop(0)))
    result = json.loads(tools._wait_for_task(task_id, timeout_s=5, poll_interval_s=0))

    assert result["ok"] is True
    assert result["final_result"]["reason"] == "finish_verified"
    assert result["timeline"][0]["plan"]["reasoning_summary"] == (
        "The doorway is the best immediate target."
    )
    assert result["timeline"][1]["post_agent"]["decision"] == "next_subtask"
    assert result["timeline"][-1]["finish_verifier"]["evidence_view"] == "front"
    assert "trajectory_memory" not in json.dumps(result)


def test_wait_for_task_reports_each_execution_summary_once(monkeypatch, capsys):
    tools = _load_tools_module()
    task_id = "task-execution"
    execution = {
        "skill_name": "NavigationSkill",
        "role": "target",
        "label": "white refrigerator",
        "view": "front",
        "confidence": 0.95,
        "observed_evidence": "A white refrigerator is visible in the front view.",
        "reasoning": "The refrigerator is the final object required by the task.",
        "selection_reason": "The target is clearly visible and overrides the route affordance.",
        "state": "reached",
        "goal_reached": True,
        "progress_m": 1.4,
        "termination_reason": "reached",
        "post_observation_id": "obs-2",
    }
    statuses = [
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "agent_phase": "verify_runtime",
            "last_event": "skill_result",
            "execution_summary": execution,
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "completed",
            "agent_phase": "done",
            "last_event": "finish_verified",
            "execution_summary": execution,
            "finish_verification": {
                "verdict": "finish",
                "evidence_summary": "The target is confirmed.",
            },
        },
    ]

    monkeypatch.setattr(tools, "_request", lambda _path: json.dumps(statuses.pop(0)))
    result = json.loads(tools._wait_for_task(task_id, timeout_s=5, poll_interval_s=0))
    output = capsys.readouterr().out

    assert result["timeline"][0]["execution"]["role"] == "target"
    assert output.count("选择动作：TARGET white refrigerator") == 1
    assert "看见了什么：A white refrigerator is visible" in output
    assert "任务判断：The refrigerator is the final object" in output
    assert "执行结果：状态=reached | 到达=yes | 前进=1.40m" in output


def test_start_agent_task_uses_agentos_harness_endpoint(monkeypatch):
    tools = _load_tools_module()
    calls = []

    def request(path, payload=None):
        calls.append((path, payload))
        return json.dumps({"ok": True, "task_id": "agent-task-1", "status": "starting"})

    monkeypatch.setattr(tools, "_request", request)
    result = json.loads(
        tools.start_agent_task(
            {
                "instruction": "Find the bed.",
                "wait_for_completion": False,
            }
        )
    )

    assert result["task_id"] == "agent-task-1"
    assert calls[0][0] == "/tasks/agent/start"
    assert calls[0][1]["timeout_s"] == 1800
    assert "max_omninav_steps_per_skill" not in calls[0][1]


def test_start_agent_task_normalizes_legacy_instruction_aliases(monkeypatch):
    tools = _load_tools_module()
    calls = []

    def request(path, payload=None):
        calls.append((path, payload))
        return json.dumps({"ok": True, "task_id": "agent-task-alias", "status": "starting"})

    monkeypatch.setattr(tools, "_request", request)
    for key in ("task", "task_description", "global_goal"):
        calls.clear()
        result = json.loads(
            tools.start_agent_task(
                {key: "  Find the fridge.  ", "wait_for_completion": False}
            )
        )
        assert result["task_id"] == "agent-task-alias"
        assert calls[0][1]["instruction"] == "Find the fridge."


def test_start_agent_task_rejects_missing_instruction_without_http_request(monkeypatch):
    tools = _load_tools_module()
    calls = []
    monkeypatch.setattr(tools, "_request", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = json.loads(tools.start_agent_task({"wait_for_completion": False}))

    assert result["ok"] is False
    assert "requires a non-empty instruction" in result["error"]
    assert calls == []


def test_wait_for_task_renders_strict_harness_events(monkeypatch):
    tools = _load_tools_module()
    task_id = "agent-task-2"
    statuses = [
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "agent_state": "PLAN",
            "last_event": "plan_created",
            "plan": {
                "rationale_summary": "Reach the room before searching for the bed.",
                "steps": [{"command": "Navigate to the bedroom."}],
                "metadata": {
                    "delegated_subtask_index": 0,
                    "planned_skill_subtasks": [
                        {
                            "skill_name": "HierarchicalNavigationSkill",
                            "instruction": "Navigate to the bedroom.",
                        },
                        {
                            "skill_name": "HierarchicalNavigationSkill",
                            "instruction": "Search the bedroom for the bed.",
                        },
                    ],
                },
            },
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "agent_state": "VERIFY_SKILL",
            "last_event": "skill_verified",
            "active_skill": "NavigationSkill",
            "verification": {
                "level": "skill",
                "status": "success",
                "recommended_action": "continue",
                "summary": "The bedroom subgoal is satisfied.",
            },
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "completed",
            "agent_state": "DONE",
            "last_event": "task_done",
            "skill_results": [
                {
                    "skill_name": "NavigationSkill",
                    "status": "success",
                    "summary": "raw result",
                    "compressed_history": "Reached the bedroom.",
                    "outputs": {"raw_frames": ["must not escape"]},
                }
            ],
            "finish_verification": {
                "level": "finish",
                "status": "success",
                "recommended_action": "finish",
                "summary": "The bed is visible.",
            },
        },
    ]

    monkeypatch.setattr(tools, "_request", lambda _path: json.dumps(statuses.pop(0)))
    result = json.loads(tools._wait_for_task(task_id, timeout_s=5, poll_interval_s=0))

    assert result["timeline"][0]["plan"]["subtasks"] == [
        "[当前] HierarchicalNavigationSkill (task) - Navigate to the bedroom.",
        "[待执行] HierarchicalNavigationSkill (task) - Search the bedroom for the bed.",
    ]
    assert result["timeline"][1]["verifier"]["level"] == "skill"
    assert result["final_result"]["finish_verification"]["recommended_action"] == "finish"
    assert result["final_result"]["skill_results"][0]["summary"] == "Reached the bedroom."
    assert "raw_frames" not in json.dumps(result)


def test_wait_for_task_keeps_steps_across_staged_plan_revisions(monkeypatch):
    tools = _load_tools_module()
    task_id = "task-staged-plan"
    statuses = [
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "agent_state": "PLAN",
            "last_event": "plan_created",
            "step": 0,
            "plan_revision": 0,
            "plan": {
                "rationale_summary": "Acquire spatial evidence first.",
                "metadata": {
                    "planned_skill_subtasks": [
                        {
                            "skill_name": "MappingSkill",
                            "task_type": "mapping",
                            "instruction": "Acquire localization and topology evidence.",
                        },
                        {
                            "skill_name": "NavigationSkill",
                            "task_type": "point_goal",
                            "instruction": "Resolve and navigate to the office.",
                            "target": "office",
                            "topology_node_id": None,
                        },
                        {
                            "skill_name": "NavigationSkill",
                            "task_type": "object_goal",
                            "instruction": "Find the refrigerator.",
                            "target": "white refrigerator",
                        },
                    ]
                },
            },
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "agent_state": "PLAN",
            "last_event": "plan_revised",
            "step": 1,
            "plan_revision": 1,
            "plan": {
                "rationale_summary": "Use the fresh topology evidence.",
                "metadata": {
                    "planned_skill_subtasks": [
                        {
                            "skill_name": "NavigationSkill",
                            "task_type": "point_goal",
                            "instruction": "Go to the office.",
                            "target": "backbone-002",
                            "topology_node_id": "backbone-002",
                        },
                        {
                            "skill_name": "NavigationSkill",
                            "task_type": "object_goal",
                            "instruction": "Find the refrigerator.",
                            "target": "white refrigerator",
                        },
                    ]
                },
            },
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "completed",
            "agent_state": "DONE",
            "last_event": "task_done",
            "step": 3,
            "distance_m": 12.0,
        },
    ]

    monkeypatch.setattr(tools, "_request", lambda _path: json.dumps(statuses.pop(0)))
    result = json.loads(tools._wait_for_task(task_id, timeout_s=5, poll_interval_s=0))

    revised = result["timeline"][1]["plan"]["subtasks"]
    assert len(revised) == 3
    assert revised[0].startswith("[已完成] MappingSkill")
    assert revised[1].startswith("[当前] NavigationSkill (point_goal)")
    assert revised[2].startswith("[待执行] NavigationSkill (object_goal)")


def test_wait_for_task_prints_verification_once_and_skill_stage_timings(
    monkeypatch, capsys
):
    tools = _load_tools_module()
    task_id = "task-flow"
    verification = {
        "level": "skill",
        "status": "success",
        "recommended_action": "continue",
        "summary": "The subgoal is satisfied.",
    }
    statuses = [
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "last_event": "skill_progress",
            "step": 1,
            "skill_progress": {
                "stage": "visual_grounding",
                "status": "running",
                "summary": "Ground the selected current-view query.",
                "slow_inference_latency_s": 1.25,
                "target_query": "white refrigerator",
            },
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "last_event": "skill_verified",
            "step": 1,
            "verification": verification,
            "verification_timing": {"duration_s": 0.75},
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "running",
            "last_event": "state_transition",
            "agent_state": "PLAN",
            "step": 1,
            "verification": verification,
            "verification_timing": {"duration_s": 0.75},
        },
        {
            "ok": True,
            "task_id": task_id,
            "status": "completed",
            "last_event": "task_done",
            "step": 1,
        },
    ]

    monkeypatch.setattr(tools, "_request", lambda _path: json.dumps(statuses.pop(0)))
    tools._wait_for_task(task_id, timeout_s=5, poll_interval_s=0)
    output = capsys.readouterr().out

    assert "【Visual Grounding｜第 1/1 步】" in output
    assert "目标查询：white refrigerator" in output
    assert "Slow=1.25s" in output
    assert output.count("【Skill Verifier｜第 1/1 步】") == 1
    assert "阶段耗时：0.75s" in output
