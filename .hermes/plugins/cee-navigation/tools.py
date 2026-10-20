from __future__ import annotations

import json
import os
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _header(prefix: str, title: str) -> str:
    return f"{prefix} {title}"


def _status(value: object) -> str:
    return str(value or "n/a")


def _request(path: str, payload: dict | None = None) -> str:
    base_url = os.environ.get("CEE_AGENTOS_TOOL_URL", "http://127.0.0.1:8765").rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=35) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        error = exc.read().decode("utf-8", "replace")
        return json.dumps({"ok": False, "error": f"HTTP {exc.code}: {error}"})
    except (URLError, TimeoutError, OSError) as exc:
        return json.dumps({"ok": False, "error": f"CEE-AgentOS unavailable: {exc}"})


def navigation_status(_args: dict, **_kwargs) -> str:
    started = time.monotonic()
    print("\n" + _header("[Hermes Client]", "【服务检查】"), flush=True)
    print("正在做：检查 AgentOS、Habitat、语义地图和导航能力。", flush=True)
    result = _request("/health")
    print(f"执行结果：服务检查返回 | 耗时={time.monotonic() - started:.2f}s", flush=True)
    return result


def start_agent_task(args: dict, **_kwargs) -> str:
    client_started = time.monotonic()
    instruction = _normalize_instruction(args)
    if instruction is None:
        return json.dumps(
            {
                "ok": False,
                "error": (
                    "cee_start_agent_task requires a non-empty instruction. "
                    "Use {\"instruction\": \"<complete user instruction>\"}."
                ),
            }
        )
    print("\n" + _header("[Hermes Client]", "【任务提交】"), flush=True)
    print(f"收到指令：{instruction}", flush=True)
    print("正在做：按 AgentOS 协议提交一次全局任务。", flush=True)
    submit_started = time.monotonic()
    started_raw = _request(
        "/tasks/agent/start",
        {
            "instruction": instruction,
            "success_conditions": args.get("success_conditions", []),
            "max_steps": args.get("max_steps", 1000),
            "timeout_s": args.get("timeout_s", 1800),
            "max_plan_revisions": args.get("max_plan_revisions", 20),
        },
    )
    submit_duration_s = time.monotonic() - submit_started
    if not args.get("wait_for_completion", True):
        print(f"执行结果：任务提交请求已返回 | 提交耗时={submit_duration_s:.2f}s", flush=True)
        return started_raw
    try:
        started = json.loads(started_raw)
    except json.JSONDecodeError:
        return started_raw
    if not started.get("ok") or not started.get("task_id"):
        return started_raw
    print(
        f"执行结果：任务已受理 task_id={started['task_id']} | 提交耗时={submit_duration_s:.2f}s",
        flush=True,
    )
    print("接下来：AgentOS 开始规划并执行，Hermes 持续显示边界事件。", flush=True)
    result = _wait_for_task(
        str(started["task_id"]),
        timeout_s=float(args.get("wait_timeout_s", 900)),
    )
    print(
        f"\n[Hermes Client] 【任务返回】\n"
        f"执行结果：AgentOS 已返回终态 | Hermes 工具总耗时={time.monotonic() - client_started:.2f}s",
        flush=True,
    )
    return result


def _normalize_instruction(args: dict) -> str | None:
    for key in ("instruction", "task", "task_description", "global_goal"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def navigation_task_status(_args: dict, **_kwargs) -> str:
    return _request("/tasks/agent/status")


def cancel_agent_task(_args: dict, **_kwargs) -> str:
    return _request("/stop", {})


def _wait_for_task(task_id: str, *, timeout_s: float, poll_interval_s: float = 1.0) -> str:
    started_at = time.monotonic()
    deadline = started_at + timeout_s
    timeline: list[dict] = []
    last_key = None
    last_plan = None
    last_post = None
    last_finish = None
    last_execution = None
    last_verification = None
    known_steps: list[dict] = []

    while True:
        raw = _request("/tasks/agent/status")
        try:
            status = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if not status.get("ok"):
            return raw
        if status.get("task_id") != task_id:
            return json.dumps(
                {"ok": False, "error": "AgentOS task changed while Hermes was waiting", "task_id": task_id}
            )

        key = _timeline_key(status)
        if key != last_key:
            plan = status.get("plan") or {}
            post = status.get("post_decision") or {}
            finish = status.get("finish_verification") or {}
            execution = status.get("execution_summary") or {}
            verification = status.get("verification") or {}
            plan_key = json.dumps(plan, sort_keys=True) if plan else None
            post_key = json.dumps(post, sort_keys=True) if post else None
            finish_key = json.dumps(finish, sort_keys=True) if finish else None
            execution_key = json.dumps(execution, sort_keys=True) if execution else None
            verification_key = (
                json.dumps(verification, sort_keys=True) if verification else None
            )
            plan_changed = plan_key is not None and plan_key != last_plan
            if plan_changed:
                _merge_known_steps(known_steps, _extract_plan_steps(plan))
            event = _compact_task_event(
                status,
                time.monotonic() - started_at,
                include_plan=plan_changed,
                include_post=post_key is not None and post_key != last_post,
                include_finish=finish_key is not None and finish_key != last_finish,
                include_execution=(
                    execution_key is not None and execution_key != last_execution
                ),
                include_verification=(
                    verification_key is not None
                    and verification_key != last_verification
                ),
                known_steps=known_steps,
            )
            timeline.append(event)
            _print_task_event(event)
            last_plan = plan_key or last_plan
            last_post = post_key or last_post
            last_finish = finish_key or last_finish
            last_execution = execution_key or last_execution
            last_verification = verification_key or last_verification
            last_key = key

        if status.get("status") in {"completed", "failed", "cancelled"}:
            return json.dumps(
                {
                    "ok": status.get("status") == "completed",
                    "task_id": task_id,
                    "timeline": timeline,
                    "final_result": _compact_final_result(status),
                },
                ensure_ascii=False,
            )
        if time.monotonic() >= deadline:
            return json.dumps(
                {
                    "ok": False,
                    "task_id": task_id,
                    "error": "Hermes wait timeout; the AgentOS task may still be running",
                    "timeline": timeline,
                    "current_state": _compact_task_event(status, time.monotonic() - started_at),
                },
                ensure_ascii=False,
            )
        time.sleep(poll_interval_s)


def _timeline_key(status: dict) -> tuple:
    finish = status.get("finish_verification") or {}
    verification = status.get("verification") or {}
    execution = status.get("execution_summary") or {}
    skill_progress = status.get("skill_progress") or {}
    return (
        status.get("status"),
        status.get("agent_state") or status.get("agent_phase"),
        status.get("last_event"),
        status.get("plan_revision"),
        status.get("agent_calls"),
        status.get("skill_run_index"),
        int(status.get("step") or 0),
        status.get("active_skill"),
        status.get("active_subtask"),
        status.get("visible_anchor"),
        status.get("anchor_view"),
        status.get("previous_subtask_status"),
        verification.get("status"),
        finish.get("verdict") or finish.get("status"),
        execution.get("role"),
        execution.get("label"),
        execution.get("post_observation_id"),
        execution.get("state"),
        json.dumps(skill_progress, sort_keys=True) if skill_progress else None,
    )


def _extract_plan_steps(plan: dict) -> list[dict]:
    metadata = plan.get("metadata") or {}
    subtasks = metadata.get("planned_skill_subtasks") or plan.get("subtasks") or []
    if not subtasks and metadata.get("delegated_subtask"):
        subtasks = [metadata["delegated_subtask"]]
    if not subtasks:
        subtasks = [
            {
                "skill_name": step.get("skill_name"),
                "instruction": step.get("command"),
            }
            for step in plan.get("steps", [])
            if step.get("command")
        ]

    normalized = []
    for subtask in subtasks:
        if isinstance(subtask, str):
            normalized.append({"instruction": subtask})
        elif isinstance(subtask, dict):
            normalized.append(
                {
                    "skill_name": subtask.get("skill_name"),
                    "task_type": subtask.get("task_type"),
                    "instruction": subtask.get("instruction")
                    or subtask.get("command")
                    or subtask.get("target"),
                    "target": subtask.get("target"),
                    "topology_node_id": subtask.get("topology_node_id"),
                    "success_condition": subtask.get("success_condition"),
                }
            )
    return normalized


def _step_identity(step: dict) -> tuple:
    return (
        str(step.get("skill_name") or ""),
        str(step.get("task_type") or ""),
        str(step.get("topology_node_id") or step.get("target") or ""),
        " ".join(str(step.get("instruction") or "").lower().split()),
    )


def _same_logical_step(existing: dict, updated: dict) -> bool:
    if (
        str(existing.get("skill_name") or "")
        != str(updated.get("skill_name") or "")
        or str(existing.get("task_type") or "")
        != str(updated.get("task_type") or "")
    ):
        return False
    task_type = str(updated.get("task_type") or "")
    if task_type == "point_goal":
        if not existing.get("topology_node_id") and updated.get("topology_node_id"):
            return True
        return str(existing.get("target") or "").strip().lower() == str(
            updated.get("target") or ""
        ).strip().lower()
    if task_type == "object_goal":
        return str(existing.get("target") or "").strip().lower() == str(
            updated.get("target") or ""
        ).strip().lower()
    if str(updated.get("skill_name") or "") == "MappingSkill":
        return True
    return _step_identity(existing) == _step_identity(updated)


def _merge_known_steps(known_steps: list[dict], new_steps: list[dict]) -> None:
    for step in new_steps:
        match = next(
            (
                index
                for index, existing in enumerate(known_steps)
                if _same_logical_step(existing, step)
            ),
            None,
        )
        if match is None:
            known_steps.append(step)
        else:
            known_steps[match] = step


def _render_step(step: dict) -> str:
    skill_name = str(step.get("skill_name") or "Skill")
    task_type = str(step.get("task_type") or "")
    instruction = _one_line(step.get("instruction"), limit=180)
    target = step.get("topology_node_id") or step.get("target")
    if skill_name == "MappingSkill":
        action = "读取当前定位、语义地图与拓扑证据"
        return f"{skill_name} - {action}"
    elif task_type == "point_goal":
        action = f"前往已知区域 {target}" if target else instruction
    elif task_type == "object_goal":
        action = f"搜索并接近 {target}" if target else instruction
    elif task_type == "instruction_following":
        action = f"执行自然语言导航：{instruction}"
    else:
        action = instruction
    return f"{skill_name} ({task_type or 'task'}) - {action}"


def _step_progress(status: dict, known_steps: list[dict]) -> dict:
    completed = max(0, int(status.get("step") or 0))
    total = max(len(known_steps), completed)
    event_name = str(status.get("last_event") or "")
    if status.get("status") in {"completed", "failed", "cancelled"}:
        current = min(completed, total) if total else 0
    elif event_name in {
        "plan_created",
        "plan_revised",
        "skill_started",
        "skill_context_opened",
        "context_built",
    }:
        current = min(completed + 1, total) if total else completed + 1
    else:
        current = min(max(completed, 1), total) if total else max(completed, 1)
    return {"current": current, "total": total, "completed": completed}


def _compact_task_event(
    status: dict,
    elapsed_s: float,
    *,
    include_plan: bool = True,
    include_post: bool = True,
    include_finish: bool = True,
    include_execution: bool = True,
    include_verification: bool = True,
    known_steps: list[dict] | None = None,
) -> dict:
    known_steps = known_steps or []
    event = {
        "elapsed_s": round(elapsed_s, 1),
        "status": status.get("status"),
        "phase": status.get("agent_state") or status.get("agent_phase"),
        "last_event": status.get("last_event"),
        "plan_revision": status.get("plan_revision", 0),
        "step": status.get("step", 0),
        "progress": _step_progress(status, known_steps),
        "distance_m": status.get("distance_m", 0.0),
        "agent_calls": status.get("agent_calls", 0),
        "active_skill": status.get("active_skill"),
        "active_subtask": status.get("active_subtask"),
        "visible_anchor": status.get("visible_anchor"),
        "anchor_view": status.get("anchor_view"),
        "event_duration_s": status.get("event_duration_s"),
    }
    if known_steps and event["progress"]["current"]:
        index = min(event["progress"]["current"], len(known_steps)) - 1
        event["current_step_description"] = _render_step(known_steps[index])
    skill_progress = status.get("skill_progress") or {}
    if skill_progress and status.get("last_event") == "skill_progress":
        event["skill_progress"] = skill_progress
    execution = status.get("execution_summary") or {}
    if execution and include_execution:
        event["execution"] = execution
    plan = status.get("plan") or {}
    if plan and include_plan:
        metadata = plan.get("metadata") or {}
        subtasks = known_steps or _extract_plan_steps(plan)
        completed = event["progress"]["completed"]
        current = event["progress"]["current"]
        rendered_subtasks = []
        for index, subtask in enumerate(subtasks):
            if index < completed:
                marker = "[已完成]"
            elif index + 1 == current:
                marker = "[当前]"
            else:
                marker = "[待执行]"
            rendered_subtasks.append(f"{marker} {_render_step(subtask)}")
        event["plan"] = {
            "summary": plan.get("task_summary") or plan.get("rationale_summary"),
            "subtasks": rendered_subtasks,
            "reasoning_summary": plan.get("rationale_summary"),
            "revision": status.get("plan_revision", plan.get("plan_revision", 0)),
            "duration_s": (status.get("main_agent_timing") or {}).get(
                "duration_s"
            ),
        }
    post = status.get("post_decision") or {}
    if post and include_post:
        event["post_agent"] = {
            "previous_subtask_status": post.get("previous_subtask_status"),
            "evidence_summary": post.get("evidence_summary"),
            "decision": post.get("decision"),
            "next_subtask": post.get("next_subtask"),
            "reasoning_summary": post.get("rationale_summary"),
        }
    finish = status.get("finish_verification") or {}
    if finish and include_finish:
        event["finish_verifier"] = {
            "verdict": finish.get("verdict") or finish.get("recommended_action") or finish.get("status"),
            "evidence_view": finish.get("evidence_view"),
            "evidence_summary": finish.get("evidence_summary") or finish.get("summary"),
            "duration_s": (status.get("verification_timing") or {}).get(
                "duration_s"
            ),
        }
    verification = status.get("verification") or {}
    if verification and not finish and include_verification:
        event["verifier"] = {
            "level": verification.get("level"),
            "status": verification.get("status"),
            "action": verification.get("recommended_action"),
            "summary": verification.get("summary"),
            "duration_s": (status.get("verification_timing") or {}).get(
                "duration_s"
            ),
        }
    runner_duration = _skill_runner_duration(status.get("skill_runner") or {})
    if runner_duration is not None:
        event["skill_runner_duration_s"] = runner_duration
    return event


def _skill_runner_duration(skill_runner: dict) -> float | None:
    session = skill_runner.get("session") or {}
    started = session.get("started_at")
    completed = session.get("completed_at")
    if not started or not completed:
        return None
    try:
        start_time = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(max(0.0, (end_time - start_time).total_seconds()), 3)


def _duration_text(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}s"
    return "n/a"


def _print_task_event(event: dict) -> None:
    prefix = f"[AgentOS {event['elapsed_s']:.1f}s]"
    progress = event.get("progress") or {}
    current = progress.get("current") or event.get("step") or 0
    total = progress.get("total") or current
    step_text = f"第 {current}/{total} 步" if total else "任务级"
    last_event = str(event.get("last_event") or "")
    plan = event.get("plan")
    if plan:
        print("\n" + _header(prefix, f"【Main Agent｜任务规划 r{plan.get('revision')}】"), flush=True)
        print(f"任务判断：{_one_line(plan.get('summary'))}", flush=True)
        print(f"阶段耗时：{_duration_text(plan.get('duration_s'))} | 累计={event['elapsed_s']:.1f}s", flush=True)
        print("任务拆解：", flush=True)
        for index, subtask in enumerate(plan.get("subtasks") or [], start=1):
            print(f"  {index}. {subtask}", flush=True)
        if current:
            print(f"接下来：执行第 {current} 步。", flush=True)
    if last_event == "skill_started":
        print("\n" + _header(prefix, f"【Skill Runner｜{step_text}】"), flush=True)
        print(f"模块：{event.get('active_skill') or 'Skill'}", flush=True)
        print(f"正在做：{_one_line(event.get('current_step_description') or event.get('active_subtask'))}", flush=True)
        print("状态：已创建独立执行上下文，开始执行。", flush=True)
    skill_progress = event.get("skill_progress")
    if skill_progress:
        stage = str(skill_progress.get("stage") or "skill").lower()
        module_names = {
            "slow_brain": "Slow Brain",
            "visual_grounding": "Visual Grounding",
            "fast_system": "Fast System",
            "decision_pipeline": "Slow/Grounding Pipeline",
        }
        module = module_names.get(stage, stage)
        print("\n" + _header(prefix, f"【{module}｜{step_text}】"), flush=True)
        print(f"正在做：{_one_line(skill_progress.get('summary'))}", flush=True)
        if skill_progress.get("target_query"):
            print(f"目标查询：{skill_progress.get('target_query')}", flush=True)
        if skill_progress.get("affordance_query"):
            print(f"通行锚点查询：{skill_progress.get('affordance_query')}", flush=True)
        print(f"状态：{_status(skill_progress.get('status') or 'updated')}", flush=True)
        durations = []
        for label, field in (
            ("Slow", "slow_inference_latency_s"),
            ("View Binding", "slow_view_binding_latency_s"),
            ("Grounding", "visual_grounding_latency_s"),
            ("本阶段", "duration_s"),
            ("决策链", "decision_pipeline_latency_s"),
        ):
            value = skill_progress.get(field)
            if isinstance(value, (int, float)):
                durations.append(f"{label}={float(value):.2f}s")
        if durations:
            print(f"阶段耗时：{' | '.join(durations)} | 累计={event['elapsed_s']:.1f}s", flush=True)
    post = event.get("post_agent")
    if post:
        print("\n" + _header(prefix, "【Main Agent｜执行复盘】"), flush=True)
        print(f"上一步：{post.get('previous_subtask_status')}", flush=True)
        print(f"观察证据：{_one_line(post.get('evidence_summary'))}", flush=True)
        print(f"判断：{_one_line(post.get('reasoning_summary'))}", flush=True)
        print(f"接下来：{post.get('decision')} - {_one_line(post.get('next_subtask'))}", flush=True)
    execution = event.get("execution")
    if execution:
        role = str(execution.get("role") or "skill").upper()
        label = execution.get("label") or execution.get("skill_name") or "n/a"
        view = str(execution.get("view") or "n/a").upper()
        confidence = execution.get("confidence")
        confidence_text = (
            f"{float(confidence):.2f}"
            if isinstance(confidence, (int, float))
            else "n/a"
        )
        reached = execution.get("goal_reached")
        reached_text = "yes" if reached is True else "no" if reached is False else "n/a"
        progress = execution.get("progress_m")
        progress_text = (
            f"{float(progress):.2f}m"
            if isinstance(progress, (int, float))
            else "n/a"
        )
        print("\n" + _header(prefix, f"【Skill Runner｜{step_text}执行结果】"), flush=True)
        print(f"模块：{execution.get('skill_name') or event.get('active_skill') or 'Skill'}", flush=True)
        print(f"看见了什么：{_one_line(execution.get('observed_evidence'))}", flush=True)
        print(f"任务判断：{_one_line(execution.get('reasoning'))}", flush=True)
        print(
            f"选择动作：{role} {label} | 视图={view} | 置信度={confidence_text}",
            flush=True,
        )
        print(f"为什么：{_one_line(execution.get('selection_reason'))}", flush=True)
        print(
            f"执行结果：状态={_status(execution.get('state'))} | 到达={reached_text} | "
            f"前进={progress_text} | 原因={execution.get('termination_reason') or 'n/a'}",
            flush=True,
        )
        timings = []
        for label, field in (
            ("Slow", "slow_brain_latency_s"),
            ("View Binding", "slow_view_binding_latency_s"),
            ("Grounding", "visual_grounding_latency_s"),
            ("决策链", "decision_pipeline_latency_s"),
            ("Fast", "fast_system_latency_s"),
        ):
            value = execution.get(field)
            if isinstance(value, (int, float)):
                timings.append(f"{label}={float(value):.2f}s")
        runner_duration = event.get("skill_runner_duration_s")
        if isinstance(runner_duration, (int, float)):
            timings.append(f"Skill Runner总计={float(runner_duration):.2f}s")
        if timings:
            print(f"阶段耗时：{' | '.join(timings)} | 累计={event['elapsed_s']:.1f}s", flush=True)
    finish = event.get("finish_verifier")
    if finish:
        print("\n" + _header(prefix, "【Finish Verifier｜完成验证】"), flush=True)
        print(f"检查结果：{_status(finish.get('verdict'))}", flush=True)
        print(f"视觉证据：{finish.get('evidence_view') or 'n/a'} - {_one_line(finish.get('evidence_summary'))}", flush=True)
        print(f"阶段耗时：{_duration_text(finish.get('duration_s'))} | 累计={event['elapsed_s']:.1f}s", flush=True)
    verifier = event.get("verifier")
    if verifier:
        level = "Runtime Verifier" if verifier.get("level") == "runtime" else "Skill Verifier"
        print("\n" + _header(prefix, f"【{level}｜{step_text}】"), flush=True)
        print(f"检查内容：{_one_line(verifier.get('summary'))}", flush=True)
        print(f"检查结果：{_status(verifier.get('status'))}", flush=True)
        print(f"接下来：{_status(verifier.get('action'))}", flush=True)
        print(f"阶段耗时：{_duration_text(verifier.get('duration_s'))} | 累计={event['elapsed_s']:.1f}s", flush=True)
    if event.get("status") in {"completed", "failed", "cancelled"}:
        print("\n" + _header(prefix, "【最终结果】"), flush=True)
        print(f"状态：{_status(event.get('status'))}", flush=True)
        print(f"总步骤：{event.get('step')} | 总移动距离：{event.get('distance_m')}m", flush=True)


def _one_line(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "n/a").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _compact_final_result(status: dict) -> dict:
    return {
        "status": status.get("status"),
        "reason": status.get("reason"),
        "steps": status.get("step", 0),
        "distance_m": status.get("distance_m", 0.0),
        "agent_calls": status.get("agent_calls", 0),
        "plan": status.get("plan"),
        "skill_results": [
            {
                "skill_name": item.get("skill_name"),
                "status": item.get("status"),
                "summary": item.get("compressed_history") or item.get("summary"),
            }
            for item in status.get("skill_results", [])
        ],
        "last_subtask": status.get("active_subtask"),
        "last_visible_anchor": status.get("visible_anchor"),
        "last_anchor_view": status.get("anchor_view"),
        "finish_verification": status.get("finish_verification"),
    }
