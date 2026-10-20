#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.runtime import AgentRuntime
from agentos.embodiments.habitat_gs import HabitatGSAdapter, HabitatGSConfig
from agentos.planning.static import StaticPlanner
from agentos.schemas.base import utc_now
from agentos.schemas.task import TaskRequest
from agentos.schemas.trace import TraceContext
from agentos.skills.habitat_navigation import HabitatNavigationSkill
from agentos.skills.registry import SkillRegistry
from agentos.verification.simple import SimpleFinishVerifier, SimpleSkillVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CEE-AgentOS Habitat-GS smoke loop.")
    parser.add_argument("--instruction", default="Execute the explicit Habitat smoke route.")
    parser.add_argument("--scene-id", default="1st_floor_exh")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/jianghaoge/habitat-gs/scenes/datasets/1st_floor_exh_collision_v1"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=100.0)
    parser.add_argument("--eye-height", type=float, default=0.45)
    parser.add_argument("--forward-step", type=float, default=0.2)
    parser.add_argument("--turn-angle-deg", type=float, default=15.0)
    parser.add_argument("--start-yaw-deg", type=float)
    parser.add_argument("--actions", nargs="+", default=["forward", "left", "forward", "stop"])
    return parser.parse_args()


def build_trace(observation_id: str) -> TraceContext:
    now = utc_now()
    return TraceContext(
        episode_id="habitat-smoke-episode",
        task_id="habitat-smoke-task",
        plan_revision=0,
        skill_run_id="task-root",
        observation_id=observation_id,
        frame_timestamp=now,
        pose_timestamp=now,
        deadline=now + timedelta(seconds=90),
    )


def build_runtime(adapter: HabitatGSAdapter) -> AgentRuntime:
    skills = SkillRegistry()
    skills.register(HabitatNavigationSkill(adapter))
    return AgentRuntime(
        planner=StaticPlanner(timeout_s=60.0),
        skills=skills,
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=SimpleFinishVerifier(),
    )


def make_contact_sheet(before, after, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    panel_specs = (
        ("BEFORE RGB", before.rgb_refs["front"]),
        ("BEFORE TOPDOWN", before.topdown_ref),
        ("AFTER RGB", after.rgb_refs["front"]),
        ("AFTER TOPDOWN", after.topdown_ref),
    )
    panels = []
    for title, path in panel_specs:
        image = Image.open(path).convert("RGB").resize((640, 360))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 38), fill=(12, 15, 18))
        draw.text((14, 11), title, fill=(235, 240, 242))
        panels.append(image)
    sheet = Image.new("RGB", (1280, 720), (8, 10, 12))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 640, (index // 2) * 360))
    sheet.save(output_path)


def make_sensor_sheet(observation, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (1440, 360), (8, 10, 12))
    for index, view in enumerate(("left", "front", "right")):
        image = Image.open(observation.rgb_refs[view]).convert("RGB").resize((480, 360))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 480, 38), fill=(12, 15, 18))
        draw.text((14, 11), view.upper(), fill=(235, 240, 242))
        sheet.paste(image, (index * 480, 0))
    sheet.save(output_path)


async def run(args: argparse.Namespace) -> dict[str, object]:
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "logs" / "habitat_smoke" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = HabitatGSAdapter(
        HabitatGSConfig(
            dataset_root=args.dataset_root,
            scene_id=args.scene_id,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            hfov=args.hfov,
            eye_height=args.eye_height,
            gpu_device_id=args.gpu_device_id,
            forward_step_m=args.forward_step,
            turn_angle_deg=args.turn_angle_deg,
            start_yaw_deg=args.start_yaw_deg,
        )
    )
    try:
        task_trace = build_trace("obs-before")
        before = await adapter.observe(
            task_trace.for_skill(skill_run_id="observation-before", observation_id="obs-before")
        )
        task = TaskRequest(
            trace=task_trace,
            instruction=args.instruction,
            metadata={"skill_inputs": {"actions": args.actions}},
        )
        runtime_result = await build_runtime(adapter).run_task(task, before)
        if not runtime_result.skill_results:
            raise RuntimeError("Runtime completed without a skill result.")
        after = await adapter.observe(
            task_trace.for_skill(skill_run_id="observation-after", observation_id="obs-after")
        )

        contact_sheet = output_dir / "contact_sheet.png"
        sensor_sheet = output_dir / "sensor_sheet.png"
        make_contact_sheet(before, after, contact_sheet)
        make_sensor_sheet(before, sensor_sheet)
        steps = runtime_result.skill_results[0].outputs.get("steps", [])
        distance_m = sum(float(step.get("distance_m", 0.0)) for step in steps)
        summary = {
            "runtime_state": runtime_result.state.value,
            "instruction": args.instruction,
            "scene_id": args.scene_id,
            "dataset_root": str(args.dataset_root),
            "actions": args.actions,
            "camera": {
                "view_order": ["left", "front", "right"],
                "width": args.width,
                "height": args.height,
                "hfov": args.hfov,
                "eye_height": args.eye_height,
            },
            "observation_before": before.model_dump(mode="json"),
            "skill_result": runtime_result.skill_results[0].model_dump(mode="json"),
            "observation_after": after.model_dump(mode="json"),
            "event_count": len(runtime_result.events),
            "executed_distance_m": distance_m,
            "pose_changed": distance_m > 1e-4,
            "contact_sheet": str(contact_sheet),
            "sensor_sheet": str(sensor_sheet),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        adapter.close()


def main() -> None:
    summary = asyncio.run(run(parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["runtime_state"] != "DONE" or not summary["pose_changed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
