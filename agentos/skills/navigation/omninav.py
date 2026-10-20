from __future__ import annotations

import json
import math
import mimetypes
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agentos.embodiments.base import EmbodimentAdapter
from agentos.schemas.base import utc_now
from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.schemas.enums import SkillStatus
from agentos.schemas.navigation import ExecutionReport
from agentos.schemas.observation import Observation
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.skills.base import SkillContext


@dataclass
class OmniNavSkill:
    """One OmniNav inference followed by at most one embodiment action."""

    endpoint_url: str
    embodiment: EmbodimentAdapter | None = None
    http_client: httpx.AsyncClient | None = None
    verify_tls: bool = True
    require_three_views: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    name: str = "NavigationSkill"
    descriptor = SkillDescriptor(
        name="NavigationSkill",
        description="Run one source-grounded OmniNav waypoint inference and one bounded embodiment action.",
        input_schema={
            "instruction": {"type": "string"},
            "observation_id": {"type": "string"},
            "reset": {"type": "boolean"},
        },
        output_schema={
            "trajectory": {"type": "array", "coord_frame": "[forward,left] meters"},
            "omninav_stop": {"type": "boolean", "scope": "local skill evidence"},
            "execution_report": {"type": "object"},
        },
        required_modalities=["rgb", "pose"],
        supported_embodiments=["habitat-gs", "robot-ros2"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["proposes_motion", "executes_one_bounded_action_when_embodiment_is_attached"],
    )

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        context.cancellation.raise_if_cancelled()
        started_at = utc_now()
        try:
            instruction = str(request.inputs.get("instruction") or request.command).strip()
            if not instruction:
                raise ValueError("instruction is required")
            files, view_order = self._build_files(observation)
            payload = self._build_payload(request, observation, instruction, view_order)
            response = await self._post(files, payload, request.timeout_s)
            outputs = self._normalize_response(response)

            evidence = [
                Evidence(
                    kind="omninav_inference",
                    summary=f"OmniNav inferred {len(outputs['trajectory'])} waypoint(s) from observation {observation.trace.observation_id}.",
                    ref=f"observation:{observation.trace.observation_id}",
                    data={
                        "endpoint": self.endpoint_url,
                        "input_view_order": view_order,
                        "trajectory_coord_frame": "[forward,left] meters",
                        "omninav_stop": outputs["omninav_stop"],
                    },
                )
            ]

            provisional = SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.SUCCESS,
                summary="OmniNav inference completed.",
                evidence=evidence,
                outputs=outputs,
                stop_reason="omninav_local_stop" if outputs["omninav_stop"] else None,
                started_at=started_at,
                completed_at=utc_now(),
            )
            if self.embodiment is not None:
                execution = await self.embodiment.apply_skill_result(provisional)
                report = self._execution_report(execution, outputs["omninav_stop"])
                outputs["execution"] = execution
                outputs["execution_report"] = report.model_dump(mode="json")
                evidence.append(
                    Evidence(
                        kind="embodiment_execution",
                        summary=f"Embodiment executed bounded action {execution.get('action', 'unknown')}.",
                        ref=f"skill_run:{request.trace.skill_run_id}",
                        data=report.model_dump(mode="json"),
                    )
                )
                provisional = provisional.model_copy(
                    update={
                        "summary": f"OmniNav inference completed; bounded action={execution.get('action', 'unknown')}.",
                        "evidence": evidence,
                        "outputs": outputs,
                        "completed_at": utc_now(),
                    }
                )
            return provisional
        except (ValueError, OSError, httpx.HTTPError, json.JSONDecodeError) as exc:
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.FAILURE,
                summary="OmniNav skill failed before a valid bounded action was produced.",
                error=str(exc),
                started_at=started_at,
                completed_at=utc_now(),
            )

    def _build_files(self, observation: Observation) -> tuple[dict[str, tuple[str, bytes, str]], list[str]]:
        required = ("left", "front", "right") if self.require_three_views else ("front",)
        missing = [view for view in required if not observation.rgb_refs.get(view)]
        if missing:
            raise ValueError(f"missing required RGB view(s): {', '.join(missing)}")

        view_order = [view for view in ("left", "front", "right") if observation.rgb_refs.get(view)]
        field_by_view = {"left": "image_left", "front": "image", "right": "image_right"}
        files: dict[str, tuple[str, bytes, str]] = {}
        for view in view_order:
            path = self._local_file(observation.rgb_refs[view], f"RGB {view}")
            files[field_by_view[view]] = (path.name, path.read_bytes(), self._mime(path, "image/jpeg"))

        depth_ref = observation.depth_refs.get("front")
        if depth_ref:
            path = self._local_file(depth_ref, "front depth")
            files["depth"] = (path.name, path.read_bytes(), self._mime(path, "image/png"))
        else:
            # The legacy /eval schema requires this part even in visual-only mode;
            # the server ignores it unless depth_mode is explicitly "sensor".
            files["depth"] = ("depth_placeholder.png", self._placeholder_depth_png(), "image/png")
        return files, view_order

    @staticmethod
    def _placeholder_depth_png() -> bytes:
        def chunk(kind: bytes, data: bytes) -> bytes:
            payload = kind + data
            return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

        signature = b"\x89PNG\r\n\x1a\n"
        header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
        pixels = zlib.compress(b"\x00\x00")
        return signature + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")

    @staticmethod
    def _local_file(ref: str, label: str) -> Path:
        path = Path(ref)
        if not path.is_file():
            raise ValueError(f"{label} reference is not a readable local file: {ref}")
        return path

    @staticmethod
    def _mime(path: Path, fallback: str) -> str:
        return mimetypes.guess_type(path.name)[0] or fallback

    def _build_payload(
        self,
        request: SkillRequest,
        observation: Observation,
        instruction: str,
        view_order: list[str],
    ) -> dict[str, Any]:
        frame_index = int(request.inputs.get("frame_index", 0))
        pose = observation.pose.model_dump(mode="json") if observation.pose else None
        return {
            "idx": frame_index,
            "frame_index": frame_index,
            "instruction": instruction,
            "reset": bool(request.inputs.get("reset", False)),
            "simulator": str(observation.metadata.get("embodiment", "cee-agentos")),
            "robot_pose": pose,
            "robot_metrics": {
                "rgb_view_order": view_order,
                "trajectory_coord_frame": "[forward,left] meters",
                "source_observation_id": observation.trace.observation_id,
            },
        }

    async def _post(
        self,
        files: dict[str, tuple[str, bytes, str]],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        data = {"json": json.dumps(payload, ensure_ascii=False)}
        if self.http_client is not None:
            response = await self.http_client.post(self.endpoint_url, files=files, data=data, timeout=timeout_s)
        else:
            async with httpx.AsyncClient(verify=self.verify_tls, headers=self.headers) as client:
                response = await client.post(self.endpoint_url, files=files, data=data, timeout=timeout_s)
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, dict):
            raise ValueError("OmniNav response must be a JSON object")
        return decoded

    @classmethod
    def _normalize_response(cls, response: dict[str, Any]) -> dict[str, Any]:
        arrive_pred = response.get("omninav_arrive_pred", response.get("arrive_pred"))
        local_stop = cls._truthy(arrive_pred) or cls._truthy(response.get("episode_done"))
        raw_trajectory = response.get("trajectory")
        if raw_trajectory is None:
            raw_trajectory = response.get("model_trajectory")
        if raw_trajectory is None and local_stop:
            raw_trajectory = []
        trajectory = cls._normalize_trajectory(raw_trajectory)
        outputs: dict[str, Any] = {
            "trajectory": trajectory,
            "model_trajectory": trajectory,
            "trajectory_coord_frame": "[forward,left] meters",
            "omninav_stop": local_stop,
            "arrive_pred": arrive_pred,
        }
        passthrough = (
            "model_goal_r",
            "model_goal_theta",
            "model_infer_sec",
            "latency_sec",
            "planned_actions",
            "model_name",
            "model_path",
            "ckpt",
        )
        for key in passthrough:
            if key in response:
                outputs[key] = response[key]
        return outputs

    @staticmethod
    def _normalize_trajectory(value: Any) -> list[list[float]]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("OmniNav trajectory must be an array")
        trajectory: list[list[float]] = []
        for index, point in enumerate(value):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(f"OmniNav trajectory point {index} must contain [forward,left]")
            forward, left = float(point[0]), float(point[1])
            if not math.isfinite(forward) or not math.isfinite(left):
                raise ValueError(f"OmniNav trajectory point {index} is not finite")
            trajectory.append([forward, left])
        return trajectory

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "stop", "done"}
        if isinstance(value, (int, float)):
            return math.isfinite(float(value)) and float(value) > 0.0
        return bool(value)

    @staticmethod
    def _execution_report(execution: dict[str, object], model_stop: bool) -> ExecutionReport:
        before = execution.get("pose_before") if isinstance(execution.get("pose_before"), dict) else {}
        after = execution.get("pose_after") if isinstance(execution.get("pose_after"), dict) else {}
        before_yaw = float(before.get("yaw", 0.0)) if before else 0.0
        after_yaw = float(after.get("yaw", before_yaw)) if after else before_yaw
        yaw_delta = math.atan2(math.sin(after_yaw - before_yaw), math.cos(after_yaw - before_yaw))
        action = str(execution.get("action", "unknown"))
        return ExecutionReport(
            command_ids=[action],
            controller="embodiment_adapter",
            distance_m=max(0.0, float(execution.get("distance_m", 0.0))),
            yaw_change_rad=yaw_delta,
            collision_count=1 if bool(execution.get("collided")) else 0,
            final_pose=dict(after),
            stopped=model_stop or action in {"stop", "hold"},
            stop_source="omninav_local_stop" if model_stop else ("embodiment_hold" if action == "hold" else None),
            metadata={"action": action, "reason": execution.get("reason")},
        )
