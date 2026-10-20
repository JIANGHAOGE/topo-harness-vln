from __future__ import annotations

import argparse
import asyncio
import json
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentos.api.habitat_tool_server import ThreadBoundHabitatAdapter
from agentos.embodiments.habitat_gs import HabitatGSAdapter, HabitatGSConfig
from agentos.schemas.base import utc_now
from agentos.schemas.observation import Observation
from agentos.schemas.trace import TraceContext


class MinimalVLNService:
    """Minimal observe/step boundary around one Habitat episode."""

    ACTIONS = {0: "stop", 1: "forward", 2: "left", 3: "right"}
    CAMERA_VIEWS = ("left", "front", "right")
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_HFOV = 120.0
    CAMERA_VFOV = 90.0

    def __init__(self, adapter: ThreadBoundHabitatAdapter, *, action_budget: int = 500) -> None:
        self.adapter = adapter
        self.action_budget = action_budget
        self.episode_id = self._new_episode_id()
        self.actions_used = 0
        self.observation_index = 0
        self.terminal = False
        self._latest_observation: dict[str, Any] | None = None
        self._last_action: int | None = None
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "minimal-interface-vln",
            "simulator_ready": self.adapter.sim is not None,
            "camera": {
                "views": list(self.CAMERA_VIEWS),
                "width": self.CAMERA_WIDTH,
                "height": self.CAMERA_HEIGHT,
                "hfov_deg": self.CAMERA_HFOV,
                "vfov_deg": self.CAMERA_VFOV,
                "eye_height_m": 0.45,
                "yaw_deg": {"left": 90.0, "front": 0.0, "right": -90.0},
            },
            "actions": {"0": "STOP", "1": "forward 0.25m", "2": "left 15deg", "3": "right 15deg"},
            "max_actions_per_call": 1,
            "action_budget": self.action_budget,
        }

    def observe(self) -> dict[str, Any]:
        with self._lock:
            self._require_active()
            observation = asyncio.run(self.adapter.observe(self._trace()))
            payload = observation.model_dump(mode="json")
            self._latest_observation = payload
            image_paths = {
                view: observation.rgb_refs[view]
                for view in self.CAMERA_VIEWS
                if view in observation.rgb_refs
            }
            missing = [view for view in self.CAMERA_VIEWS if view not in image_paths]
            if missing:
                raise RuntimeError(f"RGB observations are unavailable: {missing}")
            # This is the complete model-facing observation contract.
            return {
                "ok": True,
                "view_order": list(self.CAMERA_VIEWS),
                "image_paths": image_paths,
            }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._require_active()
            actions = payload.get("actions")
            if not isinstance(actions, list) or not actions:
                raise ValueError("actions must be a non-empty list")
            if len(actions) != 1:
                raise ValueError("exactly one action is allowed per step call")
            if any(isinstance(action, bool) or not isinstance(action, int) for action in actions):
                raise ValueError("actions must contain only integer Habitat action IDs")
            unsupported = [action for action in actions if action not in self.ACTIONS]
            if unsupported:
                raise ValueError(f"unsupported action IDs: {unsupported}")

            action = actions[0]
            self.adapter.step(self.ACTIONS[action])
            self.actions_used += 1
            self._last_action = action
            if action == 0:
                self.terminal = True
            if self.actions_used >= self.action_budget:
                self.terminal = True
            return {
                "executed": 1,
                "remaining_budget": max(0, self.action_budget - self.actions_used),
                "terminal": self.terminal,
            }

    def relay_live(self) -> dict[str, Any]:
        with self._lock:
            observation = asyncio.run(self.adapter.observe_live(self._trace()))
            payload = observation.model_dump(mode="json")
            self._latest_observation = payload
            return {
                "ok": True,
                "health": self.health(),
                "observation": payload,
                "episode": {
                    "episode_id": self.episode_id,
                    "status": "terminal" if self.terminal else "running",
                    "actions_used": self.actions_used,
                    "remaining_budget": max(0, self.action_budget - self.actions_used),
                    "last_action": self._last_action,
                },
            }

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.adapter.close()
            self.adapter.start()
            self.episode_id = self._new_episode_id()
            self.actions_used = 0
            self.observation_index = 0
            self.terminal = False
            self._latest_observation = None
            self._last_action = None
            return {"ok": True, "episode_id": self.episode_id}

    def dispatch(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "GET" and path == "/health":
            return self.health()
        if method == "GET" and path == "/relay/live":
            return self.relay_live()
        if method == "POST" and path == "/observe":
            return self.observe()
        if method == "POST" and path == "/step":
            return self.step(payload)
        if method == "POST" and path == "/episode/reset":
            return self.reset()
        raise LookupError(f"unknown endpoint: {method} {path}")

    def _trace(self) -> TraceContext:
        now = utc_now()
        index = self.observation_index
        self.observation_index += 1
        return TraceContext(
            episode_id=self.episode_id,
            task_id="minimal-vln",
            plan_revision=0,
            skill_run_id=f"minimal-vln-{index}",
            observation_id=f"obs-{index}",
            frame_timestamp=now,
            pose_timestamp=now,
            deadline=now + timedelta(seconds=30),
        )

    def _require_active(self) -> None:
        if self.terminal:
            raise ValueError("episode is terminal; start a fresh episode")

    @staticmethod
    def _new_episode_id() -> str:
        return f"minimal-vln-{utc_now().strftime('%Y%m%dT%H%M%S%f')}"


def make_handler(service: MinimalVLNService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def log_message(self, format: str, *args: object) -> None:
            if self.path != "/relay/live":
                print(f"[minimal-vln] {self.address_string()} {format % args}")

        def _handle(self, method: str) -> None:
            try:
                payload = self._read_json() if method == "POST" else {}
                self._write_json(200, service.dispatch(method, self.path, payload))
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
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal-interface agentic VLN Habitat service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
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
    parser.add_argument("--action-budget", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action_budget < 1:
        raise ValueError("action budget must be positive")
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path("logs/minimal_vln") / timestamp).resolve()
    raw_adapter = HabitatGSAdapter(
        HabitatGSConfig(
            dataset_root=args.dataset_root,
            scene_id=args.scene_id,
            output_dir=output_dir,
            width=MinimalVLNService.CAMERA_WIDTH,
            height=MinimalVLNService.CAMERA_HEIGHT,
            hfov=MinimalVLNService.CAMERA_HFOV,
            vfov=MinimalVLNService.CAMERA_VFOV,
            eye_height=0.45,
            gpu_device_id=args.gpu_device_id,
            forward_step_m=0.25,
            turn_angle_deg=15.0,
            sensor_views=MinimalVLNService.CAMERA_VIEWS,
            start_position=tuple(args.start_position) if args.start_position is not None else None,
            start_yaw_deg=args.start_yaw_deg,
        )
    )
    adapter = ThreadBoundHabitatAdapter(raw_adapter)
    adapter.start()
    service = MinimalVLNService(adapter, action_budget=args.action_budget)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Minimal-interface VLN service: http://{args.host}:{args.port}")
    print(f"Artifacts: {output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        adapter.shutdown()


if __name__ == "__main__":
    main()
