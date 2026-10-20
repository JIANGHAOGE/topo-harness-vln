from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.skill import SkillResult
from agentos.schemas.trace import TraceContext


@dataclass(frozen=True)
class HabitatGSConfig:
    dataset_root: Path
    scene_id: str
    output_dir: Path
    width: int = 640
    height: int = 480
    hfov: float = 100.0
    vfov: float | None = None
    eye_height: float = 0.45
    enable_depth: bool = False
    gpu_device_id: int = 0
    forward_step_m: float = 0.2
    turn_angle_deg: float = 15.0
    sensor_views: tuple[str, ...] = ("left", "front", "right")
    start_position: tuple[float, float, float] | None = None
    start_yaw_deg: float | None = None


@dataclass
class HabitatGSAdapter:
    """AgentOS boundary for a persistent, headless Habitat-GS simulator."""

    config: HabitatGSConfig | None = None
    sim: Any = None
    bridge: Any = None
    output_dir: Path | None = None
    minimum_waypoint_distance_m: float = 0.01
    pure_pursuit_lookahead_m: float = 0.6
    pure_pursuit_speed_mps: float = 0.5
    pure_pursuit_control_period_s: float = 0.4
    pure_pursuit_step_m: float = 0.2
    pure_pursuit_max_turn_rad: float = math.radians(10.0)
    topdown_resolution: int = 512
    name: str = "habitat-gs"
    _observation_index: int = field(default=0, init=False)
    _egl_app: Any = field(default=None, init=False, repr=False)
    _runtime_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _topdown_base: Any = field(default=None, init=False, repr=False)
    _topdown_origin_xz: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _topdown_meters_per_pixel: float | None = field(default=None, init=False, repr=False)
    _topdown_height: float | None = field(default=None, init=False, repr=False)

    ACTION_ALIASES: ClassVar[dict[str, str]] = {
        "forward": "move_forward",
        "back": "move_backward",
        "left": "turn_left",
        "right": "turn_right",
        "stop": "stop",
    }
    SENSOR_ORDER: ClassVar[tuple[str, ...]] = ("left", "front", "right")
    SENSOR_UUIDS: ClassVar[dict[str, str]] = {
        "left": "color_sensor_left",
        "front": "color_sensor",
        "right": "color_sensor_right",
    }
    DEPTH_SENSOR_UUIDS: ClassVar[dict[str, str]] = {
        "left": "depth_sensor_left",
        "front": "depth_sensor",
        "right": "depth_sensor_right",
    }
    SENSOR_YAW_DEG: ClassVar[dict[str, float]] = {
        "left": 90.0,
        "front": 0.0,
        "right": -90.0,
    }

    def __post_init__(self) -> None:
        if self.output_dir is None and self.config is not None:
            self.output_dir = self.config.output_dir
        if self.output_dir is None:
            raise ValueError("HabitatGSAdapter requires output_dir or HabitatGSConfig.output_dir.")
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        with self._runtime_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self.sim is not None:
            return
        if self.config is None:
            raise RuntimeError("A HabitatGSConfig is required to create the simulator.")

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.gpu_device_id))
        vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
        if vendor.exists():
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))

        import numpy as np
        from magnum.platform.egl import WindowlessApplication

        # Habitat-GS needs a windowless GL context before Simulator creation.
        self._egl_app = WindowlessApplication()

        import habitat_sim
        from habitat_sim.agent import ActionSpec, ActuationSpec
        from habitat_sim.utils.common import quat_from_angle_axis
        from habitat_sim.utils.settings import default_sim_settings, make_cfg

        dataset_root = self.config.dataset_root.resolve()
        dataset_config = dataset_root / f"{dataset_root.name}.scene_dataset_config.json"
        navmesh = dataset_root / "navmeshes" / f"{self.config.scene_id}.navmesh"
        for required in (dataset_config, navmesh):
            if not required.exists():
                raise FileNotFoundError(required)

        unknown_views = set(self.config.sensor_views) - set(self.SENSOR_ORDER)
        if unknown_views:
            raise ValueError(f"Unsupported RGB sensor views: {sorted(unknown_views)}")
        sensors = [
            self._camera_spec(
                habitat_sim,
                uuid=self.SENSOR_UUIDS[view],
                yaw_deg=self.SENSOR_YAW_DEG[view],
                sensor_type=habitat_sim.SensorType.COLOR,
            )
            for view in self.SENSOR_ORDER
            if view in self.config.sensor_views
        ]
        if self.config.enable_depth:
            sensors.extend(
                self._camera_spec(
                    habitat_sim,
                    uuid=self.DEPTH_SENSOR_UUIDS[view],
                    yaw_deg=self.SENSOR_YAW_DEG[view],
                    sensor_type=habitat_sim.SensorType.DEPTH,
                )
                for view in self.SENSOR_ORDER
                if view in self.config.sensor_views
            )

        render_height, render_width = self._render_resolution()
        settings: dict[str, Any] = dict(default_sim_settings)
        settings.update(
            {
                "scene_dataset_config_file": str(dataset_config),
                "scene": self.config.scene_id,
                "enable_physics": False,
                "window_width": render_width,
                "window_height": render_height,
                "width": render_width,
                "height": render_height,
                "default_agent": 0,
                "default_agent_navmesh": False,
                "enable_hbao": True,
                "gaussian_auto_play": False,
            }
        )
        simulator_config = make_cfg(settings)
        agent_config = simulator_config.agents[0]
        agent_config.sensor_specifications = sensors
        agent_config.action_space = {
            "move_forward": ActionSpec("move_forward", ActuationSpec(amount=self.config.forward_step_m)),
            "move_backward": ActionSpec("move_backward", ActuationSpec(amount=self.config.forward_step_m)),
            "turn_left": ActionSpec("turn_left", ActuationSpec(amount=self.config.turn_angle_deg)),
            "turn_right": ActionSpec("turn_right", ActuationSpec(amount=self.config.turn_angle_deg)),
        }

        self.sim = habitat_sim.Simulator(simulator_config)
        if not self.sim.pathfinder.is_loaded and not self.sim.pathfinder.load_nav_mesh(str(navmesh)):
            self.close()
            raise RuntimeError(f"Failed to load navmesh: {navmesh}")

        position, yaw_deg = self._resolve_start_pose(dataset_root)
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_angle_axis(math.radians(yaw_deg), np.array([0.0, 1.0, 0.0]))
        self.sim.get_agent(0).set_state(state)

    async def observe(self, trace: TraceContext) -> Observation:
        with self._runtime_lock:
            self._require_runtime()
            observation_dir = self.output_dir / "observations" / f"{self._observation_index:04d}"
            self._observation_index += 1
            return self._capture_observation(trace, observation_dir, include_depth=True)

    async def observe_live(self, trace: TraceContext) -> Observation:
        """Capture a relay frame without creating an unbounded frame archive."""
        with self._runtime_lock:
            self._require_runtime()
            return self._capture_observation(trace, self.output_dir / "live", include_depth=False)

    def _capture_observation(
        self,
        trace: TraceContext,
        observation_dir: Path,
        *,
        include_depth: bool,
    ) -> Observation:
        capture_started = time.perf_counter()
        observations = self.sim.get_sensor_observations()
        sensors_ready = time.perf_counter()
        observation_dir.mkdir(parents=True, exist_ok=True)

        rgb_refs: dict[str, str] = {}
        configured_views = self.config.sensor_views if self.config is not None else self.SENSOR_ORDER
        for view in configured_views:
            rgb = observations.get(self.SENSOR_UUIDS[view])
            if rgb is None:
                continue
            path = observation_dir / f"rgb_{view}.jpg"
            self._save_rgb(rgb, path)
            rgb_refs[view] = str(path)
        rgb_ready = time.perf_counter()

        depth_refs: dict[str, str] = {}
        if include_depth:
            for view in configured_views:
                depth = observations.get(self.DEPTH_SENSOR_UUIDS[view])
                if depth is None:
                    continue
                path = observation_dir / f"depth_{view}.png"
                self._save_depth(depth, path)
                depth_refs[view] = str(path)
        depth_ready = time.perf_counter()

        topdown_path = observation_dir / "topdown.png"
        topdown_metadata = self._save_topdown(topdown_path)
        topdown_ready = time.perf_counter()

        state = self.sim.get_agent(0).get_state()
        position = state.position
        pose = Pose2D(
            x=float(position[0]),
            y=float(position[2]),
            yaw=self._quat_to_yaw(state.rotation),
            source="habitat-sim",
            frame="habitat_world_xz",
        )
        return Observation(
            trace=trace,
            rgb_refs=rgb_refs,
            depth_refs=depth_refs,
            topdown_ref=str(topdown_path),
            pose=pose,
            text="Habitat-GS observation.",
            metadata={
                "embodiment": self.name,
                "rgb_view_order": [view for view in configured_views if view in rgb_refs],
                "habitat_position_xyz": [float(value) for value in position],
                "metric_position_xyz": [float(value) for value in position],
                "map_id": self.config.scene_id if self.config is not None else "unknown",
                "pose_confidence": 1.0,
                "capture_timing_ms": {
                    "sensor_render": round((sensors_ready - capture_started) * 1000.0, 2),
                    "rgb_encode": round((rgb_ready - sensors_ready) * 1000.0, 2),
                    "depth_encode": round((depth_ready - rgb_ready) * 1000.0, 2),
                    "topdown_render": round((topdown_ready - depth_ready) * 1000.0, 2),
                    "total": round((topdown_ready - capture_started) * 1000.0, 2),
                },
                "camera": self._camera_metadata() if self.config is not None else {},
                "topdown_map": topdown_metadata,
            },
        )

    async def apply_skill_result(self, result: SkillResult) -> dict[str, object]:
        with self._runtime_lock:
            self._require_runtime()
            trajectory = result.outputs.get("trajectory") or []
            before = self._pose_dict()
            if bool(result.outputs.get("omninav_stop")) or not trajectory:
                return self._hold(before, "model_stop_or_empty_trajectory")

            goal = self._pure_pursuit_goal(trajectory)
            if goal is None or goal[0] <= self.minimum_waypoint_distance_m:
                return self._hold(before, "trajectory_inside_deadzone")
            goal_r, goal_theta, target = goal
            return self._execute_pure_pursuit(goal_r, goal_theta, target)

    async def pixel_to_world(
        self,
        *,
        view: str,
        x_px: int,
        y_px: int,
        depth_ref: str,
        image_width: int,
        image_height: int,
        source_pose: Pose2D | None = None,
        source_position_xyz_m: list[float] | None = None,
    ) -> dict[str, object]:
        with self._runtime_lock:
            self._require_runtime()
            return self._pixel_to_world_locked(
                view=view,
                x_px=x_px,
                y_px=y_px,
                depth_ref=depth_ref,
                image_width=image_width,
                image_height=image_height,
                source_pose=source_pose,
                source_position_xyz_m=source_position_xyz_m,
            )

    async def world_to_pixel(
        self,
        *,
        world_point: list[float],
        depth_refs: dict[str, str],
        image_width: int,
        image_height: int,
    ) -> dict[str, object] | None:
        with self._runtime_lock:
            self._require_runtime()
            return self._world_to_pixel_locked(
                world_point=world_point,
                depth_refs=depth_refs,
                image_width=image_width,
                image_height=image_height,
            )

    async def plan_path(self, world_point: list[float]) -> dict[str, object]:
        with self._runtime_lock:
            self._require_runtime()
            return self._plan_path_locked(world_point)

    async def execute_world_path(
        self,
        world_path: list[list[float]],
        *,
        max_control_steps: int = 64,
        goal_tolerance_m: float = 0.3,
    ) -> dict[str, object]:
        with self._runtime_lock:
            self._require_runtime()
            return self._execute_world_path_locked(
                world_path,
                max_control_steps=max_control_steps,
                goal_tolerance_m=goal_tolerance_m,
            )

    async def follow_navmesh_path_step(
        self,
        world_path: list[list[float]],
        *,
        goal_tolerance_m: float = 0.3,
    ) -> dict[str, object]:
        with self._runtime_lock:
            self._require_runtime()
            return self._follow_navmesh_path_step_locked(world_path, goal_tolerance_m=goal_tolerance_m)

    async def observe_keyframe(self, trace: TraceContext, run_id: str, frame_index: int) -> Observation:
        with self._runtime_lock:
            self._require_runtime()
            safe_run_id = "".join(character if character.isalnum() or character in "._-" else "_" for character in run_id)
            output_dir = self.output_dir / "executions" / safe_run_id / f"{frame_index:04d}"
            observations = self.sim.get_sensor_observations()
            output_dir.mkdir(parents=True, exist_ok=True)
            rgb_refs: dict[str, str] = {}
            configured_views = self.config.sensor_views if self.config is not None else self.SENSOR_ORDER
            for view in configured_views:
                rgb = observations.get(self.SENSOR_UUIDS[view])
                if rgb is None:
                    continue
                path = output_dir / f"rgb_{view}.jpg"
                self._save_rgb(rgb, path)
                rgb_refs[view] = str(path)
            state = self.sim.get_agent(0).get_state()
            position = state.position
            return Observation(
                trace=trace,
                rgb_refs=rgb_refs,
                pose=Pose2D(
                    x=float(position[0]),
                    y=float(position[2]),
                    yaw=self._quat_to_yaw(state.rotation),
                    source="habitat-sim",
                    frame="habitat_world_xz",
                ),
                text="Habitat navmesh trajectory keyframe.",
                metadata={
                    "embodiment": self.name,
                    "trajectory_frame_index": frame_index,
                    "metric_position_xyz": [
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ],
                    "map_id": self.config.scene_id if self.config is not None else "unknown",
                    "pose_confidence": 1.0,
                    "camera": self._camera_metadata() if self.config is not None else {},
                },
            )

    def step(self, action: str) -> dict[str, object]:
        with self._runtime_lock:
            self._require_runtime()
            normalized = action.strip().lower()
            native_action = self.ACTION_ALIASES.get(normalized, normalized)
            if native_action not in set(self.ACTION_ALIASES.values()):
                raise ValueError(f"Unsupported Habitat action: {action}")
            executed = self._execute_native_action(native_action)
            return {**executed, "requested_action": normalized, "pose": executed["pose_after"]}

    def close(self) -> None:
        with self._runtime_lock:
            if self.sim is not None:
                self.sim.close()
                self.sim = None
            self._egl_app = None
            self._topdown_base = None
            self._topdown_origin_xz = None
            self._topdown_meters_per_pixel = None
            self._topdown_height = None

    def _camera_spec(self, habitat_sim: Any, *, uuid: str, yaw_deg: float, sensor_type: Any) -> Any:
        assert self.config is not None
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = list(self._render_resolution())
        spec.position = [0.0, self.config.eye_height, 0.0]
        spec.orientation = [0.0, math.radians(yaw_deg), 0.0]
        spec.hfov = self.config.hfov
        if sensor_type == habitat_sim.SensorType.DEPTH:
            spec.channels = 1
        return spec

    def _render_resolution(self) -> tuple[int, int]:
        assert self.config is not None
        if self.config.vfov is None:
            return self.config.height, self.config.width
        if not 0.0 < self.config.hfov < 180.0 or not 0.0 < self.config.vfov < 180.0:
            raise ValueError("camera FOV must be between 0 and 180 degrees")
        width = round(
            self.config.height
            * math.tan(math.radians(self.config.hfov) / 2.0)
            / math.tan(math.radians(self.config.vfov) / 2.0)
        )
        return self.config.height, max(1, width)

    def _camera_metadata(self) -> dict[str, object]:
        assert self.config is not None
        render_height, render_width = self._render_resolution()
        effective_vfov = math.degrees(
            2.0
            * math.atan(
                math.tan(math.radians(self.config.hfov) / 2.0)
                * render_height
                / render_width
            )
        )
        return {
            "output_resolution": [self.config.width, self.config.height],
            "render_resolution": [render_width, render_height],
            "hfov_deg": self.config.hfov,
            "vfov_deg": round(effective_vfov, 2),
        }

    def _resolve_start_pose(self, dataset_root: Path) -> tuple[list[float], float]:
        assert self.config is not None
        if self.config.start_position is not None:
            return list(self.config.start_position), float(self.config.start_yaw_deg or 0.0)
        pose_path = dataset_root / "configs" / "start_poses" / f"{self.config.scene_id}.start_pose.json"
        if pose_path.exists():
            payload = json.loads(pose_path.read_text(encoding="utf-8"))
            position = payload.get("start_position")
            if isinstance(position, list) and len(position) == 3:
                yaw = self.config.start_yaw_deg
                return [float(value) for value in position], float(payload.get("start_yaw_deg", 0.0) if yaw is None else yaw)
        point = self.sim.pathfinder.get_random_navigable_point()
        return [float(value) for value in point], float(self.config.start_yaw_deg or 0.0)

    def _save_rgb(self, rgb: Any, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        if self.bridge is not None and hasattr(self.bridge, "encode_jpeg"):
            try:
                temporary.write_bytes(self.bridge.encode_jpeg(rgb))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return
        import numpy as np
        from PIL import Image

        image = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
        if self.config is not None and image.size != (self.config.width, self.config.height):
            image = image.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS)
        try:
            image.save(temporary, format="JPEG", quality=90)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_depth(self, depth: Any, path: Path) -> None:
        if self.bridge is not None and hasattr(self.bridge, "encode_depth_png"):
            height, width = depth.shape[:2]
            path.write_bytes(self.bridge.encode_depth_png(depth, height, width, encoding="millimeters_u16"))
            return
        import numpy as np
        from PIL import Image

        depth_mm = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(path)

    def _save_topdown(self, path: Path) -> dict[str, object]:
        if self.bridge is not None and hasattr(self.bridge, "render_habitat_gs_topdown"):
            image = self.bridge.render_habitat_gs_topdown(
                self.sim, {}, resolution=self.topdown_resolution, append_history=False
            )
            if not self.bridge.cv2.imwrite(str(path), image):
                raise RuntimeError(f"Failed to save Habitat topdown image: {path}")
            shape = getattr(image, "shape", ())
            return {
                "image_resolution_px": [int(shape[1]), int(shape[0])] if len(shape) >= 2 else [],
                "world_frame": "habitat_world_xz",
            }
        import numpy as np
        from PIL import Image, ImageDraw
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.sim.get_agent(0).get_state()
        x, height, z = (float(value) for value in state.position)
        if self._topdown_base is None or self._topdown_height is None or abs(height - self._topdown_height) > 0.5:
            lower, upper = self.sim.pathfinder.get_bounds()
            span = max(float(upper[0] - lower[0]), float(upper[2] - lower[2]), 1.0)
            meters_per_pixel = span / float(self.topdown_resolution)
            occupancy = self.sim.pathfinder.get_topdown_view(meters_per_pixel, height)
            base = np.zeros((*occupancy.shape, 3), dtype=np.uint8)
            base[occupancy] = (92, 96, 92)
            self._topdown_base = base
            self._topdown_origin_xz = (float(lower[0]), float(lower[2]))
            self._topdown_meters_per_pixel = meters_per_pixel
            self._topdown_height = height

        canvas = self._topdown_base.copy()
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        assert self._topdown_origin_xz is not None
        assert self._topdown_meters_per_pixel is not None
        origin_x, origin_z = self._topdown_origin_xz
        meters_per_pixel = self._topdown_meters_per_pixel
        px = int((x - origin_x) / meters_per_pixel)
        py = int((z - origin_z) / meters_per_pixel)
        radius = max(3, int(min(image.size) * 0.012))
        forward = quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
        nose_x = int((x + float(forward[0]) * 0.8 - origin_x) / meters_per_pixel)
        nose_y = int((z + float(forward[2]) * 0.8 - origin_z) / meters_per_pixel)
        draw.line((px, py, nose_x, nose_y), fill=(255, 190, 65), width=max(2, radius // 2))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(238, 70, 70))
        image.save(path)
        return {
            "image_resolution_px": [image.width, image.height],
            "world_frame": "habitat_world_xz",
            "origin_world_xz_m": [origin_x, origin_z],
            "meters_per_pixel": meters_per_pixel,
            "pixel_axes": {"x": "+world_x", "y": "+world_z"},
        }

    def _agent_position(self) -> tuple[float, float, float]:
        position = self.sim.get_agent(0).get_state().position
        return float(position[0]), float(position[1]), float(position[2])

    def _pose_dict(self) -> dict[str, float]:
        state = self.sim.get_agent(0).get_state()
        return {
            "x": float(state.position[0]),
            "y": float(state.position[2]),
            "yaw": self._quat_to_yaw(state.rotation),
        }

    def _quat_to_yaw(self, rotation: Any) -> float:
        if self.bridge is not None and hasattr(self.bridge, "quat_to_yaw"):
            return float(self.bridge.quat_to_yaw(rotation))
        import numpy as np
        from habitat_sim.utils.common import quat_rotate_vector

        forward = quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0]))
        return float(math.atan2(-float(forward[0]), -float(forward[2])))

    def _pixel_to_world_locked(
        self,
        *,
        view: str,
        x_px: int,
        y_px: int,
        depth_ref: str,
        image_width: int,
        image_height: int,
        source_pose: Pose2D | None = None,
        source_position_xyz_m: list[float] | None = None,
    ) -> dict[str, object]:
        import numpy as np
        from PIL import Image

        if self.config is None:
            raise RuntimeError("Pixel projection requires HabitatGSConfig camera metadata.")
        if view not in self.config.sensor_views:
            raise ValueError(f"Camera view is not configured: {view}")
        if image_width < 1 or image_height < 1:
            raise ValueError("image dimensions must be positive")
        if not 0 <= x_px < image_width or not 0 <= y_px < image_height:
            raise ValueError("anchor pixel is outside the RGB image")

        depth_path = Path(depth_ref)
        if not depth_path.is_file():
            raise ValueError(f"Depth reference is not readable: {depth_ref}")
        depth_mm = np.asarray(Image.open(depth_path), dtype=np.float32)
        if depth_mm.ndim != 2:
            raise ValueError("Depth image must have one channel")
        depth_height, depth_width = depth_mm.shape
        u = min(depth_width - 1, max(0, round((x_px + 0.5) * depth_width / image_width - 0.5)))
        v = min(depth_height - 1, max(0, round((y_px + 0.5) * depth_height / image_height - 0.5)))
        patch = depth_mm[max(0, v - 2) : min(depth_height, v + 3), max(0, u - 2) : min(depth_width, u + 3)]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            raise ValueError("No valid depth near the anchor pixel")
        depth_m = float(np.median(valid) / 1000.0)

        hfov_rad = math.radians(self.config.hfov)
        fx = depth_width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        center_x = (depth_width - 1) / 2.0
        center_y = (depth_height - 1) / 2.0
        camera_right_m = (u - center_x) * depth_m / fx
        camera_up_m = -(v - center_y) * depth_m / fy
        relative_yaw = math.radians(self.SENSOR_YAW_DEG[view])
        forward_m = depth_m * math.cos(relative_yaw) + camera_right_m * math.sin(relative_yaw)
        left_m = depth_m * math.sin(relative_yaw) - camera_right_m * math.cos(relative_yaw)

        state = self.sim.get_agent(0).get_state()
        position = state.position
        if source_position_xyz_m is not None and len(source_position_xyz_m) != 3:
            raise ValueError("source_position_xyz_m must contain three coordinates")
        if source_pose is None:
            source_x = float(position[0])
            source_z = float(position[2])
            source_y = float(position[1])
            yaw = self._quat_to_yaw(state.rotation)
        else:
            source_x = source_pose.x
            source_z = source_pose.y
            source_y = (
                float(source_position_xyz_m[1])
                if source_position_xyz_m is not None
                else float(position[1])
            )
            yaw = source_pose.yaw
        world_x = source_x - math.sin(yaw) * forward_m - math.cos(yaw) * left_m
        world_z = source_z - math.cos(yaw) * forward_m + math.sin(yaw) * left_m
        surface_y = source_y + self.config.eye_height + camera_up_m
        return {
            "view": view,
            "pixel": [x_px, y_px],
            "depth_pixel": [u, v],
            "depth_m": depth_m,
            "camera_point_right_up_forward_m": [camera_right_m, camera_up_m, depth_m],
            "robot_point_forward_left_m": [forward_m, left_m],
            "world_point_xyz_m": [world_x, surface_y, world_z],
            "navigation_candidate_xyz_m": [world_x, source_y, world_z],
        }

    def _world_to_pixel_locked(
        self,
        *,
        world_point: list[float],
        depth_refs: dict[str, str],
        image_width: int,
        image_height: int,
    ) -> dict[str, object] | None:
        import numpy as np
        from PIL import Image

        if self.config is None:
            raise RuntimeError("World projection requires HabitatGSConfig camera metadata.")
        if len(world_point) != 3 or not all(
            math.isfinite(float(value)) for value in world_point
        ):
            raise ValueError("world_point must contain three finite coordinates")
        if image_width < 1 or image_height < 1:
            raise ValueError("image dimensions must be positive")

        state = self.sim.get_agent(0).get_state()
        forward_m, left_m = self._world_to_robot_forward_left(world_point)
        camera_up_m = float(world_point[1]) - (
            float(state.position[1]) + self.config.eye_height
        )
        hfov_rad = math.radians(self.config.hfov)
        render_height, render_width = self._render_resolution()
        vfov_rad = 2.0 * math.atan(
            math.tan(hfov_rad / 2.0) * render_height / render_width
        )
        fx = image_width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = image_height / (2.0 * math.tan(vfov_rad / 2.0))
        center_x = (image_width - 1) / 2.0
        center_y = (image_height - 1) / 2.0
        candidates: list[dict[str, object]] = []
        for view in self.config.sensor_views:
            relative_yaw = math.radians(self.SENSOR_YAW_DEG[view])
            depth_m = forward_m * math.cos(relative_yaw) + left_m * math.sin(
                relative_yaw
            )
            if depth_m <= 0.05:
                continue
            camera_right_m = forward_m * math.sin(relative_yaw) - left_m * math.cos(
                relative_yaw
            )
            x_px = round(center_x + camera_right_m * fx / depth_m)
            y_px = round(center_y - camera_up_m * fy / depth_m)
            if not 0 <= x_px < image_width or not 0 <= y_px < image_height:
                continue

            visible = True
            sensed_depth_m: float | None = None
            depth_ref = depth_refs.get(view)
            if depth_ref and Path(depth_ref).is_file():
                depth_mm = np.asarray(Image.open(depth_ref), dtype=np.float32)
                if depth_mm.ndim == 2:
                    depth_height, depth_width = depth_mm.shape
                    u = min(
                        depth_width - 1,
                        max(0, round((x_px + 0.5) * depth_width / image_width - 0.5)),
                    )
                    v = min(
                        depth_height - 1,
                        max(0, round((y_px + 0.5) * depth_height / image_height - 0.5)),
                    )
                    patch = depth_mm[
                        max(0, v - 2) : min(depth_height, v + 3),
                        max(0, u - 2) : min(depth_width, u + 3),
                    ]
                    valid = patch[np.isfinite(patch) & (patch > 0.0)]
                    if valid.size:
                        sensed_depth_m = float(np.median(valid) / 1000.0)
                        visible = depth_m <= sensed_depth_m + 0.5
            candidates.append(
                {
                    "view": view,
                    "pixel": [x_px, y_px],
                    "depth_m": float(depth_m),
                    "sensed_depth_m": sensed_depth_m,
                    "visible": visible,
                    "center_offset": abs(x_px - center_x) / max(image_width, 1),
                }
            )
        if not candidates:
            return None
        visible_candidates = [item for item in candidates if bool(item["visible"])]
        ranked = visible_candidates or candidates
        selected = min(ranked, key=lambda item: float(item["center_offset"]))
        return {key: value for key, value in selected.items() if key != "center_offset"}

    def _plan_path_locked(self, world_point: list[float]) -> dict[str, object]:
        import numpy as np

        if len(world_point) != 3 or not all(math.isfinite(float(value)) for value in world_point):
            raise ValueError("world_point must contain three finite coordinates")
        start = np.asarray(self.sim.get_agent(0).get_state().position, dtype=np.float32)
        requested_goal = np.asarray([float(value) for value in world_point], dtype=np.float32)
        requested_goal[1] = start[1]
        snapped_goal = np.asarray(self.sim.pathfinder.snap_point(requested_goal.tolist()), dtype=np.float32)
        if np.any(np.isnan(snapped_goal)) or not bool(self.sim.pathfinder.is_navigable(snapped_goal.tolist())):
            raise ValueError("Anchor point cannot be snapped to the Habitat navmesh")

        if self.bridge is not None and hasattr(self.bridge, "find_shortest_path"):
            found, points, geodesic_distance = self.bridge.find_shortest_path(
                self.sim, start.tolist(), snapped_goal.tolist()
            )
        else:
            import habitat_sim

            shortest_path = habitat_sim.ShortestPath()
            shortest_path.requested_start = start
            shortest_path.requested_end = snapped_goal
            found = bool(self.sim.pathfinder.find_path(shortest_path))
            points = shortest_path.points
            geodesic_distance = shortest_path.geodesic_distance
        if not found or not points:
            raise ValueError("Habitat pathfinder could not find a route to the anchor")

        world_path = [[float(value) for value in point] for point in points]
        local_path = [list(self._world_to_robot_forward_left(point)) for point in world_path]
        return {
            "requested_goal_xyz_m": [float(value) for value in world_point],
            "snapped_goal_xyz_m": [float(value) for value in snapped_goal],
            "world_path_xyz_m": world_path,
            "robot_path_forward_left_m": local_path,
            "geodesic_distance_m": float(geodesic_distance),
            "waypoint_count": len(world_path),
        }

    def _world_to_robot_forward_left(self, point: list[float]) -> tuple[float, float]:
        state = self.sim.get_agent(0).get_state()
        yaw = self._quat_to_yaw(state.rotation)
        delta_x = float(point[0]) - float(state.position[0])
        delta_z = float(point[2]) - float(state.position[2])
        forward = -math.sin(yaw) * delta_x - math.cos(yaw) * delta_z
        left = -math.cos(yaw) * delta_x + math.sin(yaw) * delta_z
        return forward, left

    def _execute_world_path_locked(
        self,
        world_path: list[list[float]],
        *,
        max_control_steps: int,
        goal_tolerance_m: float,
    ) -> dict[str, object]:
        if not world_path or any(len(point) != 3 for point in world_path):
            raise ValueError("world_path must contain at least one XYZ waypoint")
        if not 1 <= max_control_steps <= 1000:
            raise ValueError("max_control_steps must be between 1 and 1000")
        if goal_tolerance_m <= 0.0:
            raise ValueError("goal_tolerance_m must be positive")

        controls: list[dict[str, object]] = []
        total_distance = 0.0
        stalled_steps = 0
        reached = False
        for _ in range(max_control_steps):
            step = self._follow_world_path_step_locked(world_path, goal_tolerance_m=goal_tolerance_m)
            control = step.get("control")
            if isinstance(control, dict):
                controls.append(control)
            moved = float(control.get("distance_m", 0.0)) if isinstance(control, dict) else 0.0
            total_distance += moved
            stalled_steps = stalled_steps + 1 if moved < 1e-3 else 0
            reached = bool(step.get("reached"))
            if reached or stalled_steps >= 3:
                break

        final = world_path[-1]
        position = self.sim.get_agent(0).get_state().position
        remaining = math.hypot(float(final[0]) - float(position[0]), float(final[2]) - float(position[2]))
        reached = reached or remaining <= goal_tolerance_m
        return {
            "controller": "pure_pursuit",
            "reached": reached,
            "control_steps": len(controls),
            "distance_m": total_distance,
            "remaining_distance_m": remaining,
            "collisions": sum(bool(control.get("collided")) for control in controls),
            "stalled": stalled_steps >= 3,
            "pose_after": self._pose_dict(),
            "last_control": controls[-1] if controls else None,
        }

    def _follow_world_path_step_locked(
        self,
        world_path: list[list[float]],
        *,
        goal_tolerance_m: float,
    ) -> dict[str, object]:
        if not world_path or any(len(point) != 3 for point in world_path):
            raise ValueError("world_path must contain at least one XYZ waypoint")
        if goal_tolerance_m <= 0.0:
            raise ValueError("goal_tolerance_m must be positive")
        final = world_path[-1]
        position = self.sim.get_agent(0).get_state().position
        remaining = math.hypot(float(final[0]) - float(position[0]), float(final[2]) - float(position[2]))
        if remaining <= goal_tolerance_m:
            return {
                "reached": True,
                "stalled": False,
                "remaining_distance_m": remaining,
                "pose_after": self._pose_dict(),
                "control": None,
            }
        closest_index = min(
            range(len(world_path)),
            key=lambda index: math.hypot(
                float(world_path[index][0]) - float(position[0]),
                float(world_path[index][2]) - float(position[2]),
            ),
        )
        local_path = [self._world_to_robot_forward_left(point) for point in world_path[closest_index + 1 :]]
        if not local_path:
            local_path = [self._world_to_robot_forward_left(final)]
        goal = self._pure_pursuit_goal(local_path)
        if goal is None or goal[0] <= self.minimum_waypoint_distance_m:
            return {
                "reached": False,
                "stalled": True,
                "remaining_distance_m": remaining,
                "pose_after": self._pose_dict(),
                "control": None,
            }
        control = self._execute_pure_pursuit(goal[0], goal[1], goal[2])
        position = self.sim.get_agent(0).get_state().position
        remaining = math.hypot(float(final[0]) - float(position[0]), float(final[2]) - float(position[2]))
        moved = float(control.get("distance_m", 0.0))
        return {
            "reached": remaining <= goal_tolerance_m,
            "stalled": moved < 1e-3,
            "remaining_distance_m": remaining,
            "pose_after": self._pose_dict(),
            "control": control,
        }

    def _follow_navmesh_path_step_locked(
        self,
        world_path: list[list[float]],
        *,
        goal_tolerance_m: float,
    ) -> dict[str, object]:
        import numpy as np

        if not world_path or any(len(point) != 3 for point in world_path):
            raise ValueError("world_path must contain at least one XYZ waypoint")
        if goal_tolerance_m <= 0.0:
            raise ValueError("goal_tolerance_m must be positive")

        goal = np.asarray(world_path[-1], dtype=np.float32)
        follower = self.sim.make_greedy_follower(agent_id=0, goal_radius=goal_tolerance_m)
        next_action = follower.next_action_along(goal)
        if next_action is None:
            return {
                "reached": True,
                "stalled": False,
                "remaining_distance_m": self._distance_to_world_goal(goal),
                "pose_after": self._pose_dict(),
                "control": None,
            }

        control = self._execute_native_action(next_action)
        remaining = self._distance_to_world_goal(goal)
        reached = follower.next_action_along(goal) is None or remaining <= goal_tolerance_m
        control.update(
            controller="habitat_greedy_follower",
            reason="habitat_navmesh_step",
            remaining_distance_m=remaining,
        )
        return {
            "reached": reached,
            "stalled": float(control.get("distance_m", 0.0)) < 1e-3
            and str(control.get("action")) not in {"turn_left", "turn_right"},
            "remaining_distance_m": remaining,
            "pose_after": self._pose_dict(),
            "control": control,
        }

    def _distance_to_world_goal(self, goal: object) -> float:
        position = self.sim.get_agent(0).get_state().position
        return math.hypot(float(goal[0]) - float(position[0]), float(goal[2]) - float(position[2]))

    def _execute_native_action(self, action: object) -> dict[str, object]:
        action_name = str(action)
        before = self._pose_dict()
        if action_name == "stop":
            return {"action": action_name, "collided": False, "pose_before": before, "pose_after": before, "distance_m": 0.0}
        step_observation = self.sim.step(action)
        if self.bridge is not None and hasattr(self.bridge, "keep_agent_on_navmesh"):
            self.bridge.keep_agent_on_navmesh(self.sim)
        after = self._pose_dict()
        distance_m = math.hypot(after["x"] - before["x"], after["y"] - before["y"])
        collided = bool(getattr(self.sim, "previous_step_collided", False))
        if isinstance(step_observation, dict):
            collided = bool(step_observation.get("collided", collided))
        return {"action": action_name, "collided": collided, "pose_before": before, "pose_after": after, "distance_m": distance_m}

    def _execute_pure_pursuit(
        self,
        goal_r: float,
        goal_theta: float,
        target_forward_left: tuple[float, float],
        *,
        speed_mps: float | None = None,
        control_period_s: float | None = None,
        lookahead_m: float | None = None,
    ) -> dict[str, object]:
        import numpy as np
        from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

        state = self.sim.get_agent(0).get_state()
        before = self._pose_dict()
        position = np.asarray(state.position, dtype=np.float32)
        forward = np.asarray(quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0])), dtype=np.float32)
        left = np.asarray(quat_rotate_vector(state.rotation, np.array([-1.0, 0.0, 0.0])), dtype=np.float32)
        forward[1] = 0.0
        left[1] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1e-6)
        left /= max(float(np.linalg.norm(left)), 1e-6)

        clamped_theta = max(-self.pure_pursuit_max_turn_rad, min(self.pure_pursuit_max_turn_rad, goal_theta))
        command_speed_mps = self._positive_float(speed_mps, self.pure_pursuit_speed_mps)
        command_period_s = self._positive_float(control_period_s, self.pure_pursuit_control_period_s)
        command_lookahead_m = self._positive_float(lookahead_m, self.pure_pursuit_lookahead_m)
        step_m = min(command_speed_mps * command_period_s, goal_r)
        direction = math.cos(clamped_theta) * forward + math.sin(clamped_theta) * left
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        requested = position + step_m * direction
        requested[1] = position[1]
        filtered = np.asarray(self.sim.step_filter(position.tolist(), requested.tolist()), dtype=np.float32)
        requested_m = float(np.linalg.norm(requested[[0, 2]] - position[[0, 2]]))
        filtered_m = float(np.linalg.norm(filtered[[0, 2]] - position[[0, 2]]))
        collided = filtered_m + 1e-5 < requested_m

        if np.any(np.isnan(filtered)) or not bool(self.sim.pathfinder.is_navigable(filtered.tolist())):
            filtered = position
            collided = True
        else:
            snapped = np.asarray(self.sim.pathfinder.snap_point(filtered.tolist()), dtype=np.float32)
            if np.any(np.isnan(snapped)) or not bool(self.sim.pathfinder.is_navigable(snapped.tolist())):
                filtered = position
                collided = True
            else:
                filtered = snapped

        next_state = self.sim.get_agent(0).get_state()
        next_state.position = filtered
        delta_x = float(requested[0] - position[0])
        delta_z = float(requested[2] - position[2])
        if abs(delta_x) > 1e-6 or abs(delta_z) > 1e-6:
            yaw = (math.atan2(delta_x, delta_z) + math.pi) % (2.0 * math.pi)
            next_state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        self.sim.get_agent(0).set_state(next_state)
        after = self._pose_dict()
        distance_m = math.hypot(after["x"] - before["x"], after["y"] - before["y"])
        return {
            "action": "pure_pursuit",
            "controller": "pure_pursuit",
            "reason": "track_model_trajectory",
            "collided": collided,
            "pose_before": before,
            "pose_after": after,
            "distance_m": distance_m,
            "target_forward_left_m": list(target_forward_left),
            "target_r_m": goal_r,
            "target_theta_rad": goal_theta,
            "clamped_theta_rad": clamped_theta,
            "speed_mps": command_speed_mps,
            "control_period_s": command_period_s,
            "lookahead_m": command_lookahead_m,
            "commanded_step_m": step_m,
            "requested_step_m": requested_m,
            "filtered_step_m": filtered_m,
        }

    def _require_runtime(self) -> None:
        if self.sim is None:
            self.start()

    @staticmethod
    def _hold(pose: dict[str, float], reason: str) -> dict[str, object]:
        return {"action": "hold", "reason": reason, "collided": False, "pose_before": pose, "pose_after": pose, "distance_m": 0.0}

    def _pure_pursuit_goal(
        self,
        trajectory: object,
        *,
        lookahead_m: float | None = None,
    ) -> tuple[float, float, tuple[float, float]] | None:
        if not isinstance(trajectory, (list, tuple)):
            return None
        points: list[tuple[float, float]] = []
        for point in trajectory:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                forward, left = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(forward) and math.isfinite(left):
                points.append((forward, left))
        if not points:
            return None

        lookahead = max(
            self.minimum_waypoint_distance_m,
            self._positive_float(lookahead_m, self.pure_pursuit_lookahead_m),
        )
        previous = (0.0, 0.0)
        traversed = 0.0
        target = points[-1]
        for point in points:
            segment_forward = point[0] - previous[0]
            segment_left = point[1] - previous[1]
            segment_length = math.hypot(segment_forward, segment_left)
            if segment_length > 1e-6 and traversed + segment_length >= lookahead:
                alpha = max(0.0, min(1.0, (lookahead - traversed) / segment_length))
                target = (
                    previous[0] + alpha * segment_forward,
                    previous[1] + alpha * segment_left,
                )
                break
            traversed += segment_length
            previous = point
        goal_r = math.hypot(*target)
        return goal_r, math.atan2(target[1], target[0]) if goal_r > 0.0 else 0.0, target

    @staticmethod
    def _positive_float(value: object, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) and parsed > 0.0 else default
