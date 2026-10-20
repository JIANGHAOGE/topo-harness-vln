from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agentos.embodiments.habitat_gs import HabitatGSAdapter, HabitatGSConfig
from agentos.schemas.enums import SkillStatus
from agentos.schemas.observation import Pose2D
from agentos.schemas.skill import SkillResult


class FakeCV2:
    @staticmethod
    def imwrite(path: str, _image: object) -> bool:
        Path(path).write_bytes(b"topdown")
        return True


class FakeAgent:
    def __init__(self) -> None:
        self.state = SimpleNamespace(position=[0.0, 0.0, 0.0], rotation=0.0)

    def get_state(self):
        return self.state


class FakeSim:
    def __init__(self) -> None:
        self.agent = FakeAgent()
        self.actions: list[str] = []
        self.closed = False

    def get_agent(self, _index: int) -> FakeAgent:
        return self.agent

    def get_sensor_observations(self) -> dict[str, object]:
        image = SimpleNamespace(shape=(2, 2, 3))
        depth = SimpleNamespace(shape=(2, 2))
        return {
            "color_sensor": image,
            "color_sensor_left": image,
            "color_sensor_right": image,
            "depth_sensor": depth,
            "depth_sensor_left": depth,
            "depth_sensor_right": depth,
        }

    def step(self, action: str) -> dict[str, bool]:
        self.actions.append(action)
        if action == "move_forward":
            self.agent.state.position[2] -= 0.2
        return {"collided": False}

    def make_greedy_follower(self, *, agent_id: int, goal_radius: float):
        assert agent_id == 0
        sim = self

        class Follower:
            def next_action_along(self, goal):
                remaining = abs(float(goal[2]) - float(sim.agent.state.position[2]))
                return None if remaining <= goal_radius else "move_forward"

        return Follower()

    def close(self) -> None:
        self.closed = True


class FakeBridge:
    cv2 = FakeCV2()

    @staticmethod
    def encode_jpeg(_image: object) -> bytes:
        return b"jpeg"

    @staticmethod
    def encode_depth_png(_depth: object, _height: int, _width: int, *, encoding: str) -> bytes:
        assert encoding == "millimeters_u16"
        return b"depth"

    @staticmethod
    def render_habitat_gs_topdown(*_args, **_kwargs) -> object:
        return object()

    @staticmethod
    def quat_to_yaw(rotation: float) -> float:
        return rotation

    @staticmethod
    def keep_agent_on_navmesh(_sim: FakeSim) -> None:
        return None


def test_habitat_adapter_observes_three_views_and_uses_pure_pursuit(make_trace, tmp_path):
    async def run():
        sim = FakeSim()
        adapter = HabitatGSAdapter(sim=sim, bridge=FakeBridge(), output_dir=tmp_path)
        adapter._execute_pure_pursuit = lambda goal_r, goal_theta, target, **params: {
            "action": "pure_pursuit",
            "controller": "pure_pursuit",
            "distance_m": 0.2,
            "target_r_m": goal_r,
            "target_theta_rad": goal_theta,
            "target_forward_left_m": list(target),
            "speed_mps": adapter.pure_pursuit_speed_mps,
            "lookahead_m": adapter.pure_pursuit_lookahead_m,
            "control_period_s": adapter.pure_pursuit_control_period_s,
            **params,
        }
        trace = make_trace()
        observation = await adapter.observe(trace)
        result = SkillResult(
            trace=trace,
            skill_name="NavigationSkill",
            status=SkillStatus.SUCCESS,
            summary="trajectory ready",
            outputs={"trajectory": [[0.5, 0.0], [1.0, 0.0]], "omninav_stop": False},
        )
        control = await adapter.apply_skill_result(result)
        adapter.close()
        return sim, observation, control

    sim, observation, control = asyncio.run(run())
    assert list(observation.rgb_refs) == ["left", "front", "right"]
    assert observation.pose is not None
    assert observation.pose.frame == "habitat_world_xz"
    assert control["action"] == "pure_pursuit"
    assert control["controller"] == "pure_pursuit"
    assert control["distance_m"] == 0.2
    assert control["target_forward_left_m"] == [0.6, 0.0]
    assert control["speed_mps"] == 0.5
    assert control["lookahead_m"] == 0.6
    assert control["control_period_s"] == 0.4
    assert sim.actions == []
    assert sim.closed


def test_live_rgb_publication_replaces_a_complete_temporary_file(monkeypatch, tmp_path):
    adapter = HabitatGSAdapter(sim=FakeSim(), bridge=FakeBridge(), output_dir=tmp_path)
    target = tmp_path / "live" / "rgb_left.jpg"
    target.parent.mkdir(parents=True)
    original_write_bytes = Path.write_bytes

    def reject_direct_target_write(path, data):
        if path == target:
            raise AssertionError("live RGB must not be written directly to the published path")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", reject_direct_target_write)

    adapter._save_rgb(object(), target)

    assert target.read_bytes() == b"jpeg"
    assert not list(target.parent.glob("*.tmp"))


def test_pure_pursuit_interpolates_along_model_polyline(tmp_path):
    adapter = HabitatGSAdapter(sim=FakeSim(), bridge=FakeBridge(), output_dir=tmp_path)

    goal = adapter._pure_pursuit_goal([[0.3, 0.0], [0.3, 0.4], [0.7, 0.4]])

    assert goal is not None
    radius, theta, target = goal
    assert target == (0.3, 0.3)
    assert round(radius, 6) == round(2**0.5 * 0.3, 6)
    assert round(theta, 6) == round(3.141592653589793 / 4.0, 6)


def test_habitat_navmesh_step_uses_native_greedy_follower_action(tmp_path):
    sim = FakeSim()
    adapter = HabitatGSAdapter(sim=sim, bridge=FakeBridge(), output_dir=tmp_path)

    result = asyncio.run(
        adapter.follow_navmesh_path_step(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -0.4]],
            goal_tolerance_m=0.1,
        )
    )

    assert sim.actions == ["move_forward"]
    assert result["reached"] is False
    assert result["control"]["action"] == "move_forward"
    assert result["control"]["controller"] == "habitat_greedy_follower"
    assert result["remaining_distance_m"] == pytest.approx(0.2)


def test_front_only_sensor_profile_does_not_change_default_three_view_profile(make_trace, tmp_path):
    async def run():
        config = HabitatGSConfig(
            dataset_root=tmp_path,
            scene_id="fake",
            output_dir=tmp_path,
            sensor_views=("front",),
        )
        adapter = HabitatGSAdapter(config=config, sim=FakeSim(), bridge=FakeBridge())
        return await adapter.observe(make_trace())

    observation = asyncio.run(run())

    assert HabitatGSConfig(tmp_path, "fake", tmp_path).sensor_views == ("left", "front", "right")
    assert list(observation.rgb_refs) == ["front"]
    assert observation.metadata["rgb_view_order"] == ["front"]


def test_independent_fov_profile_uses_wider_internal_render_resolution(tmp_path):
    config = HabitatGSConfig(
        dataset_root=tmp_path,
        scene_id="fake",
        output_dir=tmp_path,
        width=640,
        height=480,
        hfov=120.0,
        vfov=90.0,
    )
    adapter = HabitatGSAdapter(config=config, sim=FakeSim(), bridge=FakeBridge())

    assert adapter._render_resolution() == (480, 831)
    assert adapter._camera_metadata() == {
        "output_resolution": [640, 480],
        "render_resolution": [831, 480],
        "hfov_deg": 120.0,
        "vfov_deg": 90.03,
    }


def test_anchor_pixels_backproject_in_front_and_left_camera_frames(tmp_path):
    async def run():
        depth_path = tmp_path / "depth.png"
        Image.fromarray(np.full((480, 831), 2000, dtype=np.uint16)).save(depth_path)
        config = HabitatGSConfig(
            dataset_root=tmp_path,
            scene_id="fake",
            output_dir=tmp_path,
            width=640,
            height=480,
            hfov=120.0,
            vfov=90.0,
            enable_depth=True,
        )
        adapter = HabitatGSAdapter(config=config, sim=FakeSim(), bridge=FakeBridge())
        front = await adapter.pixel_to_world(
            view="front",
            x_px=320,
            y_px=240,
            depth_ref=str(depth_path),
            image_width=640,
            image_height=480,
        )
        left = await adapter.pixel_to_world(
            view="left",
            x_px=320,
            y_px=240,
            depth_ref=str(depth_path),
            image_width=640,
            image_height=480,
        )
        return front, left

    front, left = asyncio.run(run())

    assert front["depth_m"] == 2.0
    assert front["world_point_xyz_m"][0] == pytest.approx(0.0, abs=0.01)
    assert front["world_point_xyz_m"][2] == pytest.approx(-2.0, abs=0.01)
    assert left["world_point_xyz_m"][0] == pytest.approx(-2.0, abs=0.01)
    assert left["world_point_xyz_m"][2] == pytest.approx(0.0, abs=0.01)


def test_pixel_backprojection_uses_source_observation_pose_after_robot_moves(tmp_path):
    async def run():
        depth_path = tmp_path / "depth.png"
        Image.fromarray(np.full((480, 831), 2000, dtype=np.uint16)).save(depth_path)
        config = HabitatGSConfig(
            dataset_root=tmp_path,
            scene_id="fake",
            output_dir=tmp_path,
            width=640,
            height=480,
            hfov=120.0,
            vfov=90.0,
            enable_depth=True,
        )
        sim = FakeSim()
        sim.agent.state.position = [10.0, 0.0, 10.0]
        adapter = HabitatGSAdapter(config=config, sim=sim, bridge=FakeBridge())
        return await adapter.pixel_to_world(
            view="front",
            x_px=320,
            y_px=240,
            depth_ref=str(depth_path),
            image_width=640,
            image_height=480,
            source_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
            source_position_xyz_m=[0.0, 0.0, 0.0],
        )

    projected = asyncio.run(run())

    assert projected["world_point_xyz_m"][0] == pytest.approx(0.0, abs=0.01)
    assert projected["world_point_xyz_m"][2] == pytest.approx(-2.0, abs=0.01)


def test_navmesh_lookahead_projects_to_visible_front_affordance_pixel(tmp_path):
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.full((480, 831), 3000, dtype=np.uint16)).save(depth_path)
    config = HabitatGSConfig(
        dataset_root=tmp_path,
        scene_id="fake",
        output_dir=tmp_path,
        width=640,
        height=480,
        hfov=120.0,
        vfov=90.0,
        enable_depth=True,
    )
    adapter = HabitatGSAdapter(config=config, sim=FakeSim(), bridge=FakeBridge())

    projected = asyncio.run(
        adapter.world_to_pixel(
            world_point=[0.0, 0.0, -2.0],
            depth_refs={"front": str(depth_path)},
            image_width=640,
            image_height=480,
        )
    )

    assert projected is not None
    assert projected["view"] == "front"
    assert projected["visible"] is True
    assert projected["pixel"][0] == pytest.approx(320, abs=1)
    assert 285 <= projected["pixel"][1] <= 300
