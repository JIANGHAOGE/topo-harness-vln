from __future__ import annotations

from collections.abc import Callable
import asyncio
import math
import re
from pathlib import Path
from typing import Any, Protocol

from agentos.providers.base import LLMProvider
from agentos.providers.vision import image_ref_content
from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.schemas.base import utc_now
from agentos.schemas.enums import SkillStatus
from agentos.schemas.hierarchical import (
    AnchorGroundingInput,
    AnchorGenerationInput,
    AnchorPixel,
    GeneratedNavigationPoint,
    LocalizationMapReport,
    LocalizationMapInput,
    PathExecutionInput,
    PathExecutionReport,
    PathPlanningInput,
    PixelToPointInput,
    PointNavigationInput,
    PointNavigationReport,
    PlannedPath,
    QwenAnchorMark,
    SpatialAnchorPoint,
    DualPixelGoal,
    TriViewPixelGoal,
    ViewpointAdjustmentInput,
    ViewpointAdjustmentReport,
)
from agentos.schemas.llm import ChatMessage
from agentos.schemas.observation import Observation
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.skills.base import SkillContext


class HierarchicalEmbodiment(Protocol):
    name: str

    async def pixel_to_world(self, **kwargs: Any) -> dict[str, object]: ...

    async def world_to_pixel(self, **kwargs: Any) -> dict[str, object] | None: ...

    async def plan_path(self, world_point: list[float]) -> dict[str, object]: ...

    async def execute_world_path(self, world_path: list[list[float]], **kwargs: Any) -> dict[str, object]: ...

    async def follow_navmesh_path_step(self, world_path: list[list[float]], **kwargs: Any) -> dict[str, object]: ...

    async def observe_keyframe(self, trace: Any, run_id: str, frame_index: int) -> Observation: ...

    def step(self, action: str) -> dict[str, object]: ...


class AnchorGroundingSkill:
    name = "AnchorGroundingSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Use Qwen vision once to ground a visible semantic anchor to its detection-box bottom center.",
        input_schema=AnchorGroundingInput.model_json_schema(),
        output_schema=AnchorPixel.model_json_schema(),
        required_modalities=["rgb"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
        default_timeout_s=60.0,
    )

    def __init__(self, provider_factory: Callable[[], LLMProvider]) -> None:
        self._provider_factory = provider_factory

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = AnchorGroundingInput.model_validate(request.inputs)
        missing_views = [view for view in ("left", "front", "right") if view not in observation.rgb_refs]
        if missing_views:
            raise ValueError(f"RGB views are unavailable: {missing_views}")
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, "Grounding visible anchor across synchronized left/front/right views.")
        allowed_views = tuple(inputs.allowed_views) or ("left", "front", "right")
        allowed_labels = ", ".join(view.upper() for view in allowed_views)
        prompt = (
            f"Locate the navigation anchor {inputs.anchor!r} in the supplied current camera image. "
            "The canonical camera indices are LEFT=0, FRONT=1, and RIGHT=2. "
            f"For this route step, only these camera views are allowed: {allowed_labels}. "
            "Do not select an anchor from any other view. "
            f"{inputs.direction_hint} "
            "Do not infer the panel from the task wording. "
            "Return a tight bounding box around the requested object, doorway, opening, or visible region. "
            "The bbox coordinates must be local to the selected image and normalized from 0 to 1000, "
            "with top-left=(0,0) and bottom-right=(1000,1000). Ensure x_min < x_max and y_min < y_max. "
            "Set anchor_visible=true only when the requested anchor is clearly present in the selected image. "
            "If it is absent from the supplied image, set anchor_visible=false, confidence=0, "
            "and set all bbox coordinates to 0."
        )
        self._compose_triptych(
            observation,
            inputs.image_width,
            inputs.image_height,
            request.trace.skill_run_id,
        )
        provider = self._provider_factory()
        try:
            async def ground_one(view: str) -> tuple[str, QwenAnchorMark]:
                panel_index = ("left", "front", "right").index(view)
                content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\nInspect only the robot-{view.upper()} image in this call. "
                            f"Keep panel_index={panel_index}; it is the canonical camera index, not an "
                            "image-region classification."
                        ),
                    }
                ]
                content.extend(
                    image_ref_content(
                        f"IMAGE/PANEL {panel_index} {view.upper()} ONLY",
                        observation.rgb_refs[view],
                    )
                )
                response = await provider.complete(
                    [
                        ChatMessage(
                            role="system",
                            content=(
                                f"You are grounding only the robot-{view.upper()} camera image. "
                                "Return only JSON matching the provided schema."
                            ),
                        ),
                        ChatMessage(role="user", content=content),
                    ],
                    response_schema=QwenAnchorMark.model_json_schema(),
                )
                if not isinstance(response.parsed_json, dict):
                    raise ValueError(f"Qwen {view} anchor grounding returned no JSON object")
                mark = QwenAnchorMark.model_validate(response.parsed_json)
                if mark.panel_index != panel_index:
                    raise ValueError(
                        f"Qwen {view} grounding changed panel_index to {mark.panel_index}"
                    )
                return view, mark

            candidates = await asyncio.gather(
                *(ground_one(view) for view in allowed_views)
            )
        finally:
            await provider.close()
        selected_view, mark = max(
            candidates,
            key=lambda item: item[1].confidence if item[1].anchor_visible else -1.0,
        )
        image_ref = observation.rgb_refs.get(selected_view)
        if image_ref is None:
            raise ValueError(f"Qwen selected an unavailable camera view: {selected_view}")
        bbox_px = None
        x_px = 0
        y_px = 0
        if mark.anchor_visible:
            x_min_normalized, x_max_normalized = sorted((mark.x_min, mark.x_max))
            y_min_normalized, y_max_normalized = sorted((mark.y_min, mark.y_max))
            if x_min_normalized == x_max_normalized or y_min_normalized == y_max_normalized:
                raise ValueError("Qwen returned a zero-area anchor bbox")
            bbox_px = [
                round(x_min_normalized * (inputs.image_width - 1) / 1000.0),
                round(y_min_normalized * (inputs.image_height - 1) / 1000.0),
                round(x_max_normalized * (inputs.image_width - 1) / 1000.0),
                round(y_max_normalized * (inputs.image_height - 1) / 1000.0),
            ]
            x_min, y_min, x_max, y_max = bbox_px
            x_px = round((x_min + x_max) / 2.0)
            y_px = y_max
        grounded = AnchorPixel(
            anchor=inputs.anchor,
            view=selected_view,
            x_px=x_px,
            y_px=y_px,
            anchor_visible=mark.anchor_visible,
            confidence=mark.confidence if mark.anchor_visible else 0.0,
            evidence_summary=mark.evidence_summary,
            model_coordinate_space="normalized_1000",
            bbox_px=bbox_px,
        )
        annotated_ref = (
            self._annotate_anchor(image_ref, grounded, request.trace.skill_run_id)
            if grounded.anchor_visible
            else None
        )
        grounded = grounded.model_copy(update={"annotated_ref": annotated_ref})
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=(
                f"Grounded {inputs.anchor!r} at ({grounded.x_px}, {grounded.y_px}) in {grounded.view}."
                if grounded.anchor_visible
                else f"Anchor {inputs.anchor!r} was not visible in the synchronized three-view observation."
            ),
            evidence=[Evidence(kind="rgb_pixel", summary=grounded.evidence_summary, ref=annotated_ref)],
            outputs=grounded.model_dump(mode="json"),
            compressed_history=f"Qwen grounded one visible anchor in the {grounded.view} view.",
        )

    @staticmethod
    def _compose_triptych(
        observation: Observation,
        width: int,
        height: int,
        skill_run_id: str,
    ) -> str:
        from PIL import Image, ImageDraw, ImageFont

        source = Path(observation.rgb_refs["front"])
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", skill_run_id)
        output = source.with_name(f"anchor_triptych_{safe_run_id}.jpg")
        canvas = Image.new("RGB", (width * 3, height), color=(8, 12, 12))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        for panel_index, view in enumerate(("left", "front", "right")):
            image = Image.open(observation.rgb_refs[view]).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            offset_x = panel_index * width
            canvas.paste(image, (offset_x, 0))
            label = f"PANEL {panel_index}  {view.upper()}"
            box = draw.textbbox((0, 0), label, font=font)
            label_width = box[2] - box[0] + 18
            label_height = box[3] - box[1] + 12
            draw.rectangle(
                (offset_x + 8, 8, offset_x + 8 + label_width, 8 + label_height),
                fill=(8, 12, 12),
            )
            draw.text((offset_x + 17, 14), label, fill=(255, 220, 80), font=font)
        canvas.save(output, quality=92)
        return str(output)

    @staticmethod
    def _annotate_anchor(
        image_ref: str,
        grounded: AnchorPixel,
        skill_run_id: str,
        *,
        primary_role: str = "anchor",
        secondary_target: AnchorPixel | None = None,
        secondary_affordance: AnchorPixel | None = None,
    ) -> str:
        from PIL import Image, ImageDraw, ImageFont

        source = Path(image_ref)
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", skill_run_id)
        output = source.with_name(f"anchor_{safe_run_id}_{grounded.view}.jpg")
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        radius = max(10, round(min(image.size) * 0.025))
        width = max(3, radius // 4)
        font = ImageFont.load_default()

        def draw_mark(mark: AnchorPixel, role: str) -> None:
            x, y = mark.x_px, mark.y_px
            is_target = role == "target"
            marker_color = (52, 211, 235) if is_target else (235, 55, 55)
            bbox_color = (52, 211, 235) if is_target else (255, 210, 70)
            if mark.bbox_px is not None:
                draw.rectangle(tuple(mark.bbox_px), outline=bbox_color, width=width)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=marker_color,
                width=width,
            )
            draw.line(
                (x - radius * 2, y, x + radius * 2, y),
                fill=marker_color,
                width=width,
            )
            draw.line(
                (x, y - radius * 2, x, y + radius * 2),
                fill=marker_color,
                width=width,
            )
            label = (
                f"{role.upper()} {mark.anchor} conf={mark.confidence:.2f} ({x}, {y})"
            )
            box = draw.textbbox((0, 0), label, font=font)
            label_width = box[2] - box[0] + 12
            label_height = box[3] - box[1] + 10
            label_x = min(
                max(4, x + radius + 8),
                max(4, image.width - label_width - 4),
            )
            label_y = min(
                max(4, y - label_height // 2),
                max(4, image.height - label_height - 4),
            )
            draw.rectangle(
                (label_x, label_y, label_x + label_width, label_y + label_height),
                fill=(20, 24, 23),
            )
            draw.text(
                (label_x + 6, label_y + 5),
                label,
                fill=(255, 255, 255),
                font=font,
            )

        draw_mark(grounded, primary_role)
        if secondary_target is not None:
            if secondary_target.view != grounded.view:
                raise ValueError("secondary target must use the annotated camera view")
            draw_mark(secondary_target, "target")
        if secondary_affordance is not None:
            if secondary_affordance.view != grounded.view:
                raise ValueError("secondary affordance must use the annotated camera view")
            draw_mark(secondary_affordance, "affordance")
        image.save(output, quality=92)
        return str(output)


class PixelToPointSkill:
    name = "PixelToPointSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Backproject an anchor pixel with the matching Habitat depth camera into world coordinates.",
        input_schema=PixelToPointInput.model_json_schema(),
        output_schema=SpatialAnchorPoint.model_json_schema(),
        required_modalities=["depth", "pose"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
    )

    def __init__(self, embodiment: HierarchicalEmbodiment) -> None:
        self._embodiment = embodiment

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = PixelToPointInput.model_validate(request.inputs)
        if inputs.source_observation_id != observation.trace.observation_id:
            raise ValueError("Pixel source observation does not match the depth observation")
        depth_ref = observation.depth_refs.get(inputs.view)
        if depth_ref is None:
            raise ValueError(f"Depth view is unavailable: {inputs.view}")
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, "Backprojecting anchor pixel with Habitat depth.")
        projected = await self._embodiment.pixel_to_world(
            view=inputs.view,
            x_px=inputs.x_px,
            y_px=inputs.y_px,
            depth_ref=depth_ref,
            image_width=inputs.image_width,
            image_height=inputs.image_height,
            source_pose=observation.pose,
            source_position_xyz_m=observation.metadata.get("metric_position_xyz"),
        )
        point = SpatialAnchorPoint.model_validate(
            {
                "source_observation_id": inputs.source_observation_id,
                **projected,
                "raw_depth_m": projected.get("depth_m"),
                "max_depth_m": inputs.max_depth_m,
            }
        )
        point = self._limit_projection_depth(point, observation, inputs.max_depth_m)
        depth_summary = f"Depth {point.depth_m:.3f} m."
        if point.depth_limited:
            depth_summary = (
                f"Raw depth {point.raw_depth_m:.3f} m limited to {point.max_depth_m:.3f} m "
                "along the same projection ray."
            )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=f"Projected anchor to world point {point.world_point_xyz_m}.",
            evidence=[Evidence(kind="depth_pixel", summary=depth_summary, ref=depth_ref)],
            outputs=point.model_dump(mode="json"),
        )

    @staticmethod
    def _limit_projection_depth(
        point: SpatialAnchorPoint,
        observation: Observation,
        max_depth_m: float,
    ) -> SpatialAnchorPoint:
        raw_depth = point.raw_depth_m or point.depth_m
        if raw_depth <= max_depth_m:
            return point.model_copy(
                update={
                    "raw_depth_m": raw_depth,
                    "depth_limited": False,
                    "max_depth_m": max_depth_m,
                }
            )
        origin = observation.metadata.get("metric_position_xyz")
        if not isinstance(origin, list) or len(origin) != 3:
            if observation.pose is None:
                raise ValueError("Depth limiting requires a metric robot pose")
            origin = [observation.pose.x, point.navigation_candidate_xyz_m[1], observation.pose.y]
        scale = max_depth_m / raw_depth

        def scaled(values: list[float]) -> list[float]:
            return [float(value) * scale for value in values]

        def scaled_world(values: list[float]) -> list[float]:
            return [
                float(origin[index]) + (float(values[index]) - float(origin[index])) * scale
                for index in range(3)
            ]

        return point.model_copy(
            update={
                "depth_m": max_depth_m,
                "raw_depth_m": raw_depth,
                "depth_limited": True,
                "max_depth_m": max_depth_m,
                "camera_point_right_up_forward_m": scaled(point.camera_point_right_up_forward_m),
                "robot_point_forward_left_m": scaled(point.robot_point_forward_left_m),
                "world_point_xyz_m": scaled_world(point.world_point_xyz_m),
                "navigation_candidate_xyz_m": scaled_world(point.navigation_candidate_xyz_m),
            }
        )


class PathPlanningSkill:
    name = "PathPlanningSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Snap a spatial anchor to the Habitat navmesh and generate a shortest path.",
        input_schema=PathPlanningInput.model_json_schema(),
        output_schema=PlannedPath.model_json_schema(),
        required_modalities=["pose", "navmesh"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
    )

    def __init__(self, embodiment: HierarchicalEmbodiment) -> None:
        self._embodiment = embodiment

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = PathPlanningInput.model_validate(request.inputs)
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, "Planning navmesh path to spatial anchor.")
        planned = PlannedPath.model_validate(await self._embodiment.plan_path(inputs.world_point_xyz_m))
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=f"Planned {planned.geodesic_distance_m:.2f} m route with {planned.waypoint_count} waypoints.",
            outputs=planned.model_dump(mode="json"),
        )


class PathExecutionSkill:
    name = "PathExecutionSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Execute a navmesh goal with one Habitat greedy-follower action per control cycle.",
        input_schema=PathExecutionInput.model_json_schema(),
        output_schema=PathExecutionReport.model_json_schema(),
        required_modalities=["pose", "navmesh"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["moves_robot"],
        default_timeout_s=120.0,
    )

    def __init__(self, embodiment: HierarchicalEmbodiment) -> None:
        self._embodiment = embodiment

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = PathExecutionInput.model_validate(request.inputs)
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, "Executing path with Habitat navmesh actions.")
        controls: list[dict[str, object]] = []
        frames = [self._frame_payload(0, observation)]
        stalled_steps = 0
        control: dict[str, object] = {
            "reached": False,
            "stalled": False,
            "remaining_distance_m": 0.0,
            "pose_after": observation.pose.model_dump(mode="json") if observation.pose is not None else {},
        }
        for frame_index in range(1, inputs.max_control_steps + 1):
            context.cancellation.raise_if_cancelled()
            control = await self._embodiment.follow_navmesh_path_step(
                inputs.world_path_xyz_m,
                goal_tolerance_m=inputs.goal_tolerance_m,
            )
            if control.get("control") is not None:
                controls.append(control["control"])
            frame_trace = request.trace.model_copy(
                update={
                    "observation_id": f"{request.trace.observation_id}-pp-{frame_index}",
                    "frame_timestamp": utc_now(),
                    "pose_timestamp": utc_now(),
                }
            )
            frame = await self._embodiment.observe_keyframe(
                frame_trace,
                request.trace.skill_run_id,
                frame_index,
            )
            frames.append(self._frame_payload(frame_index, frame))
            moved = float((control.get("control") or {}).get("distance_m", 0.0))
            action = str((control.get("control") or {}).get("action", ""))
            no_progress = moved < 1e-3 and action not in {"turn_left", "turn_right"}
            stalled_steps = stalled_steps + 1 if no_progress else 0
            await context.emit_progress(
                request,
                f"Habitat navmesh action {frame_index}.",
                remaining_distance_m=control.get("remaining_distance_m"),
                reached=control.get("reached"),
            )
            if bool(control.get("reached")) or stalled_steps >= 3:
                break
            if inputs.control_period_s > 0.0:
                await asyncio.sleep(inputs.control_period_s)

        last = controls[-1] if controls else None
        total_distance = sum(float(control.get("distance_m", 0.0)) for control in controls)
        remaining = float(control.get("remaining_distance_m", 0.0))
        reached = bool(control.get("reached"))
        report = PathExecutionReport(
            controller="habitat_greedy_follower",
            reached=reached,
            control_steps=len(controls),
            distance_m=total_distance,
            remaining_distance_m=remaining,
            collisions=sum(bool(item.get("collided")) for item in controls),
            stalled=stalled_steps >= 3,
            pose_after=dict(control.get("pose_after") or {}),
            last_control=last,
            trajectory_waypoints=[self._waypoint_payload(frame) for frame in frames],
            trajectory_keyframes=self._select_keyframes(frames),
        )
        status = SkillStatus.SUCCESS if report.reached else SkillStatus.RETRYABLE_FAILURE
        summary = (
            f"Habitat follower reached the anchor in {report.control_steps} actions."
            if report.reached
            else f"Habitat follower stopped {report.remaining_distance_m:.2f} m from the anchor."
        )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=status,
            summary=summary,
            outputs=report.model_dump(mode="json"),
            compressed_history=(
                f"Habitat follower: reached={report.reached}, actions={report.control_steps}, "
                f"distance={report.distance_m:.2f}m, collisions={report.collisions}."
            ),
        )

    @staticmethod
    def _frame_payload(frame_index: int, observation: Observation) -> dict[str, object]:
        point = observation.metadata.get("metric_position_xyz")
        if not isinstance(point, list) or len(point) != 3:
            point = (
                [observation.pose.x, 0.0, observation.pose.y]
                if observation.pose is not None
                else [0.0, 0.0, 0.0]
            )
        return {
            "frame_index": frame_index,
            "observation_id": observation.trace.observation_id,
            "timestamp": observation.trace.frame_timestamp.isoformat(),
            "pose": observation.pose.model_dump(mode="json") if observation.pose is not None else {},
            "map_id": str(observation.metadata.get("map_id") or "unknown"),
            "pose_confidence": float(observation.metadata.get("pose_confidence", 1.0)),
            "navigation_point_xyz_m": [round(float(value), 3) for value in point],
            "rgb_refs": dict(observation.rgb_refs),
        }

    @staticmethod
    def _select_keyframes(frames: list[dict[str, object]]) -> list[dict[str, object]]:
        if len(frames) <= 4:
            return frames
        last = len(frames) - 1
        indices = sorted({round(position * last / 3) for position in range(4)})
        return [frames[index] for index in indices]

    @staticmethod
    def _waypoint_payload(frame: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in frame.items() if key != "rgb_refs"}


class ViewpointAdjustmentSkill:
    name = "ViewpointAdjustmentSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Execute one bounded motion or a 90-degree turn to acquire a more useful three-view observation.",
        input_schema=ViewpointAdjustmentInput.model_json_schema(),
        output_schema=ViewpointAdjustmentReport.model_json_schema(),
        required_modalities=["pose"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.LOW,
        side_effects=["moves_robot"],
    )

    def __init__(self, embodiment: HierarchicalEmbodiment) -> None:
        self._embodiment = embodiment

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = ViewpointAdjustmentInput.model_validate(request.inputs)
        if inputs.observation_id != observation.trace.observation_id:
            raise ValueError("Viewpoint adjustment source does not match the current observation")
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, f"Adjusting viewpoint with {inputs.action}.")
        turning = inputs.action in {"turn_left", "turn_right"}
        target_rotation = math.radians(inputs.rotation_degrees) if turning else 0.0
        accumulated_rotation = 0.0
        controls: list[dict[str, Any]] = []
        while not controls or (turning and accumulated_rotation < target_rotation - 1e-3):
            context.cancellation.raise_if_cancelled()
            control = self._embodiment.step(inputs.action)
            controls.append(control)
            before_yaw = (control.get("pose_before") or {}).get("yaw")
            after_yaw = (control.get("pose_after") or {}).get("yaw")
            if turning and isinstance(before_yaw, (int, float)) and isinstance(
                after_yaw, (int, float)
            ):
                delta = (float(after_yaw) - float(before_yaw) + math.pi) % (
                    2.0 * math.pi
                ) - math.pi
                accumulated_rotation += abs(delta)
            elif turning:
                break
            if len(controls) >= 36:
                break
        first = controls[0]
        last = controls[-1]
        report = ViewpointAdjustmentReport.model_validate(
            {
                "slow_step_id": inputs.slow_step_id,
                "action": last.get("action"),
                "collided": any(control.get("collided", False) for control in controls),
                "distance_m": sum(float(control.get("distance_m", 0.0)) for control in controls),
                "control_steps": len(controls),
                "requested_rotation_degrees": inputs.rotation_degrees if turning else 0.0,
                "rotation_degrees": math.degrees(accumulated_rotation),
                "pose_before": first.get("pose_before") or {},
                "pose_after": last.get("pose_after") or {},
            }
        )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=(
                f"Rotated {report.rotation_degrees:.1f} degrees with {report.action}."
                if turning
                else f"Executed one {report.action} viewpoint adjustment."
            ),
            outputs=report.model_dump(mode="json"),
            compressed_history=(
                f"Viewpoint adjustment: action={report.action}, collided={report.collided}, "
                f"distance={report.distance_m:.2f}m."
            ),
        )


class LocalizationMapSkill:
    name = "LocalizationMapSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Expose the current Habitat pose and marked navmesh top-down map for route-level reasoning.",
        input_schema=LocalizationMapInput.model_json_schema(),
        output_schema=LocalizationMapReport.model_json_schema(),
        required_modalities=["pose", "map"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
    )

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        LocalizationMapInput.model_validate(request.inputs)
        if observation.pose is None or observation.topdown_ref is None:
            raise ValueError("Habitat localization or top-down map is unavailable")
        context.cancellation.raise_if_cancelled()
        await context.emit_progress(request, "Reading Habitat localization and navmesh map.")
        report = LocalizationMapReport(
            observation_id=observation.trace.observation_id,
            pose=observation.pose,
            topdown_ref=observation.topdown_ref,
            legend={
                "gray": "navigable navmesh",
                "black": "blocked or unmapped area",
                "red": "current robot position",
                "yellow": "current heading",
            },
        )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary="Captured current Habitat pose and marked navmesh map.",
            evidence=[Evidence(kind="topdown_map", summary="Current Habitat navmesh localization.", ref=report.topdown_ref)],
            outputs=report.model_dump(mode="json"),
            compressed_history=(
                f"Habitat localization: x={report.pose.x:.2f}, y={report.pose.y:.2f}, "
                f"yaw={report.pose.yaw:.2f} rad."
            ),
        )


class AnchorGenerationSkill:
    name = "AnchorGenerationSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Convert one currently visible semantic anchor into a standard world-space navigation point.",
        input_schema=AnchorGenerationInput.model_json_schema(),
        output_schema=GeneratedNavigationPoint.model_json_schema(),
        required_modalities=["rgb", "depth", "pose"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
        default_timeout_s=60.0,
    )

    def __init__(self, provider_factory: Callable[[], LLMProvider], embodiment: HierarchicalEmbodiment) -> None:
        self._grounding = AnchorGroundingSkill(provider_factory)
        self._projection = PixelToPointSkill(embodiment)

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = AnchorGenerationInput.model_validate(request.inputs)
        if inputs.source_observation_id != observation.trace.observation_id:
            raise ValueError("Anchor source does not match the current observation")
        grounding_request = request.model_copy(
            update={
                "skill_name": AnchorGroundingSkill.name,
                "command": AnchorGroundingSkill.name,
                "inputs": {
                    "anchor": inputs.anchor,
                    "view": inputs.view,
                    "image_width": inputs.image_width,
                    "image_height": inputs.image_height,
                    "allowed_views": inputs.allowed_views,
                    "direction_hint": inputs.direction_hint,
                },
            }
        )
        grounded = await self._grounding.run(grounding_request, observation, context)
        pixel = AnchorPixel.model_validate(grounded.outputs)
        projected = None
        spatial = None
        if pixel.anchor_visible:
            projection_request = request.model_copy(
                update={
                    "skill_name": PixelToPointSkill.name,
                    "command": PixelToPointSkill.name,
                    "inputs": {
                        "source_observation_id": inputs.source_observation_id,
                        "view": pixel.view,
                        "x_px": pixel.x_px,
                        "y_px": pixel.y_px,
                        "image_width": inputs.image_width,
                        "image_height": inputs.image_height,
                        "max_depth_m": inputs.max_depth_m,
                    },
                }
            )
            projected = await self._projection.run(projection_request, observation, context)
            spatial = SpatialAnchorPoint.model_validate(projected.outputs)
        point = GeneratedNavigationPoint(
            target_id=f"anchor-{request.trace.skill_run_id}",
            source_observation_id=inputs.source_observation_id,
            anchor=inputs.anchor,
            anchor_role=inputs.anchor_role,
            view=pixel.view,
            grounding_pixel=([pixel.x_px, pixel.y_px] if pixel.anchor_visible else None),
            target_pixel=(
                [pixel.x_px, pixel.y_px]
                if pixel.anchor_visible
                and inputs.anchor_role == "target"
                else None
            ),
            bbox_px=pixel.bbox_px,
            world_point_xyz_m=(spatial.navigation_candidate_xyz_m if spatial is not None else None),
            anchor_visible=pixel.anchor_visible,
            confidence=pixel.confidence,
            evidence_summary=pixel.evidence_summary,
            annotated_ref=pixel.annotated_ref,
            depth_m=(spatial.depth_m if spatial is not None else None),
            raw_depth_m=(spatial.raw_depth_m if spatial is not None else None),
            depth_limited=(spatial.depth_limited if spatial is not None else False),
            max_depth_m=inputs.max_depth_m,
        )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=f"Generated a navigation point for visible anchor {inputs.anchor!r}.",
            evidence=[*grounded.evidence, *(projected.evidence if projected is not None else [])],
            outputs=point.model_dump(mode="json"),
            compressed_history=f"Visible anchor {inputs.anchor!r} became one standard navigation point.",
        )


class PointNavigationSkill:
    name = "PointNavigationSkill"
    descriptor = SkillDescriptor(
        name=name,
        description="Plan and execute Habitat navmesh navigation to any standard world-space point.",
        input_schema=PointNavigationInput.model_json_schema(),
        output_schema=PointNavigationReport.model_json_schema(),
        required_modalities=["pose", "navmesh"],
        supported_embodiments=["habitat-gs"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["moves_robot"],
        default_timeout_s=120.0,
    )

    def __init__(self, embodiment: HierarchicalEmbodiment) -> None:
        self._embodiment = embodiment
        self._planning = PathPlanningSkill(embodiment)
        self._execution = PathExecutionSkill(embodiment)

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        inputs = PointNavigationInput.model_validate(request.inputs)
        planning_request = request.model_copy(
            update={
                "skill_name": PathPlanningSkill.name,
                "command": PathPlanningSkill.name,
                "inputs": {"world_point_xyz_m": inputs.world_point_xyz_m},
            }
        )
        planned_result = await self._planning.run(planning_request, observation, context)
        planned = PlannedPath.model_validate(planned_result.outputs)
        pixel_goal = await self._build_pixel_goal(inputs, observation, planned)
        await context.emit_progress(
            request,
            "Prepared ABot dual-pixel guidance before Fast execution.",
            phase="pixel_guidance_ready",
            pixel_goal=(
                pixel_goal.model_dump(mode="json") if pixel_goal is not None else None
            ),
        )
        execution_request = request.model_copy(
            update={
                "skill_name": PathExecutionSkill.name,
                "command": PathExecutionSkill.name,
                "inputs": {
                    "world_path_xyz_m": planned.world_path_xyz_m,
                    "max_control_steps": inputs.max_control_steps,
                    "goal_tolerance_m": inputs.goal_tolerance_m,
                    "control_period_s": inputs.control_period_s,
                },
            }
        )
        executed_result = await self._execution.run(execution_request, observation, context)
        executed = PathExecutionReport.model_validate(executed_result.outputs)
        final_pose = dict(executed.pose_after)
        orientation_steps = 0
        orientation_error: float | None = None
        if executed.reached and inputs.target_yaw is not None and "yaw" in final_pose:
            for _ in range(24):
                delta = self._angle_delta(inputs.target_yaw, float(final_pose["yaw"]))
                orientation_error = abs(delta)
                if orientation_error <= 0.14:
                    break
                control = self._embodiment.step("turn_left" if delta > 0.0 else "turn_right")
                final_pose = dict(control.get("pose_after") or final_pose)
                orientation_steps += 1
            orientation_error = abs(
                self._angle_delta(inputs.target_yaw, float(final_pose.get("yaw", 0.0)))
            )
        report = PointNavigationReport(
            **executed.model_dump(mode="python", exclude={"pose_after"}),
            pose_after=final_pose,
            slow_step_id=inputs.slow_step_id,
            target_id=inputs.target_id,
            requested_goal_xyz_m=planned.requested_goal_xyz_m,
            snapped_goal_xyz_m=planned.snapped_goal_xyz_m,
            geodesic_distance_m=planned.geodesic_distance_m,
            waypoint_count=planned.waypoint_count,
            target_yaw=inputs.target_yaw,
            final_yaw=float(final_pose["yaw"]) if "yaw" in final_pose else None,
            orientation_error_rad=orientation_error,
            orientation_steps=orientation_steps,
            pixel_goal=pixel_goal,
        )
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=executed_result.status,
            summary=f"Point navigation to {inputs.target_id}: reached={report.reached}.",
            evidence=executed_result.evidence,
            outputs=report.model_dump(mode="json"),
            compressed_history=(
                f"Point {inputs.target_id}: reached={report.reached}, actions={report.control_steps}, "
                f"distance={report.distance_m:.2f}m."
            ),
        )

    @staticmethod
    def _angle_delta(target: float, current: float) -> float:
        return (target - current + math.pi) % (2.0 * math.pi) - math.pi

    async def _build_pixel_goal(
        self,
        inputs: PointNavigationInput,
        observation: Observation,
        planned: PlannedPath,
    ) -> DualPixelGoal | None:
        if inputs.source_observation_id is None:
            return None
        if inputs.source_observation_id != observation.trace.observation_id:
            raise ValueError("pixel-goal source does not match the planning observation")

        affordance_view = None
        affordance_pixel = None
        affordance_world_point = None
        projector = getattr(self._embodiment, "world_to_pixel", None)
        if projector is not None:
            total_distance = self._path_length(planned.world_path_xyz_m)
            maximum = min(inputs.affordance_lookahead_m, total_distance)
            lookaheads = [maximum]
            lookaheads.extend(
                distance
                for distance in (2.5, 2.0, 1.5, 1.0, 0.5)
                if 0.0 < distance < maximum - 1e-6
            )
            fallback = None
            camera = observation.metadata.get("camera") or {}
            resolution = camera.get("output_resolution") or [640, 480]
            for distance in lookaheads:
                world_point = self._path_point_at_distance(
                    planned.world_path_xyz_m, distance
                )
                projection = await projector(
                    world_point=world_point,
                    depth_refs=observation.depth_refs,
                    image_width=int(resolution[0]),
                    image_height=int(resolution[1]),
                )
                if projection is None:
                    continue
                fallback = fallback or (projection, world_point)
                if bool(projection.get("visible", True)):
                    fallback = (projection, world_point)
                    break
            if fallback is not None:
                projection, affordance_world_point = fallback
                affordance_view = str(projection["view"])
                affordance_pixel = [int(value) for value in projection["pixel"]]

        return DualPixelGoal(
            source_observation_id=inputs.source_observation_id,
            affordance_pixel=TriViewPixelGoal.from_view(
                affordance_view, affordance_pixel
            ),
            target_pixel=TriViewPixelGoal.from_view(
                inputs.target_view, inputs.target_pixel
            ),
            affordance_world_point_xyz_m=affordance_world_point,
        )

    @staticmethod
    def _path_length(path: list[list[float]]) -> float:
        return sum(
            math.dist(previous, current)
            for previous, current in zip(path, path[1:])
        )

    @staticmethod
    def _path_point_at_distance(
        path: list[list[float]], distance_m: float
    ) -> list[float]:
        if not path:
            raise ValueError("path must contain at least one point")
        remaining = max(0.0, distance_m)
        previous = path[0]
        for current in path[1:]:
            segment = math.dist(previous, current)
            if segment > 1e-9 and remaining <= segment:
                ratio = remaining / segment
                return [
                    float(previous[index])
                    + (float(current[index]) - float(previous[index])) * ratio
                    for index in range(3)
                ]
            remaining -= segment
            previous = current
        return [float(value) for value in path[-1]]
