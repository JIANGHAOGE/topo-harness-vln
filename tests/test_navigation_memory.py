from __future__ import annotations

from agentos.memory.navigation import AgentNavigationMemory
from agentos.memory.store import SQLiteGraphMemory
from agentos.schemas.hierarchical import SemanticMapUpdate
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.semantic_map import EditableSemanticMap, NodeSemantics
from agentos.topology import BackboneEdge, BackboneNode, BackboneTopology


def _observation(make_trace, index: int) -> Observation:
    return Observation(
        trace=make_trace(observation_id=f"obs-{index}"),
        pose=Pose2D(
            x=float(index),
            y=0.0,
            yaw=0.0,
            source="robot-localization",
            frame="odom",
        ),
        metadata={
            "metric_position_xyz": [float(index), 0.0, 0.0],
            "map_id": "robot-map-1",
            "pose_confidence": 0.9,
        },
    )


def _fixed_backbone() -> BackboneTopology:
    return BackboneTopology(
        map_id="robot-map-1",
        geometry_revision="map-revision-1",
        source="ros2_occupancy_grid",
        frame="odom",
        cache_key="cache-key-1",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=2.0,
                y=0.0,
                row=1,
                column=2,
                clearance_m=0.8,
                navigation_point_xyz_m=[2.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=5.0,
                y=0.0,
                row=1,
                column=5,
                clearance_m=1.4,
                navigation_point_xyz_m=[5.0, 0.0, 0.0],
            ),
        ],
        edges=[BackboneEdge(source="backbone-000", target="backbone-001", distance_m=3.0)],
    )


def test_topo_memory_exposes_abot_n0_four_layers(tmp_path):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    memory.record_backbone_topology(_fixed_backbone())
    memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="robot-map-1",
            geometry_revision="map-revision-1",
            nodes={
                "backbone-001": NodeSemantics(
                    place_type="kitchen",
                    function="food preparation and appliance storage",
                    objects=["white refrigerator"],
                    landmarks=["wood shelving"],
                    task_affordances=["find refrigerator"],
                )
            },
        )
    )

    topo = memory.topo_memory_view()

    assert [block.topology_node_id for block in topo.block_layer] == [
        "backbone-000",
        "backbone-001",
    ]
    assert topo.road_layer[0].source == "backbone-000"
    assert topo.road_layer[0].target == "backbone-001"
    assert topo.road_layer[0].distance_m == 3.0
    assert topo.function_layer[0].function.startswith("food preparation")
    assert topo.object_poi_layer[0].objects == ["white refrigerator"]
    store.close()


def test_known_node_visit_uses_the_navigation_region_tolerance(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    memory.record_backbone_topology(_fixed_backbone())
    target = memory.get_topology_node("backbone-001")
    assert target is not None
    observation = _observation(make_trace, 4)

    outside_default = memory.mark_known_visit(observation, target)
    inside_region = memory.mark_known_visit(
        observation,
        target,
        reach_tolerance_m=1.4,
    )

    assert outside_default.current_place is None
    assert inside_region.current_place is not None
    assert memory.topo_memory_view().current_topology_node_id == "backbone-001"
    store.close()


def test_topology_goal_search_is_separate_from_task_scoped_visits(tmp_path):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    topology = _fixed_backbone()
    memory.record_backbone_topology(topology)

    first = memory.record_topology_search(
        topology_node_id="backbone-001",
        goal="white refrigerator",
        status="no_target_observed",
        observation_id="obs-search-1",
        evidence="No refrigerator was visible after local inspection.",
    )
    second = memory.record_topology_search(
        topology_node_id="backbone-001",
        goal="white refrigerator",
        status="no_target_observed",
        observation_id="obs-search-2",
        evidence="A second view also showed no refrigerator.",
    )
    confirmed = memory.record_topology_search(
        topology_node_id="backbone-001",
        planned_topology_node_id="backbone-001",
        observed_topology_node_id="backbone-000",
        goal="white refrigerator",
        status="target_confirmed",
        observation_id="obs-search-2",
        evidence="Independent Skill Verifier confirmed the target.",
        increment_attempt=False,
    )

    block = next(
        item
        for item in memory.topo_memory_view().block_layer
        if item.topology_node_id == "backbone-001"
    )
    graph_node = next(
        item
        for item in memory.route_graph()["nodes"]
        if item["topology_node_id"] == "backbone-001"
    )
    assert first.attempt_count == 1
    assert second.attempt_count == 2
    assert confirmed.attempt_count == 2
    assert confirmed.planned_topology_node_id == "backbone-001"
    assert confirmed.observed_topology_node_id == "backbone-000"
    assert block.visited is True
    assert block.visit_count == 1
    assert block.goal_search[0].status == "target_confirmed"
    assert graph_node["goal_search"][0]["attempt_count"] == 2
    assert memory.topo_memory_view().current_topology_node_id == "backbone-000"

    memory.start_task(task_id="task-2", episode_id="episode-2")
    memory.record_backbone_topology(topology)
    fresh_block = next(
        item
        for item in memory.topo_memory_view().block_layer
        if item.topology_node_id == "backbone-001"
    )
    assert fresh_block.visited is False
    assert fresh_block.goal_search == []
    store.close()


def test_navigation_memory_view_remains_bounded_as_places_grow(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    semantics = SemanticMapUpdate(
        scene_summary="corridor with doors and cabinets",
        left=["cabinet"],
        front=["doorway"],
        right=["wall"],
    )

    for index in range(40):
        memory.record_place(_observation(make_trace, index), semantics)

    view = memory.view("find the doorway to the bedroom")
    payload = view.model_dump_json()
    assert memory.place_count == 40
    assert view.total_places == 40
    assert len(view.relevant_places) == 5
    assert len(payload) < 7000
    assert "habitat" not in payload.casefold()
    store.close()

    reopened = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    restored = AgentNavigationMemory(
        reopened,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    assert restored.place_count == 40
    assert len(restored.view().relevant_places) == 5
    reopened.close()


def test_trajectory_keyframes_become_pose_aware_known_locations(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    memory.record_place(
        _observation(make_trace, 0),
        SemanticMapUpdate(
            scene_summary="starting corridor",
            left=[],
            front=["hallway"],
            right=[],
        ),
    )
    frames = [
        {
            "frame_index": 0,
            "observation_id": "obs-frame-0",
            "timestamp": "2026-08-17T00:00:00+00:00",
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0, "source": "odom", "frame": "odom"},
            "map_id": "robot-map-1",
            "pose_confidence": 0.9,
            "navigation_point_xyz_m": [0.0, 0.0, 0.0],
            "rgb_refs": {"front": "/frames/0.jpg"},
        },
        {
            "frame_index": 8,
            "observation_id": "obs-frame-8",
            "timestamp": "2026-08-17T00:00:08+00:00",
            "pose": {"x": 1.2, "y": -0.4, "yaw": 1.57, "source": "odom", "frame": "odom"},
            "map_id": "robot-map-1",
            "pose_confidence": 0.9,
            "navigation_point_xyz_m": [1.2, 0.0, -0.4],
            "rgb_refs": {"front": "/frames/8.jpg"},
        },
    ]

    registered = memory.record_trajectory_keyframes(frames, source_target="bedroom doorway")

    assert registered[0]["location_id"] == "location-001"
    assert registered[1]["location_id"] == "location-002"
    location = memory.get_place("location-002")
    assert location is not None
    assert location.source == "trajectory_keyframe"
    assert location.trajectory_frame_index == 8
    assert location.pose.yaw == 1.57
    assert location.navigation_point_xyz_m == [1.2, 0.0, -0.4]
    assert location.front_image_ref == "/frames/8.jpg"
    assert memory.view().current_place.source == "observation"
    assert memory.route_node_count == 0
    visuals = memory.visual_memories()
    assert len(visuals) == 2
    assert len({visual.subtask_id for visual in visuals}) == 1
    assert [visual.image_ref for visual in memory.recall_visuals(scope="latest_subtask")] == [
        "/frames/0.jpg",
        "/frames/8.jpg",
    ]

    memory.start_task(task_id="task-2", episode_id="episode-2")
    assert memory.place_count == 0
    store.close()


def test_high_confidence_semantics_annotate_fixed_backbone_without_new_route_nodes(
    tmp_path, make_trace
):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
        place_merge_radius_m=0.75,
    )
    memory.record_place(
        _observation(make_trace, 0),
        SemanticMapUpdate(scene_summary="wooden door at start", left=[], front=["wooden door"], right=[]),
    )
    waypoints = [
        {
            "frame_index": index,
            "observation_id": f"route-{index}",
            "timestamp": f"2026-08-17T00:00:{index:02d}+00:00",
            "pose": {
                "x": distance,
                "y": 0.0,
                "yaw": 0.0,
                "source": "odom",
                "frame": "odom",
            },
            "map_id": "robot-map-1",
            "pose_confidence": 0.9,
            "navigation_point_xyz_m": [distance, 0.0, 0.0],
        }
        for index, distance in enumerate([0.0, 0.3, 0.7, 0.9, 1.4, 1.8])
    ]

    memory.record_route_waypoints(waypoints, source_target="bedroom corridor")
    frames = [
        {**waypoints[index], "rgb_refs": {"front": f"/frames/{index}.jpg"}}
        for index in (0, 2, 5)
    ]
    registered = memory.record_trajectory_keyframes(frames, source_target="bedroom corridor")

    assert memory.place_count == 3
    assert [item["location_id"] for item in registered] == [
        "location-001",
        "location-002",
        "location-003",
    ]
    assert memory.get_place("location-003").source == "trajectory_keyframe"
    assert memory.get_place("location-003").front_image_ref == "/frames/5.jpg"
    assert memory.route_graph()["nodes"] == []

    anchor_observation = _observation(make_trace, 0)
    memory.mark_global_anchor(
        anchor_observation,
        SemanticMapUpdate(scene_summary="start", front=["wooden door"]),
        "find the bedroom",
    )
    backbone = memory.record_backbone_topology(_fixed_backbone())
    route_count_before = memory.route_node_count
    junction_observation = _observation(make_trace, 2)
    promoted = memory.promote_semantic_landmark(
        junction_observation,
        SemanticMapUpdate(
            scene_summary="junction with left and front corridors",
            left=["left corridor"],
            front=["front corridor"],
            route_node_type="junction",
            traversable_directions=["left", "front"],
        ),
        label="bedroom corridor junction",
        confidence=0.95,
        task_relevance=0.96,
    )
    open_observation = _observation(make_trace, 5)
    rejected = memory.promote_semantic_landmark(
        open_observation,
        SemanticMapUpdate(
            scene_summary="center of a broad lobby",
            left=["open floor"],
            front=["open floor"],
            right=["doorway"],
            route_node_type="open_space_center",
            traversable_directions=["left", "front", "right"],
        ),
        label="broad lobby",
        confidence=0.89,
        task_relevance=0.98,
    )

    graph = memory.route_graph()
    assert promoted is not None
    assert rejected is None
    assert memory.route_node_count == route_count_before == 2
    assert len(graph["nodes"]) == 2
    assert graph["edges"] == [
        {
            "source": backbone["node_locations"]["backbone-000"],
            "target": backbone["node_locations"]["backbone-001"],
        },
    ]
    assert not any(node["is_global_anchor"] for node in graph["nodes"])
    annotated = memory.get_place(backbone["node_locations"]["backbone-000"])
    assert annotated is not None
    assert annotated.source == "map_backbone"
    assert annotated.route_node_type == "junction"
    assert annotated.semantic_confidence == 0.95
    assert annotated.task_relevance == 0.96
    assert annotated.traversable_directions == ["left", "front"]
    assert annotated.semantic_observations[-1].scene_summary == "junction with left and front corridors"
    assert next(node for node in graph["nodes"] if node["location_id"] == annotated.place_id)[
        "is_current"
    ] is True
    assert memory.previous_place(junction_observation).place_id == "location-001"
    assert memory.view(current_observation=junction_observation).backtrack_location.place_id == "location-001"
    assert memory.view().route_node_count == 2
    store.close()


def test_cached_backbone_excludes_the_task_start_from_the_route_graph(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    start = _observation(make_trace, 0)
    start_semantics = SemanticMapUpdate(scene_summary="task start", front=["corridor"])
    memory.mark_global_anchor(start, start_semantics, "find the bedroom")
    topology = BackboneTopology(
        map_id="robot-map-1",
        geometry_revision="map-revision-1",
        source="ros2_occupancy_grid",
        frame="odom",
        cache_key="cache-key-1",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="junction",
                x=2.0,
                y=0.0,
                row=1,
                column=2,
                clearance_m=0.8,
                navigation_point_xyz_m=[2.0, 0.0, 0.0],
            ),
            BackboneNode(
                node_id="backbone-001",
                node_type="open_space_center",
                x=5.0,
                y=0.0,
                row=1,
                column=5,
                clearance_m=1.4,
                navigation_point_xyz_m=[5.0, 0.0, 0.0],
            ),
        ],
        edges=[
            BackboneEdge(
                source="backbone-000",
                target="backbone-001",
                distance_m=3.0,
                path_world_xy_m=[(2.0, 0.0), (3.5, 0.5), (5.0, 0.0)],
            )
        ],
    )

    update = memory.record_backbone_topology(topology)
    graph = memory.route_graph()

    assert update["node_locations"] == {
        "backbone-000": "location-002",
        "backbone-001": "location-003",
    }
    assert [node["source"] for node in graph["nodes"]] == ["map_backbone", "map_backbone"]
    assert graph["edges"] == [
        {
            "source": "location-002",
            "target": "location-003",
            "distance_m": 3.0,
            "path_world_xy_m": [[2.0, 0.0], [3.5, 0.5], [5.0, 0.0]],
        },
    ]
    assert not any(node["is_global_anchor"] for node in graph["nodes"])
    store.close()


def test_latest_unrecorded_pose_exposes_nearest_route_node_for_backtracking(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    anchor = _observation(make_trace, 0)
    semantics = SemanticMapUpdate(scene_summary="task start", front=["corridor"])
    memory.record_place(anchor, semantics)
    memory.mark_global_anchor(anchor, semantics, "find the bedroom")

    latest = _observation(make_trace, 3)
    view = memory.view("find the bedroom", current_observation=latest)

    assert view.backtrack_location is not None
    assert view.backtrack_location.place_id == "location-001"
    assert view.backtrack_location.route_node_type == "none"
    store.close()


def test_ordinary_nearby_observation_does_not_move_semantic_route_node(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    backbone = memory.record_backbone_topology(_fixed_backbone())
    node_id = backbone["node_locations"]["backbone-000"]
    promoted = memory.promote_semantic_landmark(
        _observation(make_trace, 2),
        SemanticMapUpdate(
            scene_summary="central junction",
            left=["left corridor"],
            front=["front corridor"],
            right=["right corridor"],
            route_node_type="junction",
            traversable_directions=["left", "front", "right"],
        ),
        label="task-relevant central junction",
        confidence=0.95,
        task_relevance=0.95,
    )
    assert promoted is not None
    nearby = _observation(make_trace, 2).model_copy(
        update={
            "trace": make_trace(observation_id="obs-nearby"),
            "pose": Pose2D(
                x=2.4,
                y=0.0,
                yaw=0.0,
                source="robot-localization",
                frame="odom",
            ),
            "metadata": {
                "metric_position_xyz": [2.4, 0.0, 0.0],
                "map_id": "robot-map-1",
                "pose_confidence": 0.9,
            },
        }
    )
    memory.record_place(
        nearby,
        SemanticMapUpdate(
            scene_summary="ordinary corridor transit",
            left=["wall"],
            front=["corridor"],
            right=["wall"],
            traversable_directions=["front"],
        ),
    )

    node = memory.get_place(node_id)
    assert node is not None
    assert node.pose.x == 2.0
    assert node.navigation_point_xyz_m == [2.0, 0.0, 0.0]
    assert node.scene_summary == "central junction"
    assert node.route_node_type == "junction"
    assert node.traversable_directions == ["left", "front", "right"]
    store.close()


def test_initial_instruction_and_views_become_task_scoped_global_anchor(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    observation = _observation(make_trace, 0).model_copy(
        update={"rgb_refs": {"left": "/first/left.jpg", "front": "/first/front.jpg", "right": "/first/right.jpg"}}
    )
    semantics = SemanticMapUpdate(
        scene_summary="entry hall with a wooden door",
        left=["cabinet"],
        front=["wooden door"],
        right=["corridor"],
    )
    memory.record_place(observation, semantics)

    anchor = memory.mark_global_anchor(
        observation,
        semantics,
        "go through the wooden door and find the bedroom",
    )

    assert anchor.place_id == "location-001"
    assert anchor.semantic_labels == ["cabinet", "wooden door", "corridor"]
    assert anchor.image_refs["front"] == "/first/front.jpg"
    assert memory.view().global_anchor.anchor_instruction == "go through the wooden door and find the bedroom"
    recalled = memory.recall_visuals(query="wooden door")
    assert {visual.view for visual in recalled} == {"left", "front", "right"}
    assert all(visual.location_id == "location-001" for visual in recalled)
    memory.start_task(task_id="task-2", episode_id="episode-2")
    assert memory.view().global_anchor is None
    assert memory.visual_memories() == []
    store.close()


def test_task_visual_overview_spans_all_subtasks_without_growing_prompt(tmp_path, make_trace):
    store = SQLiteGraphMemory(tmp_path / "memory.sqlite3")
    memory = AgentNavigationMemory(
        store,
        environment_id="robot:test",
        task_id="task-1",
        episode_id="episode-1",
    )
    memory.record_place(
        _observation(make_trace, 0),
        SemanticMapUpdate(scene_summary="start", left=[], front=["hall"], right=[]),
    )
    for subtask in range(3):
        frames = [
            {
                "frame_index": index,
                "observation_id": f"subtask-{subtask}-frame-{index}",
                "timestamp": f"2026-08-17T00:0{subtask}:{index:02d}+00:00",
                "pose": {
                    "x": float(subtask * 4 + index),
                    "y": 0.0,
                    "yaw": 0.0,
                    "source": "odom",
                    "frame": "odom",
                },
                "map_id": "robot-map-1",
                "navigation_point_xyz_m": [float(subtask * 4 + index), 0.0, 0.0],
                "rgb_refs": {"front": f"/frames/{subtask}-{index}.jpg"},
            }
            for index in range(4)
        ]
        memory.record_trajectory_keyframes(frames, source_target=f"route branch {subtask}")

    overview = memory.recall_visuals(scope="task_overview", limit=4)
    latest = memory.recall_visuals(scope="latest_subtask", limit=4)

    assert len(memory.visual_memories()) == 12
    assert len(overview) == 4
    assert overview[0].image_ref == "/frames/0-0.jpg"
    assert overview[-1].image_ref == "/frames/2-3.jpg"
    assert [visual.image_ref for visual in latest] == [
        "/frames/2-0.jpg",
        "/frames/2-1.jpg",
        "/frames/2-2.jpg",
        "/frames/2-3.jpg",
    ]
    view = memory.view()
    assert view.visual_archive_count == 12
    assert view.latest_subtask_keyframe_count == 4
    store.close()
