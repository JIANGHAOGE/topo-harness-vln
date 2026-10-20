from __future__ import annotations

import numpy as np

from agentos.mapping import ROS2OccupancyGridMapProvider
from agentos.topology import (
    BackboneNode,
    BackboneTopology,
    BackboneTopologyCache,
    NavigabilityGrid,
    SparseBackboneExtractor,
    normalize_ros2_occupancy_grid,
)


def _ros_message(free):
    return {
        "header": {"frame_id": "map"},
        "info": {
            "width": int(free.shape[1]),
            "height": int(free.shape[0]),
            "resolution": 0.2,
            "origin": {
                "position": {"x": -1.0, "y": -2.0},
                "orientation": {"z": 0.0, "w": 1.0},
            },
        },
        "data": np.where(free, 0, 100).reshape(-1).tolist(),
    }


def test_habitat_and_ros2_grids_share_the_same_backbone_extractor():
    free = np.zeros((30, 40), dtype=bool)
    free[4:26, 5:35] = True
    habitat_grid = NavigabilityGrid(
        map_id="first-floor",
        geometry_revision="navmesh-v1",
        source="habitat_navmesh",
        frame="map",
        resolution_m=0.2,
        origin_xy_m=(-1.0, -2.0),
        free_mask=free,
        navigation_plane="xy",
    )
    ros_grid = normalize_ros2_occupancy_grid(
        _ros_message(free),
        map_id="first-floor",
        geometry_revision="occupancy-v1",
    )
    extractor = SparseBackboneExtractor(
        minimum_clearance_m=0.4,
        minimum_node_spacing_m=1.0,
        maximum_nodes=12,
    )

    habitat = extractor.extract(habitat_grid)
    ros = extractor.extract(ros_grid)

    assert habitat.nodes
    assert [(node.node_type, node.row, node.column) for node in habitat.nodes] == [
        (node.node_type, node.row, node.column) for node in ros.nodes
    ]
    assert [(round(node.x, 3), round(node.y, 3)) for node in habitat.nodes] == [
        (round(node.x, 3), round(node.y, 3)) for node in ros.nodes
    ]
    assert [(edge.source, edge.target) for edge in habitat.edges] == [
        (edge.source, edge.target) for edge in ros.edges
    ]


def test_backbone_edges_follow_skeleton_adjacency_instead_of_crossing_walls():
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    skeleton[1:9, 1] = 1
    skeleton[8, 1:4] = 1
    skeleton[1:9, 3] = 1
    free_components = np.where(skeleton, 1, 0).astype(np.int32)
    nodes = [
        BackboneNode(
            node_id="backbone-000",
            node_type="junction",
            x=1.0,
            y=1.0,
            row=1,
            column=1,
            clearance_m=1.0,
            navigation_point_xyz_m=[1.0, 1.0, 0.0],
        ),
        BackboneNode(
            node_id="backbone-001",
            node_type="junction",
            x=1.0,
            y=8.0,
            row=8,
            column=1,
            clearance_m=1.0,
            navigation_point_xyz_m=[1.0, 8.0, 0.0],
        ),
        BackboneNode(
            node_id="backbone-002",
            node_type="junction",
            x=3.0,
            y=1.0,
            row=1,
            column=3,
            clearance_m=1.0,
            navigation_point_xyz_m=[3.0, 1.0, 0.0],
        ),
    ]

    edges = SparseBackboneExtractor._skeleton_adjacency_edges(
        nodes,
        skeleton,
        free_components,
        NavigabilityGrid(
            map_id="u-map",
            geometry_revision="u-v1",
            source="ros2_occupancy_grid",
            frame="map",
            resolution_m=1.0,
            origin_xy_m=(0.0, 0.0),
            free_mask=free_components > 0,
        ),
    )

    assert [(edge.source, edge.target) for edge in edges] == [
        ("backbone-000", "backbone-001"),
        ("backbone-001", "backbone-002"),
    ]
    assert [round(edge.distance_m, 3) for edge in edges] == [7.0, 8.414]
    assert edges[0].path_world_xy_m == [(1.5, 1.5), (1.5, 8.5)]
    assert ("backbone-000", "backbone-002") not in {
        (edge.source, edge.target) for edge in edges
    }


def test_ros2_provider_supplies_grid_without_an_embodiment_dependency(tmp_path):
    free = np.zeros((12, 16), dtype=bool)
    free[2:10, 3:13] = True
    calls = []
    provider = ROS2OccupancyGridMapProvider(
        message_loader=lambda: calls.append("map") or _ros_message(free),
        map_id="robot-building-map",
        cache_dir=tmp_path / "topology-cache",
    )

    grid = provider.load_grid()

    assert calls == ["map"]
    assert grid.source == "ros2_occupancy_grid"
    assert grid.map_id == "robot-building-map"
    assert grid.frame == "map"
    assert np.array_equal(grid.free_mask, free)


def test_backbone_cache_is_keyed_by_map_geometry_revision(tmp_path):
    class CountingExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, grid):
            self.calls += 1
            return BackboneTopology(
                map_id=grid.map_id,
                geometry_revision=grid.geometry_revision,
                source=grid.source,
                frame=grid.frame,
                cache_key=grid.cache_key,
                nodes=[
                    BackboneNode(
                        node_id="backbone-000",
                        node_type="open_space_center",
                        x=0.5,
                        y=0.5,
                        row=0,
                        column=0,
                        clearance_m=1.0,
                        navigation_point_xyz_m=[0.5, 0.5, 0.0],
                    )
                ],
            )

    extractor = CountingExtractor()
    cache = BackboneTopologyCache(tmp_path, extractor=extractor)
    grid = NavigabilityGrid(
        map_id="robot-map",
        geometry_revision="revision-7",
        source="ros2_occupancy_grid",
        frame="map",
        resolution_m=1.0,
        origin_xy_m=(0.0, 0.0),
        free_mask=np.ones((2, 2), dtype=bool),
    )

    first, first_hit = cache.get_or_build(grid)
    second, second_hit = cache.get_or_build(grid)

    assert first_hit is False
    assert second_hit is True
    assert first == second
    assert extractor.calls == 1
    assert len(list(tmp_path.glob("robot-map-*.json"))) == 1
