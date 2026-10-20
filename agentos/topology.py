from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from agentos.schemas.base import SchemaModel


TopologySource = Literal["habitat_navmesh", "ros2_occupancy_grid"]
BackboneNodeType = Literal["junction", "open_space_center"]


@dataclass(frozen=True)
class NavigabilityGrid:
    """Environment-neutral 2D free-space grid used by the topology extractor."""

    map_id: str
    geometry_revision: str
    source: TopologySource
    frame: str
    resolution_m: float
    origin_xy_m: tuple[float, float]
    free_mask: Any = field(repr=False)
    navigation_plane: Literal["xy", "xz"] = "xy"
    navigation_height_m: float = 0.0
    origin_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        import numpy as np

        mask = np.asarray(self.free_mask, dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("free_mask must be a non-empty 2D array")
        if not self.map_id or not self.geometry_revision or not self.frame:
            raise ValueError("map_id, geometry_revision, and frame are required")
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")
        object.__setattr__(self, "free_mask", mask)

    @property
    def cache_key(self) -> str:
        payload = f"{self.source}:{self.map_id}:{self.geometry_revision}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def pixel_to_world(self, row: int, column: int) -> tuple[float, float]:
        local_x = (float(column) + 0.5) * self.resolution_m
        local_y = (float(row) + 0.5) * self.resolution_m
        cosine = math.cos(self.origin_yaw_rad)
        sine = math.sin(self.origin_yaw_rad)
        return (
            self.origin_xy_m[0] + cosine * local_x - sine * local_y,
            self.origin_xy_m[1] + sine * local_x + cosine * local_y,
        )

    def navigation_point(self, x: float, y: float) -> list[float]:
        if self.navigation_plane == "xz":
            return [x, self.navigation_height_m, y]
        return [x, y, self.navigation_height_m]


class BackboneNode(SchemaModel):
    node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    node_type: BackboneNodeType
    x: float
    y: float
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    clearance_m: float = Field(ge=0.0)
    navigation_point_xyz_m: list[float] = Field(min_length=3, max_length=3)


class BackboneEdge(SchemaModel):
    source: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    target: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    distance_m: float = Field(ge=0.0)
    path_world_xy_m: list[tuple[float, float]] = Field(default_factory=list)


class BackboneTopology(SchemaModel):
    map_id: str = Field(min_length=1)
    geometry_revision: str = Field(min_length=1)
    source: TopologySource
    frame: str = Field(min_length=1)
    cache_key: str = Field(min_length=1)
    nodes: list[BackboneNode] = Field(default_factory=list)
    edges: list[BackboneEdge] = Field(default_factory=list)


def normalize_ros2_occupancy_grid(
    message: dict[str, Any],
    *,
    map_id: str,
    geometry_revision: str | None = None,
    free_threshold: int = 20,
) -> NavigabilityGrid:
    """Convert a ROS2 OccupancyGrid-shaped dictionary to the common grid contract."""
    import numpy as np

    info = message.get("info") or {}
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    resolution = float(info.get("resolution") or 0.0)
    data = np.asarray(message.get("data") or [], dtype=np.int16)
    if width <= 0 or height <= 0 or data.size != width * height:
        raise ValueError("invalid ROS2 OccupancyGrid dimensions or data length")
    origin = (info.get("origin") or {}).get("position") or {}
    orientation = (info.get("origin") or {}).get("orientation") or {}
    quaternion_z = float(orientation.get("z") or 0.0)
    quaternion_w = float(orientation.get("w") if orientation.get("w") is not None else 1.0)
    origin_yaw = 2.0 * math.atan2(quaternion_z, quaternion_w)
    free_mask = data.reshape((height, width))
    free_mask = (free_mask >= 0) & (free_mask <= free_threshold)
    if geometry_revision is None:
        digest = hashlib.sha256()
        digest.update(data.tobytes())
        digest.update(
            (
                f"{width}:{height}:{resolution}:"
                f"{float(origin.get('x') or 0.0)}:{float(origin.get('y') or 0.0)}:{origin_yaw}"
            ).encode("ascii")
        )
        geometry_revision = digest.hexdigest()[:20]
    return NavigabilityGrid(
        map_id=map_id,
        geometry_revision=geometry_revision,
        source="ros2_occupancy_grid",
        frame=str((message.get("header") or {}).get("frame_id") or "map"),
        resolution_m=resolution,
        origin_xy_m=(float(origin.get("x") or 0.0), float(origin.get("y") or 0.0)),
        free_mask=free_mask,
        navigation_plane="xy",
        origin_yaw_rad=origin_yaw,
    )


class SparseBackboneExtractor:
    """Extract stable junctions and broad-space centers from a normalized grid."""

    algorithm_version = "sparse-backbone-v3"

    def __init__(
        self,
        *,
        minimum_clearance_m: float = 1.35,
        minimum_node_spacing_m: float = 2.0,
        maximum_nodes: int = 24,
        junction_merge_radius_m: float = 2.25,
        branch_ring_radius_m: float = 1.1,
        open_branch_clearance_m: float = 1.55,
        minimum_component_area_m2: float = 4.1,
        local_maximum_radius_m: float = 0.9,
    ) -> None:
        self.minimum_clearance_m = minimum_clearance_m
        self.minimum_node_spacing_m = minimum_node_spacing_m
        self.maximum_nodes = maximum_nodes
        self.junction_merge_radius_m = junction_merge_radius_m
        self.branch_ring_radius_m = branch_ring_radius_m
        self.open_branch_clearance_m = open_branch_clearance_m
        self.minimum_component_area_m2 = minimum_component_area_m2
        self.local_maximum_radius_m = local_maximum_radius_m

    def cache_key(self, grid: NavigabilityGrid) -> str:
        payload = (
            f"{grid.cache_key}:{self.algorithm_version}:{self.minimum_clearance_m}:"
            f"{self.minimum_node_spacing_m}:{self.maximum_nodes}:"
            f"{self.junction_merge_radius_m}:{self.branch_ring_radius_m}:"
            f"{self.open_branch_clearance_m}:{self.minimum_component_area_m2}:"
            f"{self.local_maximum_radius_m}"
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:20]

    def extract(self, grid: NavigabilityGrid) -> BackboneTopology:
        import cv2
        import numpy as np

        free = self._retain_components(
            np.asarray(grid.free_mask, dtype=np.uint8),
            grid.resolution_m,
        )
        distance_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        skeleton = self._thin(free)
        crossings = self._transitions(skeleton)
        branch_pixels = np.column_stack(np.where((skeleton > 0) & (crossings >= 3)))

        candidates: list[tuple[str, int, int, float]] = []
        merge_radius_px = max(1.0, self.junction_merge_radius_m / grid.resolution_m)
        ring_radius_px = max(2, int(round(self.branch_ring_radius_m / grid.resolution_m)))
        for pixels in self._single_link_clusters(branch_pixels, merge_radius_px):
            values = distance_px[pixels[:, 0], pixels[:, 1]]
            index = int(np.argmax(values))
            row, column = int(pixels[index, 0]), int(pixels[index, 1])
            clearance = float(values[index])
            node_type = (
                "open_space_center"
                if clearance * grid.resolution_m >= self.open_branch_clearance_m
                else "junction"
            )
            if node_type == "junction" and self._radial_branch_count(
                skeleton,
                row=row,
                column=column,
                radius_px=ring_radius_px,
            ) < 3:
                continue
            candidates.append((node_type, row, column, clearance))

        clearance_px = max(1.0, self.minimum_clearance_m / grid.resolution_m)
        local_radius_px = max(1, int(round(self.local_maximum_radius_m / grid.resolution_m)))
        local_kernel = np.ones((local_radius_px * 2 + 1, local_radius_px * 2 + 1), np.uint8)
        local_max = ((distance_px >= cv2.dilate(distance_px, local_kernel) - 1e-6) & (distance_px >= clearance_px)).astype(np.uint8)
        count, labels = cv2.connectedComponents(local_max)
        spacing_px = self.minimum_node_spacing_m / grid.resolution_m
        for label in range(1, count):
            rows, columns = np.where(labels == label)
            if not len(rows):
                continue
            values = distance_px[rows, columns]
            index = int(np.argmax(values))
            row, column = int(rows[index]), int(columns[index])
            if any(
                math.hypot(row - old_row, column - old_column) < spacing_px
                for _, old_row, old_column, _ in candidates
            ):
                continue
            candidates.append(("open_space_center", row, column, float(values[index])))

        if len(candidates) > self.maximum_nodes:
            candidates = sorted(candidates, key=lambda item: item[3], reverse=True)[
                : self.maximum_nodes
            ]
        selected = sorted(candidates, key=lambda item: (item[1], item[2]))

        nodes: list[BackboneNode] = []
        for index, (node_type, row, column, clearance) in enumerate(selected):
            x, y = grid.pixel_to_world(row, column)
            nodes.append(
                BackboneNode(
                    node_id=f"backbone-{index:03d}",
                    node_type=node_type,
                    x=x,
                    y=y,
                    row=row,
                    column=column,
                    clearance_m=clearance * grid.resolution_m,
                    navigation_point_xyz_m=grid.navigation_point(x, y),
                )
            )

        _, free_components = cv2.connectedComponents(free)
        edges = self._skeleton_adjacency_edges(
            nodes,
            skeleton,
            free_components,
            grid,
        )
        return BackboneTopology(
            map_id=grid.map_id,
            geometry_revision=grid.geometry_revision,
            source=grid.source,
            frame=grid.frame,
            cache_key=self.cache_key(grid),
            nodes=nodes,
            edges=edges,
        )

    def _retain_components(self, free: Any, resolution_m: float) -> Any:
        import cv2
        import numpy as np

        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, 8)
        minimum_pixels = max(1, int(round(self.minimum_component_area_m2 / resolution_m**2)))
        retained = [
            component
            for component in range(1, count)
            if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_pixels
        ]
        return np.isin(labels, retained).astype(np.uint8)

    @staticmethod
    def _transitions(image: Any) -> Any:
        import numpy as np

        padded = np.pad(image, 1)
        neighbors = [
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, 2:],
            padded[2:, 2:],
            padded[2:, 1:-1],
            padded[2:, :-2],
            padded[1:-1, :-2],
            padded[:-2, :-2],
        ]
        return sum(
            (
                (neighbors[index] == 0)
                & (neighbors[(index + 1) % len(neighbors)] == 1)
            ).astype(np.uint8)
            for index in range(len(neighbors))
        )

    @classmethod
    def _thin(cls, image: Any) -> Any:
        import numpy as np

        result = (image > 0).astype(np.uint8)
        while True:
            changed = 0
            for step in (0, 1):
                padded = np.pad(result, 1)
                neighbors = [
                    padded[:-2, 1:-1],
                    padded[:-2, 2:],
                    padded[1:-1, 2:],
                    padded[2:, 2:],
                    padded[2:, 1:-1],
                    padded[2:, :-2],
                    padded[1:-1, :-2],
                    padded[:-2, :-2],
                ]
                neighbor_count = sum(neighbors)
                crossing_count = cls._transitions(result)
                north, _, east, _, south, _, west, _ = neighbors
                if step == 0:
                    connectivity = (north * east * south == 0) & (east * south * west == 0)
                else:
                    connectivity = (north * east * west == 0) & (north * south * west == 0)
                removable = (
                    (result == 1)
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (crossing_count == 1)
                    & connectivity
                )
                changed += int(removable.sum())
                result[removable] = 0
            if changed == 0:
                return result

    @staticmethod
    def _single_link_clusters(points: Any, radius_px: float) -> list[Any]:
        import numpy as np

        if not len(points):
            return []
        parents = list(range(len(points)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        cell_size = max(1.0, radius_px)
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, (row, column) in enumerate(points):
            cell = (int(row // cell_size), int(column // cell_size))
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    for other in buckets.get(
                        (cell[0] + row_offset, cell[1] + column_offset), []
                    ):
                        if math.hypot(
                            float(row - points[other, 0]),
                            float(column - points[other, 1]),
                        ) <= radius_px:
                            union(index, other)
            buckets.setdefault(cell, []).append(index)

        clusters: dict[int, list[int]] = {}
        for index in range(len(points)):
            clusters.setdefault(find(index), []).append(index)
        return [points[np.asarray(indices)] for indices in clusters.values()]

    @staticmethod
    def _radial_branch_count(
        skeleton: Any,
        *,
        row: int,
        column: int,
        radius_px: int,
    ) -> int:
        import cv2
        import numpy as np

        row_grid, column_grid = np.indices(skeleton.shape)
        distance = np.hypot(column_grid - column, row_grid - row)
        ring = (
            (distance >= radius_px - 1)
            & (distance <= radius_px + 1)
            & (skeleton > 0)
        ).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(ring, 8)
        return sum(
            int(stats[component, cv2.CC_STAT_AREA]) > 0
            for component in range(1, count)
        )

    @staticmethod
    def _skeleton_adjacency_edges(
        nodes: list[BackboneNode],
        skeleton: Any,
        free_components: Any,
        grid: NavigabilityGrid,
    ) -> list[BackboneEdge]:
        """Connect node Voronoi regions only where they meet along the skeleton."""
        import heapq

        import numpy as np

        if len(nodes) < 2 or not np.any(skeleton):
            return []

        skeleton_pixels = np.column_stack(np.where(skeleton > 0))
        owners = np.full(skeleton.shape, -1, dtype=np.int32)
        distances = np.full(skeleton.shape, np.inf, dtype=np.float64)
        previous_rows = np.full(skeleton.shape, -1, dtype=np.int32)
        previous_columns = np.full(skeleton.shape, -1, dtype=np.int32)
        queue: list[tuple[float, int, int, int]] = []

        for node_index, node in enumerate(nodes):
            component_id = int(free_components[node.row, node.column])
            if component_id == 0:
                continue
            same_component = skeleton_pixels[
                free_components[skeleton_pixels[:, 0], skeleton_pixels[:, 1]] == component_id
            ]
            if not len(same_component):
                continue
            squared_distance = (
                np.square(same_component[:, 0] - node.row)
                + np.square(same_component[:, 1] - node.column)
            )
            seed = same_component[int(np.argmin(squared_distance))]
            row, column = int(seed[0]), int(seed[1])
            seed_distance = math.hypot(row - node.row, column - node.column)
            if seed_distance >= distances[row, column]:
                continue
            owners[row, column] = node_index
            distances[row, column] = seed_distance
            heapq.heappush(queue, (seed_distance, node_index, row, column))

        neighbor_offsets = [
            (row_offset, column_offset, math.hypot(row_offset, column_offset))
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
            if row_offset or column_offset
        ]
        height, width = skeleton.shape
        while queue:
            distance, owner, row, column = heapq.heappop(queue)
            if distance != distances[row, column] or owner != owners[row, column]:
                continue
            for row_offset, column_offset, step_distance in neighbor_offsets:
                next_row, next_column = row + row_offset, column + column_offset
                if (
                    next_row < 0
                    or next_column < 0
                    or next_row >= height
                    or next_column >= width
                    or not skeleton[next_row, next_column]
                ):
                    continue
                next_distance = distance + step_distance
                if next_distance < distances[next_row, next_column]:
                    distances[next_row, next_column] = next_distance
                    owners[next_row, next_column] = owner
                    previous_rows[next_row, next_column] = row
                    previous_columns[next_row, next_column] = column
                    heapq.heappush(
                        queue,
                        (next_distance, owner, next_row, next_column),
                    )

        candidates: dict[
            tuple[int, int],
            tuple[float, tuple[int, int], tuple[int, int]],
        ] = {}
        rows, columns = np.where(skeleton > 0)
        for row, column in zip(rows.tolist(), columns.tolist()):
            owner = int(owners[row, column])
            if owner < 0:
                continue
            for row_offset, column_offset, step_distance in neighbor_offsets:
                next_row, next_column = row + row_offset, column + column_offset
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                other = int(owners[next_row, next_column])
                if other < 0 or other == owner:
                    continue
                pair = tuple(sorted((owner, other)))
                route_distance = (
                    distances[row, column]
                    + step_distance
                    + distances[next_row, next_column]
                )
                current = candidates.get(pair)
                if current is None or route_distance < current[0]:
                    candidates[pair] = (
                        route_distance,
                        (row, column),
                        (next_row, next_column),
                    )

        edges: list[BackboneEdge] = []
        for (source, target), (distance, left, right) in sorted(candidates.items()):
            if int(owners[left]) != source:
                left, right = right, left
            source_half = SparseBackboneExtractor._trace_to_seed(
                left,
                previous_rows,
                previous_columns,
            )
            target_half = SparseBackboneExtractor._trace_to_seed(
                right,
                previous_rows,
                previous_columns,
            )
            pixel_path = [*reversed(source_half), *target_half]
            source_pixel = (nodes[source].row, nodes[source].column)
            target_pixel = (nodes[target].row, nodes[target].column)
            if pixel_path[0] != source_pixel:
                pixel_path.insert(0, source_pixel)
            if pixel_path[-1] != target_pixel:
                pixel_path.append(target_pixel)
            component_id = int(free_components[source_pixel])
            pixel_path = SparseBackboneExtractor._simplify_free_path(
                pixel_path,
                free_components,
                component_id,
            )
            edges.append(
                BackboneEdge(
                    source=nodes[source].node_id,
                    target=nodes[target].node_id,
                    distance_m=distance * grid.resolution_m,
                    path_world_xy_m=[
                        grid.pixel_to_world(row, column) for row, column in pixel_path
                    ],
                )
            )
        return edges

    @staticmethod
    def _trace_to_seed(
        point: tuple[int, int],
        previous_rows: Any,
        previous_columns: Any,
    ) -> list[tuple[int, int]]:
        path = [point]
        row, column = point
        while previous_rows[row, column] >= 0:
            row, column = (
                int(previous_rows[row, column]),
                int(previous_columns[row, column]),
            )
            path.append((row, column))
        return path

    @staticmethod
    def _simplify_free_path(
        path: list[tuple[int, int]],
        free_components: Any,
        component_id: int,
    ) -> list[tuple[int, int]]:
        import numpy as np

        if len(path) <= 2:
            return path

        def line_is_free(start: tuple[int, int], end: tuple[int, int]) -> bool:
            steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
            rows = np.rint(np.linspace(start[0], end[0], steps)).astype(np.int32)
            columns = np.rint(np.linspace(start[1], end[1], steps)).astype(np.int32)
            return bool(np.all(free_components[rows, columns] == component_id))

        simplified = [path[0]]
        start = 0
        while start < len(path) - 1:
            end = len(path) - 1
            while end > start + 1 and not line_is_free(path[start], path[end]):
                end -= 1
            simplified.append(path[end])
            start = end
        return simplified


class BackboneTopologyCache:
    def __init__(self, cache_dir: Path, extractor: SparseBackboneExtractor | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.extractor = extractor or SparseBackboneExtractor()

    def get_or_build(self, grid: NavigabilityGrid) -> tuple[BackboneTopology, bool]:
        path = self._cache_path(grid)
        if path.exists():
            return BackboneTopology.model_validate_json(path.read_text(encoding="utf-8")), True
        topology = self.extractor.extract(grid)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(topology.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
        return topology, False

    def load(self, *, map_id: str, cache_key: str) -> BackboneTopology | None:
        path = self._path_for_key(map_id, cache_key)
        if not path.is_file():
            return None
        return BackboneTopology.model_validate_json(path.read_text(encoding="utf-8"))

    def _cache_path(self, grid: NavigabilityGrid) -> Path:
        cache_key_builder = getattr(self.extractor, "cache_key", None)
        cache_key = cache_key_builder(grid) if callable(cache_key_builder) else grid.cache_key
        return self._path_for_key(grid.map_id, cache_key)

    def _path_for_key(self, map_id: str, cache_key: str) -> Path:
        map_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", map_id).strip("._") or "map"
        return self.cache_dir / f"{map_slug}-{cache_key}.json"


def topologies_are_planar_equivalent(
    previous: BackboneTopology,
    current: BackboneTopology,
    *,
    node_tolerance_m: float = 0.10,
    edge_tolerance_m: float = 0.25,
) -> bool:
    """Accept a semantic-map rebind only when the 2D route graph stayed stable."""
    if previous.map_id != current.map_id or previous.source != current.source:
        return False
    previous_nodes = {node.node_id: node for node in previous.nodes}
    current_nodes = {node.node_id: node for node in current.nodes}
    if previous_nodes.keys() != current_nodes.keys():
        return False
    for node_id, old_node in previous_nodes.items():
        new_node = current_nodes[node_id]
        if old_node.node_type != new_node.node_type:
            return False
        if math.hypot(old_node.x - new_node.x, old_node.y - new_node.y) > node_tolerance_m:
            return False

    def edge_index(topology: BackboneTopology) -> dict[tuple[str, str], float]:
        return {
            tuple(sorted((edge.source, edge.target))): edge.distance_m
            for edge in topology.edges
        }

    previous_edges = edge_index(previous)
    current_edges = edge_index(current)
    if previous_edges.keys() != current_edges.keys():
        return False
    return all(
        abs(previous_edges[key] - current_edges[key]) <= edge_tolerance_m
        for key in previous_edges
    )
