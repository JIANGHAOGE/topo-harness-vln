from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from agentos.topology import NavigabilityGrid, normalize_ros2_occupancy_grid


class NavigabilityMapProvider(Protocol):
    """Mapping-layer source for an environment-neutral traversability grid."""

    @property
    def cache_dir(self) -> Path:
        ...

    def load_grid(self) -> NavigabilityGrid:
        ...


@dataclass
class StaticNavigabilityMapProvider:
    grid: NavigabilityGrid
    cache_dir: Path

    def load_grid(self) -> NavigabilityGrid:
        return self.grid


@dataclass
class HabitatNavmeshMapProvider:
    """Habitat-specific map ingestion kept outside the embodiment adapter."""

    run_serialized: Callable[[Callable[[Any], NavigabilityGrid]], NavigabilityGrid]
    cache_dir: Path
    resolution_px: int = 512

    def load_grid(self) -> NavigabilityGrid:
        return self.run_serialized(self._extract_grid)

    def _extract_grid(self, adapter: Any) -> NavigabilityGrid:
        import numpy as np

        if adapter.sim is None:
            raise RuntimeError("Habitat simulator is not running")
        state = adapter.sim.get_agent(0).get_state()
        height = float(state.position[1])
        lower, upper = adapter.sim.pathfinder.get_bounds()
        span = max(float(upper[0] - lower[0]), float(upper[2] - lower[2]), 1.0)
        resolution = span / float(self.resolution_px)
        free_mask = np.asarray(
            adapter.sim.pathfinder.get_topdown_view(resolution, height),
            dtype=bool,
        )
        scene_id = adapter.config.scene_id if adapter.config is not None else "unknown"
        digest = hashlib.sha256()
        digest.update(
            (
                f"{scene_id}:{free_mask.shape}:{resolution:.8f}:{round(height, 2)}:"
                f"{float(lower[0]):.6f}:{float(lower[2]):.6f}"
            ).encode("ascii")
        )
        digest.update(np.packbits(free_mask).tobytes())
        return NavigabilityGrid(
            map_id=scene_id,
            geometry_revision=digest.hexdigest()[:20],
            source="habitat_navmesh",
            frame="habitat_world_xz",
            resolution_m=resolution,
            origin_xy_m=(float(lower[0]), float(lower[2])),
            free_mask=free_mask,
            navigation_plane="xz",
            navigation_height_m=height,
        )


@dataclass
class ROS2OccupancyGridMapProvider:
    """ROS-independent boundary; a ROS2 node only supplies the latest map message."""

    message_loader: Callable[[], dict[str, Any]]
    map_id: str
    cache_dir: Path
    geometry_revision_loader: Callable[[dict[str, Any]], str | None] | None = None
    free_threshold: int = 20

    def load_grid(self) -> NavigabilityGrid:
        message = self.message_loader()
        revision = (
            self.geometry_revision_loader(message)
            if self.geometry_revision_loader is not None
            else None
        )
        return normalize_ros2_occupancy_grid(
            message,
            map_id=self.map_id,
            geometry_revision=revision,
            free_threshold=self.free_threshold,
        )
