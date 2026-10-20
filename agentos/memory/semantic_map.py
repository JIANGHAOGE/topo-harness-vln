from __future__ import annotations

import json
import re
from pathlib import Path

from agentos.schemas.semantic_map import EditableSemanticMap, NodeSemantics
from agentos.topology import (
    BackboneTopology,
    BackboneTopologyCache,
    topologies_are_planar_equivalent,
)


class SemanticMapStore:
    """Human-editable, environment-level semantics keyed by stable backbone IDs."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def path_for(self, map_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", map_id).strip("._") or "map"
        return self.root_dir / f"{slug}.json"

    def load(
        self,
        *,
        map_id: str,
        geometry_revision: str,
        topology_cache_key: str = "",
    ) -> EditableSemanticMap:
        path = self.path_for(map_id)
        if not path.is_file():
            return EditableSemanticMap(
                map_id=map_id,
                geometry_revision=geometry_revision,
                topology_cache_key=topology_cache_key,
            )
        semantic_map = EditableSemanticMap.model_validate_json(path.read_text(encoding="utf-8"))
        if semantic_map.map_id != map_id:
            raise ValueError(
                f"semantic map_id mismatch: expected {map_id}, got {semantic_map.map_id}"
            )
        if semantic_map.geometry_revision != geometry_revision:
            raise ValueError(
                "semantic geometry_revision mismatch: "
                f"expected {geometry_revision}, got {semantic_map.geometry_revision}"
            )
        return semantic_map

    def save(self, semantic_map: EditableSemanticMap) -> Path:
        path = self.path_for(semantic_map.map_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(semantic_map.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load_for_topology(
        self,
        topology: BackboneTopology,
        topology_cache: BackboneTopologyCache,
        *,
        persist_rebind: bool = True,
    ) -> tuple[EditableSemanticMap, dict[str, str] | None]:
        """Load semantics, rebinding only across an equivalent planar route graph."""
        path = self.path_for(topology.map_id)
        if not path.is_file():
            return (
                EditableSemanticMap(
                    map_id=topology.map_id,
                    geometry_revision=topology.geometry_revision,
                    topology_cache_key=topology.cache_key,
                ),
                None,
            )
        semantic_map = EditableSemanticMap.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if semantic_map.map_id != topology.map_id:
            raise ValueError(
                f"semantic map_id mismatch: expected {topology.map_id}, got {semantic_map.map_id}"
            )
        if semantic_map.geometry_revision == topology.geometry_revision:
            return semantic_map, None

        previous = topology_cache.load(
            map_id=semantic_map.map_id,
            cache_key=semantic_map.topology_cache_key,
        )
        if previous is None or not topologies_are_planar_equivalent(previous, topology):
            raise ValueError(
                "semantic geometry_revision mismatch and topology changed: "
                f"expected {topology.geometry_revision}, got {semantic_map.geometry_revision}"
            )
        old_revision = semantic_map.geometry_revision
        old_cache_key = semantic_map.topology_cache_key
        rebound = semantic_map.model_copy(
            update={
                "geometry_revision": topology.geometry_revision,
                "topology_cache_key": topology.cache_key,
            }
        )
        if persist_rebind:
            self.save(rebound)
        return rebound, {
            "from_geometry_revision": old_revision,
            "from_topology_cache_key": old_cache_key,
        }

    def upsert(
        self,
        *,
        map_id: str,
        geometry_revision: str,
        topology_cache_key: str,
        topology_node_id: str,
        semantics: NodeSemantics,
    ) -> Path:
        semantic_map = self.load(
            map_id=map_id,
            geometry_revision=geometry_revision,
            topology_cache_key=topology_cache_key,
        )
        nodes = dict(semantic_map.nodes)
        nodes[topology_node_id] = semantics
        semantic_map = semantic_map.model_copy(
            update={
                "topology_cache_key": topology_cache_key,
                "nodes": nodes,
            }
        )
        return self.save(semantic_map)
