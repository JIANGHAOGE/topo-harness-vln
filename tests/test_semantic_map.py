from __future__ import annotations

import json

import pytest

from agentos.memory.navigation import AgentNavigationMemory
from agentos.memory.semantic_map import SemanticMapStore
from agentos.memory.store import InMemoryGraphMemory
from agentos.schemas.semantic_map import EditableSemanticMap, NodeSemantics
from agentos.topology import BackboneNode, BackboneTopology, BackboneTopologyCache


def _topology() -> BackboneTopology:
    return BackboneTopology(
        map_id="test-floor",
        geometry_revision="revision-1",
        source="ros2_occupancy_grid",
        frame="map",
        cache_key="cache-1",
        nodes=[
            BackboneNode(
                node_id="backbone-000",
                node_type="open_space_center",
                x=1.0,
                y=2.0,
                row=1,
                column=2,
                clearance_m=1.0,
                navigation_point_xyz_m=[1.0, 2.0, 0.0],
            )
        ],
    )


def test_editable_semantic_map_round_trip_and_human_override(tmp_path):
    store = SemanticMapStore(tmp_path)
    semantic_map = EditableSemanticMap(
        map_id="test-floor",
        geometry_revision="revision-1",
        topology_cache_key="cache-1",
        nodes={
            "backbone-000": NodeSemantics(
                place_type="laboratory",
                function="robot testing",
                objects=["workbench", "robot"],
                aliases=["lab"],
                confidence=0.9,
            )
        },
    )
    path = store.save(semantic_map)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["nodes"]["backbone-000"]["place_type"] = "bedroom"
    payload["nodes"]["backbone-000"]["source"] = "human"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load(map_id="test-floor", geometry_revision="revision-1")

    assert loaded.nodes["backbone-000"].place_type == "bedroom"
    assert loaded.nodes["backbone-000"].source == "human"


def test_semantic_map_rejects_wrong_geometry_revision(tmp_path):
    store = SemanticMapStore(tmp_path)
    store.save(
        EditableSemanticMap(
            map_id="test-floor",
            geometry_revision="old-revision",
        )
    )

    with pytest.raises(ValueError, match="geometry_revision mismatch"):
        store.load(map_id="test-floor", geometry_revision="revision-1")


def test_semantic_map_rebinds_when_planar_topology_is_unchanged(tmp_path):
    cache = BackboneTopologyCache(tmp_path / "topology")
    store = SemanticMapStore(tmp_path / "semantics")
    previous = _topology().model_copy(
        update={"geometry_revision": "old-revision", "cache_key": "old-cache"}
    )
    current = _topology().model_copy(
        update={
            "geometry_revision": "new-revision",
            "cache_key": "new-cache",
            "nodes": [
                _topology().nodes[0].model_copy(
                    update={"navigation_point_xyz_m": [1.0, 2.0, 0.009]}
                )
            ],
        }
    )
    cache.cache_dir.mkdir(parents=True)
    (cache.cache_dir / "test-floor-old-cache.json").write_text(
        previous.model_dump_json(indent=2),
        encoding="utf-8",
    )
    store.save(
        EditableSemanticMap(
            map_id="test-floor",
            geometry_revision="old-revision",
            topology_cache_key="old-cache",
            nodes={
                "backbone-000": NodeSemantics(
                    place_type="office",
                    objects=["refrigerator"],
                    source="human",
                )
            },
        )
    )

    semantic_map, rebound = store.load_for_topology(current, cache)

    assert semantic_map.geometry_revision == "new-revision"
    assert semantic_map.topology_cache_key == "new-cache"
    assert semantic_map.nodes["backbone-000"].objects == ["refrigerator"]
    assert rebound == {
        "from_geometry_revision": "old-revision",
        "from_topology_cache_key": "old-cache",
    }
    persisted = json.loads(store.path_for("test-floor").read_text(encoding="utf-8"))
    assert persisted["geometry_revision"] == "new-revision"


def test_semantic_map_rebind_rejects_changed_planar_nodes(tmp_path):
    cache = BackboneTopologyCache(tmp_path / "topology")
    store = SemanticMapStore(tmp_path / "semantics")
    previous = _topology().model_copy(
        update={"geometry_revision": "old-revision", "cache_key": "old-cache"}
    )
    moved_node = _topology().nodes[0].model_copy(update={"x": 2.0})
    current = _topology().model_copy(
        update={
            "geometry_revision": "new-revision",
            "cache_key": "new-cache",
            "nodes": [moved_node],
        }
    )
    cache.cache_dir.mkdir(parents=True)
    (cache.cache_dir / "test-floor-old-cache.json").write_text(
        previous.model_dump_json(indent=2),
        encoding="utf-8",
    )
    store.save(
        EditableSemanticMap(
            map_id="test-floor",
            geometry_revision="old-revision",
            topology_cache_key="old-cache",
        )
    )

    with pytest.raises(ValueError, match="topology changed"):
        store.load_for_topology(current, cache)


def test_semantic_map_can_rebind_in_memory_without_updating_the_json(tmp_path):
    cache = BackboneTopologyCache(tmp_path / "topology")
    store = SemanticMapStore(tmp_path / "semantics")
    previous = _topology().model_copy(
        update={"geometry_revision": "old-revision", "cache_key": "old-cache"}
    )
    current = _topology().model_copy(
        update={"geometry_revision": "new-revision", "cache_key": "new-cache"}
    )
    cache.cache_dir.mkdir(parents=True)
    (cache.cache_dir / "test-floor-old-cache.json").write_text(
        previous.model_dump_json(indent=2),
        encoding="utf-8",
    )
    path = store.save(
        EditableSemanticMap(
            map_id="test-floor",
            geometry_revision="old-revision",
            topology_cache_key="old-cache",
            nodes={
                "backbone-000": NodeSemantics(
                    place_type="kitchen",
                    objects=["robot"],
                )
            },
        )
    )

    rebound, metadata = store.load_for_topology(
        current,
        cache,
        persist_rebind=False,
    )

    persisted = EditableSemanticMap.model_validate_json(path.read_text(encoding="utf-8"))
    assert rebound.geometry_revision == "new-revision"
    assert rebound.nodes["backbone-000"].objects == ["robot"]
    assert metadata is not None
    assert persisted.geometry_revision == "old-revision"
    assert persisted.topology_cache_key == "old-cache"


def test_navigation_memory_uses_node_semantics_for_search():
    memory = AgentNavigationMemory(
        InMemoryGraphMemory(),
        environment_id="robot:test-floor",
        task_id="task-1",
        episode_id="episode-1",
    )
    materialized = memory.record_backbone_topology(_topology())
    result = memory.apply_semantic_map(
        EditableSemanticMap(
            map_id="test-floor",
            geometry_revision="revision-1",
            nodes={
                "backbone-000": NodeSemantics(
                    place_type="bedroom",
                    function="sleeping and resting",
                    objects=["bed", "bedside table"],
                    landmarks=["wooden bed frame"],
                    aliases=["sleeping room"],
                    source="human",
                )
            },
        )
    )

    location_id = materialized["node_locations"]["backbone-000"]
    place = memory.get_place(location_id)
    graph_node = memory.route_graph()["nodes"][0]

    assert result["applied_node_count"] == 1
    assert place is not None
    assert place.topology_node_id == "backbone-000"
    assert place.node_semantics is not None
    assert place.node_semantics.place_type == "bedroom"
    assert "bed" in place.semantic_labels
    assert graph_node["node_semantics"]["function"] == "sleeping and resting"
    assert graph_node["scene_summary"] == "bedroom: sleeping and resting"
    search = memory.search_semantic_nodes("find the bedroom")
    assert search[0]["location_id"] == location_id
    assert search[0]["topology_node_id"] == "backbone-000"
    assert search[0]["matched_terms"] == ["bedroom"]
