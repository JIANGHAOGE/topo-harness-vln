from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from agentos.schemas.memory import MemoryEdge, MemoryHit, MemoryNode, MemoryQuery


def observation_node_id(episode_id: str, observation_id: str) -> str:
    """Return an episode-scoped source key for one physical observation."""
    return f"source:observation:{episode_id}:{observation_id}"


class GraphMemoryStore(Protocol):
    def upsert_node(self, node: MemoryNode) -> None:
        ...

    def add_edge(self, edge: MemoryEdge) -> None:
        ...

    def get_node(self, node_id: str) -> MemoryNode | None:
        ...

    def search(self, query: MemoryQuery) -> list[MemoryHit]:
        ...

    def neighbors(self, node_id: str, *, limit: int = 20) -> list[MemoryNode]:
        ...


def _terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "at",
        "find",
        "for",
        "go",
        "in",
        "into",
        "locate",
        "navigate",
        "of",
        "please",
        "the",
        "to",
        "toward",
        "towards",
    }
    return {
        term.casefold()
        for term in re.findall(r"[\w\-]+", text, flags=re.UNICODE)
        if term and term.casefold() not in stopwords
    }


def _score(node: MemoryNode, query: MemoryQuery) -> tuple[float, list[str]]:
    query_terms = _terms(query.text)
    if not query_terms:
        return 1.0, []
    searchable = _terms(f"{node.title} {node.content} {json.dumps(node.metadata, ensure_ascii=False)}")
    matches = sorted(query_terms & searchable)
    return len(matches) / len(query_terms), matches


def _matches_filters(node: MemoryNode, query: MemoryQuery) -> bool:
    return (
        (not query.node_types or node.node_type in query.node_types)
        and (query.episode_id is None or node.episode_id == query.episode_id)
        and (query.task_id is None or node.task_id == query.task_id)
        and (not query.persistent_only or node.task_id is None)
    )


class InMemoryGraphMemory:
    def __init__(self) -> None:
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[str, MemoryEdge] = {}

    def upsert_node(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: MemoryEdge) -> None:
        self.edges[edge.edge_id] = edge

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    def search(self, query: MemoryQuery) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for node in self.nodes.values():
            if not _matches_filters(node, query):
                continue
            score, matched_terms = _score(node, query)
            if query.text and score == 0.0:
                continue
            hits.append(MemoryHit(node=node, score=score, matched_terms=matched_terms))
        hits.sort(key=lambda hit: (hit.score, hit.node.created_at), reverse=True)
        return hits[: query.limit]

    def neighbors(self, node_id: str, *, limit: int = 20) -> list[MemoryNode]:
        neighbor_ids: list[str] = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbor_ids.append(edge.target_id)
            elif edge.target_id == node_id:
                neighbor_ids.append(edge.source_id)
        return [self.nodes[item] for item in neighbor_ids[:limit] if item in self.nodes]


class SQLiteGraphMemory:
    """Persistent source-grounded graph memory; raw image bytes stay outside SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        # Hermes currently ships with SQLite 3.50.4, whose WAL implementation
        # has a known reset issue. Memory writes are small, so prefer safety.
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _create_schema(self) -> None:
        with self._lock:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    episode_id TEXT,
                    task_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_nodes_episode ON memory_nodes(episode_id)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_nodes_task ON memory_nodes(task_id)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_edges_source ON memory_edges(source_id)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_edges_target ON memory_edges(target_id)")
            self.connection.commit()

    def upsert_node(self, node: MemoryNode) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO memory_nodes(node_id, node_type, episode_id, task_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    node_type=excluded.node_type,
                    episode_id=excluded.episode_id,
                    task_id=excluded.task_id,
                    created_at=excluded.created_at,
                    payload=excluded.payload
                """,
                (
                    node.node_id,
                    node.node_type.value,
                    node.episode_id,
                    node.task_id,
                    node.created_at.isoformat(),
                    node.model_dump_json(),
                ),
            )
            self.connection.commit()

    def add_edge(self, edge: MemoryEdge) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO memory_edges(edge_id, source_id, target_id, edge_type, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET payload=excluded.payload
                """,
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type.value,
                    edge.created_at.isoformat(),
                    edge.model_dump_json(),
                ),
            )
            self.connection.commit()

    def get_node(self, node_id: str) -> MemoryNode | None:
        with self._lock:
            row = self.connection.execute("SELECT payload FROM memory_nodes WHERE node_id = ?", (node_id,)).fetchone()
        return MemoryNode.model_validate_json(row[0]) if row else None

    def search(self, query: MemoryQuery) -> list[MemoryHit]:
        clauses: list[str] = []
        params: list[str] = []
        if query.node_types:
            placeholders = ",".join("?" for _ in query.node_types)
            clauses.append(f"node_type IN ({placeholders})")
            params.extend(node_type.value for node_type in query.node_types)
        if query.episode_id is not None:
            clauses.append("episode_id = ?")
            params.append(query.episode_id)
        if query.task_id is not None:
            clauses.append("task_id = ?")
            params.append(query.task_id)
        if query.persistent_only:
            clauses.append("task_id IS NULL")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.connection.execute(
                f"SELECT payload FROM memory_nodes{where} ORDER BY created_at DESC LIMIT 500", params
            ).fetchall()
        hits: list[MemoryHit] = []
        for row in rows:
            node = MemoryNode.model_validate_json(row[0])
            score, matched_terms = _score(node, query)
            if query.text and score == 0.0:
                continue
            hits.append(MemoryHit(node=node, score=score, matched_terms=matched_terms))
        hits.sort(key=lambda hit: (hit.score, hit.node.created_at), reverse=True)
        return hits[: query.limit]

    def neighbors(self, node_id: str, *, limit: int = 20) -> list[MemoryNode]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT source_id, target_id FROM memory_edges
                WHERE source_id = ? OR target_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (node_id, node_id, limit),
            ).fetchall()
        neighbor_ids = [target if source == node_id else source for source, target in rows]
        return [node for neighbor_id in neighbor_ids if (node := self.get_node(neighbor_id)) is not None]
