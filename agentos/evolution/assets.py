from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now


class EvoAssetStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class EvoAssetGate(SchemaModel):
    name: str = Field(min_length=1)
    passed: bool
    summary: str = Field(min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)


class EvoAsset(SchemaModel):
    """Versioned JSON DSL asset; generated Python is intentionally forbidden."""

    asset_id: str = Field(min_length=1)
    asset_type: str = Field(min_length=1)
    version: int = Field(ge=1)
    status: EvoAssetStatus = EvoAssetStatus.CANDIDATE
    trigger_failure: str = Field(min_length=1)
    target_layer: str = "skill"
    triggering_condition: dict[str, Any] = Field(default_factory=dict)
    permitted_action: str = "apply_operations"
    safety_constraints: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    operations: list[dict[str, Any]] = Field(min_length=1)
    gates: list[EvoAssetGate] = Field(default_factory=list)
    parent_version: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class JsonEvoAssetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, asset: EvoAsset) -> Path:
        path = self._path(asset.asset_id, asset.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(asset.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def load(self, asset_id: str, version: int) -> EvoAsset:
        return EvoAsset.model_validate_json(self._path(asset_id, version).read_text(encoding="utf-8"))

    def list_versions(self, asset_id: str) -> list[EvoAsset]:
        directory = self.root / asset_id
        if not directory.exists():
            return []
        assets = [EvoAsset.model_validate_json(path.read_text(encoding="utf-8")) for path in directory.glob("v*.json")]
        return sorted(assets, key=lambda asset: asset.version)

    def promote(self, asset_id: str, version: int) -> EvoAsset:
        asset = self.load(asset_id, version)
        if not asset.gates or not all(gate.passed for gate in asset.gates):
            raise ValueError("All target and regression gates must pass before promotion")
        promoted = asset.model_copy(update={"status": EvoAssetStatus.PROMOTED})
        self.save(promoted)
        (self.root / asset_id / "active.json").write_text(
            json.dumps({"version": version}, indent=2), encoding="utf-8"
        )
        return promoted

    def rollback(self, asset_id: str, version: int) -> EvoAsset:
        active_path = self.root / asset_id / "active.json"
        if not active_path.exists():
            raise FileNotFoundError(f"No promoted version exists for {asset_id}")
        active_version = int(json.loads(active_path.read_text(encoding="utf-8"))["version"])
        if version == active_version:
            return self.load(asset_id, active_version)
        target = self.load(asset_id, version)
        if target.status not in {EvoAssetStatus.PROMOTED, EvoAssetStatus.ROLLED_BACK}:
            raise ValueError("Rollback target must have been promoted previously")
        active = self.load(asset_id, active_version)
        self.save(active.model_copy(update={"status": EvoAssetStatus.ROLLED_BACK}))
        target = target.model_copy(update={"status": EvoAssetStatus.PROMOTED})
        self.save(target)
        active_path.write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")
        return target

    def _path(self, asset_id: str, version: int) -> Path:
        return self.root / asset_id / f"v{version}.json"
