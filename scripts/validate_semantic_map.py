from __future__ import annotations

import argparse
from pathlib import Path

from agentos.schemas.semantic_map import EditableSemanticMap


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a human-editable node semantic map.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    semantic_map = EditableSemanticMap.model_validate_json(
        args.path.read_text(encoding="utf-8")
    )
    print(
        f"valid semantic map: map_id={semantic_map.map_id} "
        f"geometry_revision={semantic_map.geometry_revision} nodes={len(semantic_map.nodes)}"
    )


if __name__ == "__main__":
    main()
