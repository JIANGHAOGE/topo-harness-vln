from __future__ import annotations

from pathlib import Path

from agentos.schemas.events import Event


class JsonlEventJournal:
    """Append-only event sink suitable for replay and audit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Event) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")

