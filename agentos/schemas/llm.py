from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agentos.schemas.base import SchemaModel


class ChatMessage(SchemaModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]


class LLMResponse(SchemaModel):
    model: str
    content: str
    raw: dict[str, Any]
    parsed_json: dict[str, Any] | list[Any] | None = None
