from __future__ import annotations

import json
import re
from typing import Any, Protocol

from agentos.schemas.llm import ChatMessage, LLMResponse


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        ...

    async def close(self) -> None:
        ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_content(content: str) -> dict[str, Any] | list[Any] | None:
    text = content.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _decode_embedded_json(text)
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _decode_embedded_json(text: str) -> Any:
    """Decode the first complete JSON value when a model adds prose around it."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None
