from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

import httpx
from pydantic import Field

from agentos.providers.base import parse_json_content
from agentos.schemas.base import SchemaModel
from agentos.schemas.llm import ChatMessage, LLMResponse


class QwenProviderConfig(SchemaModel):
    provider: Literal["qwen"] = "qwen"
    model: str = Field(min_length=1)
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    api_key_env: str = "QWEN_API_KEY"
    temperature: float = 0.0
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)


class QwenProvider:
    def __init__(
        self,
        config: QwenProviderConfig,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout_s, transport=transport)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agentos_response",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        response = None
        for attempt in range(3):
            try:
                response = await self._client.post(
                    self._chat_url(), headers=self._headers(), json=body
                )
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.25 * (2**attempt))
        assert response is not None
        response.raise_for_status()
        raw = response.json()
        content = str(raw["choices"][0]["message"]["content"])
        return LLMResponse(
            model=str(raw.get("model") or self.config.model),
            content=content,
            raw=raw,
            parsed_json=parse_json_content(content),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing Qwen API key. Set {self.config.api_key_env} or config.api_key.")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
