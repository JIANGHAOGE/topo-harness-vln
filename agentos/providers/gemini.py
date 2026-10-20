from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from pydantic import Field

from agentos.providers.base import parse_json_content
from agentos.providers.vision import gemini_parts, text_content
from agentos.schemas.base import SchemaModel
from agentos.schemas.llm import ChatMessage, LLMResponse


class GeminiProviderConfig(SchemaModel):
    provider: Literal["gemini"] = "gemini"
    model: str = Field(min_length=1)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key: str | None = None
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)


class GeminiProvider:
    def __init__(
        self,
        config: GeminiProviderConfig,
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
            "contents": self._contents(messages),
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        system_instruction = self._system_instruction(messages)
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if response_schema is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseJsonSchema"] = response_schema

        response = await self._client.post(self._generate_url(), headers=self._headers(), json=body)
        response.raise_for_status()
        raw = response.json()
        content = self._extract_text(raw)
        return LLMResponse(
            model=self.config.model,
            content=content,
            raw=raw,
            parsed_json=parse_json_content(content),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _generate_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/models/{self.config.model}:generateContent"

    def _headers(self) -> dict[str, str]:
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing Gemini API key. Set {self.config.api_key_env} or config.api_key.")
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    @staticmethod
    def _system_instruction(messages: list[ChatMessage]) -> str:
        return "\n".join(text_content(message.content) for message in messages if message.role == "system").strip()

    @staticmethod
    def _contents(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        contents = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": gemini_parts(message.content)})
        return contents

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> str:
        candidates = raw.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(part.get("text", "")) for part in parts)
