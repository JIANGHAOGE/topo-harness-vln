from __future__ import annotations

import asyncio
import json

import httpx

from agentos.providers.config import build_provider_from_config
from agentos.providers.base import parse_json_content
from agentos.providers.gemini import GeminiProvider, GeminiProviderConfig
from agentos.providers.qwen import QwenProvider, QwenProviderConfig
from agentos.schemas.llm import ChatMessage


def test_parse_json_content_accepts_model_prose_around_json():
    content = 'Here is the requested plan:\n{"plan": [{"command": "move"}]}\nDone.'

    assert parse_json_content(content) == {"plan": [{"command": "move"}]}


def test_parse_json_content_rejects_incomplete_embedded_json():
    assert parse_json_content('Partial response: {"plan": [') is None


def test_qwen_provider_uses_openai_compatible_chat_completions():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "qwen-test",
                "choices": [{"message": {"content": "{\"ok\": true}"}}],
            },
        )

    async def run():
        provider = QwenProvider(
            QwenProviderConfig(model="qwen-test", base_url="https://example.test/v1", api_key="key"),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await provider.complete(
                [ChatMessage(role="user", content="Return JSON.")],
                response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )
        finally:
            await provider.close()

    response = asyncio.run(run())
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer key"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert response.parsed_json == {"ok": True}


def test_qwen_provider_retries_transient_transport_disconnect():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(
            200,
            json={
                "model": "qwen-test",
                "choices": [{"message": {"content": "{\"ok\": true}"}}],
            },
        )

    async def run():
        provider = QwenProvider(
            QwenProviderConfig(
                model="qwen-test",
                base_url="https://example.test/v1",
                api_key="key",
            ),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await provider.complete([ChatMessage(role="user", content="Return JSON.")])
        finally:
            await provider.close()

    response = asyncio.run(run())
    assert attempts == 2
    assert response.parsed_json == {"ok": True}


def test_gemini_provider_uses_generate_content_with_schema():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "{\"plan\": []}"}],
                        }
                    }
                ]
            },
        )

    async def run():
        provider = GeminiProvider(
            GeminiProviderConfig(model="gemini-test", base_url="https://example.test/v1beta", api_key="key"),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await provider.complete(
                [
                    ChatMessage(role="system", content="You are a planner."),
                    ChatMessage(role="user", content="Return a plan."),
                ],
                response_schema={"type": "object", "properties": {"plan": {"type": "array"}}},
            )
        finally:
            await provider.close()

    response = asyncio.run(run())
    assert seen["path"] == "/v1beta/models/gemini-test:generateContent"
    assert seen["key"] == "key"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["body"]["generationConfig"]["responseJsonSchema"]["type"] == "object"
    assert response.parsed_json == {"plan": []}


def test_provider_factory_switches_by_config():
    qwen = build_provider_from_config(
        {
            "llm": {
                "active_provider": "qwen",
                "qwen": {"provider": "qwen", "model": "qwen-test", "api_key": "key"},
            }
        }
    )
    gemini = build_provider_from_config(
        {
            "llm": {
                "active_provider": "gemini",
                "gemini": {"provider": "gemini", "model": "gemini-test", "api_key": "key"},
            }
        }
    )
    assert isinstance(qwen, QwenProvider)
    assert isinstance(gemini, GeminiProvider)
    asyncio.run(qwen.close())
    asyncio.run(gemini.close())
