from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import yaml

from agentos.providers.gemini import GeminiProvider, GeminiProviderConfig
from agentos.providers.qwen import QwenProvider, QwenProviderConfig


def load_llm_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def build_provider_from_config(
    config: dict[str, Any],
    *,
    active_provider: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
):
    llm = config.get("llm", config)
    if not isinstance(llm, dict):
        raise ValueError("llm config must be a mapping")
    provider_name = active_provider or llm.get("active_provider")
    if not provider_name:
        raise ValueError("Missing llm.active_provider")
    provider_config = llm.get(provider_name)
    if not isinstance(provider_config, dict):
        raise ValueError(f"Missing config section for provider: {provider_name}")
    if provider_name == "qwen":
        return QwenProvider(QwenProviderConfig(**provider_config), transport=transport)
    if provider_name == "gemini":
        return GeminiProvider(GeminiProviderConfig(**provider_config), transport=transport)
    raise ValueError(f"Unsupported provider: {provider_name}")

