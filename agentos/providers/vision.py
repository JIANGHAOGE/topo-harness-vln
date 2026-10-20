from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentos.providers.base import LLMProvider
from agentos.schemas.llm import ChatMessage
from agentos.schemas.observation import Observation
from agentos.schemas.task_reasoning import SlowBrainSingleViewAudit, SlowBrainTriViewAudit


_VIEW_AUDIT_CACHE_SIZE = 64
_view_audit_cache: OrderedDict[tuple[str, ...], SlowBrainTriViewAudit] = OrderedDict()


def observation_content(prompt: str, observation: Observation) -> list[dict[str, Any]]:
    """Build an OpenAI-compatible, explicitly ordered embodied observation."""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{prompt}\n"
                "The camera order below is robot-left, robot-front, robot-right. "
                "Image-local left/right must not be confused with camera identity."
            ),
        }
    ]
    for view in ("left", "front", "right"):
        ref = observation.rgb_refs.get(view)
        if not ref:
            continue
        content.append({"type": "text", "text": f"ROBOT_{view.upper()}_CAMERA"})
        content.append({"type": "image_url", "image_url": {"url": _data_url(Path(ref))}})
    if observation.pose is not None:
        content.append(
            {
                "type": "text",
                "text": f"Current robot pose: {observation.pose.model_dump(mode='json')}",
            }
        )
    return content


async def audit_observation_views(
    provider: LLMProvider,
    observation: Observation,
) -> SlowBrainTriViewAudit | None:
    """Describe each synchronized camera independently and cache the bound result."""
    refs = observation.rgb_refs
    if any(not refs.get(view) for view in ("left", "front", "right")):
        return None
    key = _view_audit_key(provider, observation)
    cached = _view_audit_cache.get(key)
    if cached is not None:
        _view_audit_cache.move_to_end(key)
        return cached

    async def audit_one(view: str) -> SlowBrainSingleViewAudit:
        prompt = (
            f"Audit only the robot-{view.upper()} camera image from synchronized observation "
            f"{observation.trace.observation_id}. Report visible facts only. visible_objects should list "
            "physical entities. navigable_semantic_anchors should list visible, approachable objects or "
            "transitions with usable ground access, such as doorways, corridor entrances, room openings, "
            "corners, or passages beside landmarks. target_relevant_evidence should list distinctive visual "
            "facts potentially useful for navigation or search. scene_summary must be one concise sentence "
            "under 180 characters describing the place and layout. Do not infer another camera, map the "
            "image's left/center/right regions to camera identities, choose a route, or claim task completion."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            image_ref_content(
                f"CURRENT_OBSERVATION_{observation.trace.observation_id}_{view.upper()}_ONLY",
                refs[view],
            )
        )
        response = await provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        f"You are auditing exactly one robot-{view.upper()} camera image. "
                        "Return compact current-frame visual facts without navigation decisions."
                    ),
                ),
                ChatMessage(role="user", content=content),
            ],
            response_schema=SlowBrainSingleViewAudit.model_json_schema(),
        )
        if not isinstance(response.parsed_json, dict):
            raise ValueError(f"{view} current-view audit returned no JSON object")
        try:
            audit = SlowBrainSingleViewAudit.model_validate(response.parsed_json)
        except ValidationError as exc:
            raise ValueError(f"{view} current-view audit violated its schema: {exc}") from exc
        if audit.view != view:
            raise ValueError(
                f"single-view audit changed camera identity from {view} to {audit.view}"
            )
        return audit

    left, front, right = await asyncio.gather(
        audit_one("left"),
        audit_one("front"),
        audit_one("right"),
    )
    audit = SlowBrainTriViewAudit(left=left, front=front, right=right)
    _view_audit_cache[key] = audit
    _view_audit_cache.move_to_end(key)
    while len(_view_audit_cache) > _VIEW_AUDIT_CACHE_SIZE:
        _view_audit_cache.popitem(last=False)
    return audit


def audited_observation_content(
    prompt: str,
    observation: Observation,
    audit: SlowBrainTriViewAudit | None,
) -> list[dict[str, Any]]:
    """Build model input from camera-bound facts instead of a multi-image request."""
    if audit is None:
        return observation_content(prompt, observation)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{prompt}\n"
                "Authoritative synchronized tri-view audit. Each entry was generated from exactly one "
                "physical robot camera; camera identity must not be reinterpreted:\n"
                f"{json.dumps(audit.model_dump(mode='json'), ensure_ascii=False, separators=(',', ':'))}"
            ),
        }
    ]
    if observation.pose is not None:
        content.append(
            {
                "type": "text",
                "text": f"Current robot pose: {observation.pose.model_dump(mode='json')}",
            }
        )
    return content


def _view_audit_key(provider: LLMProvider, observation: Observation) -> tuple[str, ...]:
    config = getattr(provider, "config", None)
    return (
        type(provider).__qualname__,
        str(getattr(config, "model", "")),
        str(getattr(config, "base_url", "")),
        observation.trace.observation_id,
        str(observation.trace.frame_timestamp),
        observation.rgb_refs["left"],
        observation.rgb_refs["front"],
        observation.rgb_refs["right"],
    )


def image_ref_content(label: str, ref: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": _data_url(Path(ref))}},
    ]


def text_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text", "")) for part in content if part.get("type") == "text").strip()


def gemini_parts(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for item in content:
        kind = item.get("type")
        if kind == "text":
            parts.append({"text": str(item.get("text", ""))})
            continue
        if kind != "image_url":
            continue
        url = str((item.get("image_url") or {}).get("url", ""))
        if not url.startswith("data:") or ";base64," not in url:
            raise ValueError("Gemini embodied observations require data URL images")
        header, data = url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0]
        parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
    return parts


def _data_url(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Observation image is not readable: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
