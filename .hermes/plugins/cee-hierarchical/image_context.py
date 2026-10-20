from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any


_IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})
_LOCAL_IMAGE_PREFIX = "cee-local-image:"
_NAVIGATION_TOOLS = frozenset(
    {
        "hierarchical_observe",
        "begin_navigation_task",
        "recall_navigation_memory",
        "advance_navigation",
        "adjust_viewpoint",
        "inspect_localization_map",
        "navigate_to_known_location",
        "navigate_to_visible_anchor",
    }
)
_MOVEMENT_TOOLS = frozenset(
    {"advance_navigation", "navigate_to_known_location", "navigate_to_visible_anchor"}
)


def prune_historical_navigation_images(
    *, request: dict[str, Any], **_kwargs: Any
) -> dict[str, Any] | None:
    """Send a bounded navigation window and hydrate only its local image refs."""
    message_key = "messages" if isinstance(request.get("messages"), list) else "input"
    messages = request.get(message_key)
    if not isinstance(messages, list):
        return None

    navigation_indices = [
        index for index, message in enumerate(messages) if _is_navigation_tool_message(message)
    ]
    image_indices = [index for index in navigation_indices if _has_images(messages[index])]
    if not navigation_indices and not _contains_local_image_ref(messages):
        return None

    keep_tools = _tool_messages_to_keep(messages, navigation_indices, image_indices)
    paired_assistants = {
        pair
        for index in navigation_indices
        if index not in keep_tools
        for pair in [_paired_assistant_index(messages, index)]
        if pair is not None
    }

    newest_visual = image_indices[-1] if image_indices else None
    movement_indices = [index for index in image_indices if _is_movement_message(messages[index])]
    newest_movement = movement_indices[-1] if movement_indices else None
    kept_messages: list[Any] = []
    removed_messages = 0
    removed_images = 0
    for index, message in enumerate(messages):
        if index in paired_assistants or (index in navigation_indices and index not in keep_tools):
            removed_messages += 1
            continue
        if index in keep_tools and isinstance(message, dict):
            trajectory_only = (
                newest_movement is not None
                and index == newest_movement
                and index != newest_visual
            )
            if trajectory_only:
                message, removed = _prune_message(message, trajectory_only=True)
                removed_images += removed
            else:
                message = dict(message)
                message.pop("api_content", None)
        kept_messages.append(message)

    hydrated_messages, hydrated = _hydrate_local_images(kept_messages)
    updated_request = dict(request)
    updated_request[message_key] = hydrated_messages
    return {
        "request": updated_request,
        "source": "cee-hierarchical-context-window",
        "reason": (
            f"kept {len(keep_tools)} navigation results; removed {removed_messages} messages and "
            f"{removed_images} images; hydrated {hydrated} current images"
        ),
    }


def _tool_messages_to_keep(
    messages: list[Any], navigation_indices: list[int], image_indices: list[int]
) -> set[int]:
    if not navigation_indices:
        return set()
    if not image_indices:
        return set(navigation_indices[-2:])

    newest_visual = image_indices[-1]
    keep = {newest_visual}
    movement_indices = [index for index in image_indices if _is_movement_message(messages[index])]
    if movement_indices and movement_indices[-1] != newest_visual:
        keep.add(movement_indices[-1])

    trailing_results = [index for index in navigation_indices if index > newest_visual]
    keep.update(trailing_results[-2:])
    return keep


def _paired_assistant_index(messages: list[Any], tool_index: int) -> int | None:
    tool_message = messages[tool_index]
    call_id = tool_message.get("tool_call_id") if isinstance(tool_message, dict) else None
    for index in range(tool_index - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            break
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if call_id and isinstance(calls, list) and any(
            isinstance(call, dict) and call.get("id") == call_id for call in calls
        ):
            return index
        if index == tool_index - 1 and _assistant_calls_navigation_tool(message):
            return index
    return None


def _assistant_calls_navigation_tool(message: dict[str, Any]) -> bool:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        if name in _NAVIGATION_TOOLS:
            return True
    return False


def _is_navigation_tool_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "tool"
        and _tool_name(message) in _NAVIGATION_TOOLS
    )


def _has_images(message: Any) -> bool:
    content = message.get("content") if isinstance(message, dict) else None
    return isinstance(content, list) and any(_is_image_part(part) for part in content)


def _tool_name(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    value = message.get("name") or message.get("tool_name")
    return str(value) if value is not None else None


def _is_movement_message(message: Any) -> bool:
    name = _tool_name(message)
    if name not in _MOVEMENT_TOOLS:
        return False
    if name != "advance_navigation" or not isinstance(message, dict):
        return True
    content = message.get("content")
    if not isinstance(content, list):
        return False
    try:
        payload = json.loads(_structured_summary(content))
    except json.JSONDecodeError:
        return False
    return (payload.get("decision") or {}).get("action_type") in {
        "visible_anchor",
        "known_location",
    }


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in _IMAGE_TYPES


def _prune_message(
    message: dict[str, Any], *, trajectory_only: bool
) -> tuple[dict[str, Any], int]:
    content = message.get("content")
    assert isinstance(content, list)
    kept_parts: list[dict[str, Any]] = [
        {"type": "text", "text": _structured_summary(content)},
        {
            "type": "text",
            "text": "[Latest movement trajectory keyframes retained as short-term visual memory.]",
        },
    ]
    removed = 0
    keep_next_image = False
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
            text = str(part.get("text", ""))
            keep_next_image = trajectory_only and "trajectory keyframe" in text.casefold()
            if keep_next_image:
                kept_parts.append(dict(part))
            continue
        if _is_image_part(part):
            if keep_next_image:
                kept_parts.append(dict(part))
            else:
                removed += 1
            keep_next_image = False

    updated = dict(message)
    updated["content"] = kept_parts
    updated.pop("api_content", None)
    return updated, removed


def _structured_summary(content: list[Any]) -> str:
    text_parts = [
        str(part.get("text", "")).strip()
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
    ]
    for text in text_parts:
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(payload, dict):
                payload.pop("memory_view", None)
                return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            return text
    return next((text for text in text_parts if text), "Navigation observation completed.")


def _contains_local_image_ref(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_local_image_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_local_image_ref(item) for item in value)
    return isinstance(value, str) and value.startswith(_LOCAL_IMAGE_PREFIX)


def _hydrate_local_images(messages: list[Any]) -> tuple[list[Any], int]:
    hydrated = 0

    def visit(value: Any) -> Any:
        nonlocal hydrated
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        updated = {key: visit(item) for key, item in value.items()}
        image_url = updated.get("image_url")
        if not isinstance(image_url, dict):
            return updated
        url = image_url.get("url")
        if not isinstance(url, str) or not url.startswith(_LOCAL_IMAGE_PREFIX):
            return updated
        path = Path(url.removeprefix(_LOCAL_IMAGE_PREFIX))
        if not path.is_file():
            updated["image_url"] = {"url": "data:image/gif;base64,R0lGODlhAQABAAAAACw="}
            return updated
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        updated["image_url"] = {"url": f"data:{mime};base64,{data}"}
        hydrated += 1
        return updated

    return [visit(message) for message in messages], hydrated
