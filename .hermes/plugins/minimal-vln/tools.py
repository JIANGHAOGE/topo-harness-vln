from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VIEW_ORDER = ("left", "front", "right")


def _request(path: str, payload: dict | None = None) -> dict:
    base_url = os.environ.get("MINIMAL_VLN_TOOL_URL", "http://127.0.0.1:8770").rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=35) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        return {"ok": False, "error": f"HTTP {exc.code}: {detail}"}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"minimal VLN service unavailable: {exc}"}
    return result if isinstance(result, dict) else {"ok": False, "error": "invalid service response"}


def observe(_args: dict, **_kwargs) -> dict | str:
    result = _request("/observe", {})
    if not result.get("ok"):
        return json.dumps(result)
    image_paths = result.get("image_paths")
    if not isinstance(image_paths, dict):
        return json.dumps({"ok": False, "error": "RGB payload is unavailable"})
    content = []
    for view in VIEW_ORDER:
        image_path = Path(str(image_paths.get(view, "")))
        if not image_path.is_file():
            return json.dumps({"ok": False, "error": f"{view} RGB image is unavailable"})
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        image_url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
        content.extend(
            [
                {"type": "text", "text": f"{view.upper()} camera RGB observation."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": "Left, front, and right RGB observations.",
    }


def step(args: dict, **_kwargs) -> str:
    result = _request("/step", {"actions": args.get("actions")})
    if result.get("ok", True):
        result = {
            "executed": result.get("executed"),
            "remaining_budget": result.get("remaining_budget"),
            "terminal": result.get("terminal"),
        }
    return json.dumps(result)
