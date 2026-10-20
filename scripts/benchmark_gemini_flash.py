from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Gemini Flash front-image request latency.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default="gemini-3.6-flash")
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")

    image_data = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Choose exactly one navigation function for this front camera image."},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_data}},
                ],
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "observe",
                        "description": "Capture another front image.",
                        "parameters": {"type": "OBJECT", "properties": {}},
                    },
                    {
                        "name": "step",
                        "description": "Execute one navigation primitive.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "actions": {
                                    "type": "ARRAY",
                                    "items": {"type": "INTEGER", "minimum": 0, "maximum": 3},
                                }
                            },
                            "required": ["actions"],
                        },
                    },
                ]
            }
        ],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        "generationConfig": {
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": False},
        },
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent"
    latencies: list[float] = []
    for index in range(args.runs):
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        function_name = next(
            (part.get("functionCall", {}).get("name") for part in parts if part.get("functionCall")),
            None,
        )
        print(f"run={index + 1} latency_s={elapsed:.3f} function={function_name or 'none'}")

    print(
        json.dumps(
            {
                "model": args.model,
                "runs": len(latencies),
                "cold_s": round(latencies[0], 3),
                "mean_s": round(statistics.mean(latencies), 3),
                "median_s": round(statistics.median(latencies), 3),
                "p90_s": round(percentile(latencies, 0.9), 3),
                "min_s": round(min(latencies), 3),
                "max_s": round(max(latencies), 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
