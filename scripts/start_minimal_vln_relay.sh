#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jianghaoge/miniconda3/bin/python}"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m agentos.api.minimal_vln_web_relay_server \
  --host "${MINIMAL_VLN_RELAY_HOST:-127.0.0.1}" \
  --port "${MINIMAL_VLN_RELAY_PORT:-5915}" \
  --tool-url "${MINIMAL_VLN_TOOL_URL:-http://127.0.0.1:8770}"
