#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jianghaoge/miniconda3/bin/python}"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m agentos.api.web_relay_server \
  --host "${CEE_HIERARCHICAL_RELAY_HOST:-127.0.0.1}" \
  --port "${CEE_HIERARCHICAL_RELAY_PORT:-5916}" \
  --tool-url "${CEE_HIERARCHICAL_TOOL_URL:-http://127.0.0.1:8766}"
