#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_SOURCE="${PROJECT_ROOT}/third_party/hermes-agent"
HERMES_VENV="${PROJECT_ROOT}/runtime/hermes_venv"

test -f "${HERMES_SOURCE}/pyproject.toml" || {
  echo "Missing Hermes source at ${HERMES_SOURCE}" >&2
  exit 1
}

mkdir -p "${PROJECT_ROOT}/runtime"
if [[ ! -x "${HERMES_VENV}/bin/python" ]]; then
  uv venv --python 3.12 "${HERMES_VENV}"
fi
uv pip install --python "${HERMES_VENV}/bin/python" -e "${HERMES_SOURCE}"
echo "Hermes environment ready: ${HERMES_VENV}"
