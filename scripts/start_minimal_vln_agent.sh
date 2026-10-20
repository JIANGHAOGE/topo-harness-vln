#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${MINIMAL_VLN_HERMES_HOME:-${PROJECT_ROOT}/runtime/minimal_vln_hermes_home}"
HERMES_BIN="${PROJECT_ROOT}/runtime/hermes_venv/bin/hermes"
HERMES_CONFIG="${MINIMAL_VLN_HERMES_CONFIG:-${PROJECT_ROOT}/configs/minimal_vln_hermes.yaml}"
PYTHON="${PYTHON:-/home/jianghaoge/miniconda3/bin/python}"
INSTRUCTION="${VLN_INSTRUCTION:-${*:-}}"
ACTION_BUDGET="${MINIMAL_VLN_ACTION_BUDGET:-500}"
TIMEOUT_SECONDS="${MINIMAL_VLN_TIMEOUT_SECONDS:-2400}"
TOOL_URL="${MINIMAL_VLN_TOOL_URL:-http://127.0.0.1:8770}"
MODEL_PROVIDER="$("${PYTHON}" -c \
  'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["model"]["provider"])' \
  "${HERMES_CONFIG}")"

test -x "${HERMES_BIN}" || {
  echo "Hermes is not installed. Run: bash scripts/bootstrap_hermes.sh" >&2
  exit 1
}
test -f "${HERMES_CONFIG}" || {
  echo "Hermes config not found: ${HERMES_CONFIG}" >&2
  exit 2
}
mkdir -p "${HERMES_HOME}"
cp "${HERMES_CONFIG}" "${HERMES_HOME}/config.yaml"
cd "${PROJECT_ROOT}"
if [[ -n "${INSTRUCTION}" ]]; then
  "${PYTHON}" -m agentos.minimal_vln_prompt \
    --instruction "${INSTRUCTION}" \
    --budget "${ACTION_BUDGET}" \
    --output "${HERMES_HOME}/SOUL.md"
else
  "${PYTHON}" -m agentos.minimal_vln_prompt \
    --interactive \
    --budget "${ACTION_BUDGET}" \
    --output "${HERMES_HOME}/SOUL.md"
fi

export HERMES_HOME
export HERMES_ENABLE_PROJECT_PLUGINS=1
export MINIMAL_VLN_TOOL_URL="${TOOL_URL}"
if [[ "${MODEL_PROVIDER}" == "gemini" ]]; then
  unset OPENAI_BASE_URL OPENAI_API_KEY
else
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18011/v1}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
fi

# Episode reset is a harness operation and is intentionally not exposed as a model tool.
curl --fail --silent --show-error --max-time 30 \
  -X POST "${TOOL_URL%/}/episode/reset" >/dev/null

if [[ -n "${INSTRUCTION}" ]]; then
  exec timeout --foreground "${TIMEOUT_SECONDS}s" "${HERMES_BIN}" chat \
    --toolsets minimal-vln \
    --max-turns 200 \
    -q "Begin navigating. Call observe() first to see where you are."
fi

exec "${HERMES_BIN}" chat --toolsets minimal-vln --max-turns 200 --cli
