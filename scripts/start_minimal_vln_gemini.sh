#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_URL="${MINIMAL_VLN_TOOL_URL:-http://127.0.0.1:8770}"
BENCHMARK_RUNS="${GEMINI_BENCHMARK_RUNS:-0}"
SECRETS_FILE="${MINIMAL_VLN_SECRETS_FILE:-${PROJECT_ROOT}/.env.local}"

if [[ -f "${SECRETS_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}"
  set +a
fi

if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "Set GOOGLE_API_KEY or GEMINI_API_KEY before starting Gemini Flash." >&2
    exit 2
  fi
  read -r -s -p "Gemini API key: " GEMINI_API_KEY
  echo
  test -n "${GEMINI_API_KEY}" || {
    echo "Gemini API key is required." >&2
    exit 2
  }
  export GEMINI_API_KEY
fi

if [[ "${BENCHMARK_RUNS}" != "0" ]]; then
  OBSERVATION_JSON="$(curl --fail --silent --show-error --max-time 30 \
    -X POST "${TOOL_URL%/}/observe" \
    -H 'Content-Type: application/json' \
    -d '{}')"
  FRONT_IMAGE="$("${PROJECT_ROOT}/.venv/bin/python" -c \
    'import json,sys; print(json.load(sys.stdin)["image_paths"]["front"])' \
    <<<"${OBSERVATION_JSON}")"
  LOG_DIR="${PROJECT_ROOT}/logs/minimal_vln"
  LOG_PATH="${LOG_DIR}/gemini_flash_latency_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${LOG_DIR}"
  echo "Benchmarking Gemini Flash with ${BENCHMARK_RUNS} front-image requests..."
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/benchmark_gemini_flash.py" \
    "${FRONT_IMAGE}" --runs "${BENCHMARK_RUNS}" 2>&1 | tee "${LOG_PATH}"
  echo "Latency log: ${LOG_PATH}"
fi

export MINIMAL_VLN_HERMES_CONFIG="${PROJECT_ROOT}/configs/minimal_vln_gemini_flash.yaml"
exec bash "${PROJECT_ROOT}/scripts/start_minimal_vln_agent.sh" "$@"
