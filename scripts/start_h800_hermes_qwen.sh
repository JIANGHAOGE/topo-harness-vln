#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="${QWEN_SSH_TARGET:-algorithm-server-H800}"
REMOTE_PORT="${HERMES_QWEN_REMOTE_PORT:-8011}"
GPU_DEVICE="${HERMES_QWEN_GPU_DEVICE:-1}"
CONTAINER_NAME="${HERMES_QWEN_CONTAINER_NAME:-cee-hermes-qwen-8011}"
CACHE_DIR="${HERMES_QWEN_CACHE_DIR:-/user-docker-data/data/jhg/cache/cee-hermes-qwen-vllm}"

ssh "${SSH_TARGET}" \
  "bash -lic 'vpn; export HERMES_QWEN_REMOTE_PORT=${REMOTE_PORT}; export HERMES_QWEN_GPU_DEVICE=${GPU_DEVICE}; export HERMES_QWEN_CONTAINER_NAME=${CONTAINER_NAME}; export HERMES_QWEN_CACHE_DIR=${CACHE_DIR}; bash -s'" <<'REMOTE'
set -euo pipefail

if docker inspect "${HERMES_QWEN_CONTAINER_NAME}" >/dev/null 2>&1; then
  if [ "$(docker inspect -f '{{.State.Running}}' "${HERMES_QWEN_CONTAINER_NAME}")" = "true" ]; then
    echo "Hermes Qwen container is already running on port ${HERMES_QWEN_REMOTE_PORT}."
    exit 0
  fi
  docker rm "${HERMES_QWEN_CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${HERMES_QWEN_CACHE_DIR}"

docker run -d \
  --name "${HERMES_QWEN_CONTAINER_NAME}" \
  --gpus "device=${HERMES_QWEN_GPU_DEVICE}" \
  --ipc=host \
  --shm-size=16g \
  -p "127.0.0.1:${HERMES_QWEN_REMOTE_PORT}:${HERMES_QWEN_REMOTE_PORT}" \
  -v /user-docker-data/data/hwz/qwen36_35b_vl:/qwen36_35b_vl \
  -v "${HERMES_QWEN_CACHE_DIR}:/root/.cache/vllm" \
  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.17.1 \
  --model /qwen36_35b_vl/models/Qwen3.6-35B-A3B-FP8 \
  --served-model-name qwen3.6-vl-35b \
  --host 0.0.0.0 \
  --port "${HERMES_QWEN_REMOTE_PORT}" \
  --trust-remote-code \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml

echo "Started ${HERMES_QWEN_CONTAINER_NAME} on GPU ${HERMES_QWEN_GPU_DEVICE}, port ${HERMES_QWEN_REMOTE_PORT}."
REMOTE
