#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${QWEN_TUNNEL_LOCAL_PORT:-18011}"
REMOTE_PORT="${QWEN_REMOTE_PORT:-8011}"
SSH_TARGET="${QWEN_SSH_TARGET:-algorithm-server-H800}"

if curl -fsS --max-time 2 "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null; then
  echo "Qwen tunnel already healthy on 127.0.0.1:${LOCAL_PORT}."
  exit 0
fi

exec ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o TCPKeepAlive=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${SSH_TARGET}"
