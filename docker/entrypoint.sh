#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -n "${WAIT_FOR_HOST:-}" ]]; then
  wait_port="${WAIT_FOR_PORT:-3306}"
  echo "Waiting for ${WAIT_FOR_HOST}:${wait_port}..."
  until nc -z "$WAIT_FOR_HOST" "$wait_port"; do
    sleep 1
  done
fi

mkdir -p artifacts/audio artifacts/vectorstore/medical_faiss_v4

exec "$@"
