#!/bin/bash
# Pull latest pipeline code from GitHub before starting the handler.
# This means pushing to GitHub is all that's needed to deploy code changes —
# no Docker rebuild, no rollout.
#
# Heartbeats: every step prints a marker so the RunPod logs always tell us
# exactly which phase is taking time, even before the Python handler logs.
set -e
echo "[start] === $(date -u +%H:%M:%S) start.sh begin ===" >&2

echo "[start] === $(date -u +%H:%M:%S) git pull start ===" >&2
cd /workspace/skinmapper-pipeline
# 30s timeout + retries so we never hang on a flaky GitHub connection
timeout 30 git pull origin main || {
  echo "[start] === git pull failed/timed out — continuing with cached code ===" >&2
}
echo "[start] === $(date -u +%H:%M:%S) git pull done, HEAD=$(git rev-parse --short HEAD) ===" >&2

echo "[start] === $(date -u +%H:%M:%S) launching python handler ===" >&2
exec python -u /workspace/skinmapper-pipeline/handler.py
