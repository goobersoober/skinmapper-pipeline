#!/bin/bash
# Pull latest pipeline code from GitHub before starting the handler.
# This means pushing to GitHub is all that's needed to deploy code changes —
# no Docker rebuild, no rollout.
set -e
echo "[start] pulling latest pipeline code..."
cd /workspace/skinmapper-pipeline
git pull origin main
echo "[start] launching handler..."
exec python -u /workspace/skinmapper-pipeline/handler.py
