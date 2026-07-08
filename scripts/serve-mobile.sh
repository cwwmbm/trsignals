#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Building frontend…"
(cd frontend && npm run build)

echo "Starting unified server on 0.0.0.0:8000"
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
