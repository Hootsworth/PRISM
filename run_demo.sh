#!/usr/bin/env bash
# PRISM Smartphone ISP Demo Launcher
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================="
echo "       PRISM — Smartphone ISP Computational Engine Demo          "
echo "================================================================="
echo "Starting unified PRISM Web Application..."
echo "Backend: FastAPI (Apple Silicon MPS / CUDA / CPU)"
echo "Frontend: React + TypeScript (Vite bundle)"
echo "Access URL: http://localhost:8000"
echo "Press Ctrl+C to terminate."
echo "-----------------------------------------------------------------"

export PYTHONPATH="$DIR/src:$DIR"
exec "$DIR/.venv/bin/python3" -m uvicorn demo.backend.api:app --host 0.0.0.0 --port 8000 --reload
