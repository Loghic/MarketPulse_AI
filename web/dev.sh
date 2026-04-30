#!/bin/bash
# dev.sh – Start FastAPI backend + React frontend for development.
#
# Usage:
#   ./web/dev.sh
#
# Backend: http://localhost:8000 (API + Swagger docs at /docs)
# Frontend: http://localhost:5173 (proxies /api to backend)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════╗"
echo "║  MarketPulse AI — Dev Server         ║"
echo "╠══════════════════════════════════════╣"
echo "║  Backend:  http://localhost:8000     ║"
echo "║  API docs: http://localhost:8000/docs║"
echo "║  Frontend: http://localhost:5173     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Install web deps if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing FastAPI..."
    uv pip install -e ".[web]"
fi

# Install frontend deps if needed
if [ ! -d "web/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd web/frontend && npm install && cd "$PROJECT_ROOT"
fi

# Start backend in background
echo "Starting backend..."
uv run uvicorn web.backend.app:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend..."
cd web/frontend && npm run dev &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

echo ""
echo "Press Ctrl+C to stop both servers."
wait
