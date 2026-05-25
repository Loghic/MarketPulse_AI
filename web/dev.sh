#!/bin/bash
# dev.sh – Start FastAPI backend + React frontend for development.
#
# Usage:
#   ./web/dev.sh                # default ports 8000 / 5173
#   BACKEND_PORT=8001 ./web/dev.sh
#
# Backend: http://localhost:8000 (API + Swagger docs at /docs)
# Frontend: http://localhost:5173 (proxies /api to backend)
#
# Robust against leftover processes from a previous interrupted run —
# the trap-based cleanup at the bottom only fires inside this shell, so
# orphans from a hard-killed previous invocation (e.g. CTRL+C during a
# long backtest, leaked semaphores) still own the ports. We detect that
# and reclaim them before starting.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════╗"
echo "║  MarketPulse AI — Dev Server         ║"
echo "╠══════════════════════════════════════╣"
printf "║  Backend:  http://localhost:%-9s║\n" "$BACKEND_PORT"
printf "║  API docs: http://localhost:%s/docs║\n" "$BACKEND_PORT"
printf "║  Frontend: http://localhost:%-9s║\n" "$FRONTEND_PORT"
echo "╚══════════════════════════════════════╝"
echo ""

# Free up the ports if a previous run left something behind.
# (`lsof -t` prints the PIDs holding the port; harmless if empty.)
reclaim_port() {
    local port="$1" label="$2"
    if command -v lsof >/dev/null 2>&1; then
        local pids
        pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "  ⚠  Port $port ($label) is in use by PID(s): $pids — reclaiming..."
            # SIGTERM first, then SIGKILL after a beat. This handles the
            # "previous uvicorn didn't exit cleanly" case the user hit.
            kill $pids 2>/dev/null || true
            sleep 1
            pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                kill -9 $pids 2>/dev/null || true
            fi
        fi
    fi
}
reclaim_port "$BACKEND_PORT" "backend"
reclaim_port "$FRONTEND_PORT" "frontend"

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
echo "Starting backend on port $BACKEND_PORT..."
uv run uvicorn web.backend.app:app --reload --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on port $FRONTEND_PORT..."
cd web/frontend && npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Cleanup on exit — kill our PIDs AND anything left on the ports just in
# case (e.g. children spawned by the reloader).
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    sleep 0.5
    # Belt-and-suspenders: free the ports for the next run.
    if command -v lsof >/dev/null 2>&1; then
        local leftover
        leftover=$(lsof -ti tcp:"$BACKEND_PORT" -ti tcp:"$FRONTEND_PORT" 2>/dev/null || true)
        if [ -n "$leftover" ]; then
            kill -9 $leftover 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

echo ""
echo "Press Ctrl+C to stop both servers."
wait
