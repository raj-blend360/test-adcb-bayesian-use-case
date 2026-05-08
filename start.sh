#!/usr/bin/env bash
# Start both the FastAPI backend and the React frontend

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Backend
echo "Starting FastAPI backend on :8000 ..."
cd "$ROOT"
pip install -q -r api/requirements.txt
uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Frontend
echo "Starting React frontend on :5173 ..."
cd "$ROOT/webapp"
npm install --silent
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
