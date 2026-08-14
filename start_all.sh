#!/bin/bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "Starting Backend API Server in background..."
(python3 api_server.py || python api_server.py) &
BACKEND_PID=$!

echo "Starting Frontend Dev Server..."
cd frontend
npm run dev

kill $BACKEND_PID
