@echo off
echo Starting Backend API Server...
start "Multi-Agent RAG API" cmd /k "set PYTHONUTF8=1 && python api_server.py"

echo Starting Frontend Dev Server...
cd frontend
start "Multi-Agent RAG Frontend" cmd /k "npm run dev"
