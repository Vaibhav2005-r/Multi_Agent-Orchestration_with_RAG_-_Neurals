@echo off
setlocal
echo ===================================================
echo   Starting Multi-Agent RAG Flask API (Windows)
echo ===================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python api_server.py
pause
