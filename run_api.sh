#!/bin/bash
echo "==================================================="
echo "  Starting Multi-Agent RAG Flask API (macOS/Linux)"
echo "==================================================="
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python3 api_server.py || python api_server.py
