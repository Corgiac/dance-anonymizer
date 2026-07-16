#!/bin/bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
echo "========================================"
echo "  DanceAnon"
echo "  浏览器打开 http://localhost:8002"
echo "  按 Ctrl+C 停止"
echo "========================================"
python -m uvicorn api:app --host 0.0.0.0 --port 8002
