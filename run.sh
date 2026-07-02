#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "========================================"
echo "  舞蹈视频智能打码"
echo "  浏览器打开 http://localhost:8002"
echo "  按 Ctrl+C 可以停止"
echo "========================================"
uvicorn api:app --host 0.0.0.0 --port 8002
