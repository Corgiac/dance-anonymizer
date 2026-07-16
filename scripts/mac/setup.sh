#!/bin/bash
# DanceAnon - 一键安装脚本
set -e
cd "$(dirname "$0")/../.."

echo "========================================"
echo "  DanceAnon - 环境安装"
echo "========================================"

echo "[1/3] 检查 Python..."
python3 --version || { echo "请先安装 Python 3.10+"; exit 1; }

echo "[2/3] 创建虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "[3/3] 安装 Python 依赖..."
pip install -r requirements.txt -q

echo ""
echo "========================================"
echo "  安装完成！"
echo "  启动: bash scripts/mac/run.sh"
echo "  浏览器打开 http://localhost:8002"
echo ""
echo "  可选: brew install ffmpeg (用于音频合成)"
echo "========================================"
