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
echo "[可选] 安装 ffmpeg (用于音频)..."

if command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg 已安装 ✓"
elif command -v brew &>/dev/null; then
    echo "  正在通过 Homebrew 安装 ffmpeg..."
    brew install ffmpeg -q 2>/dev/null && echo "  ffmpeg 安装完成 ✓" || echo "  ffmpeg 安装失败，可稍后手动执行: brew install ffmpeg"
else
    echo "  未检测到 ffmpeg，视频生成后将无音频"
    echo "  如需音频，请先安装 Homebrew (https://brew.sh)"
    echo "  然后执行: brew install ffmpeg"
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "  启动: bash scripts/mac/run.sh"
echo "  浏览器打开 http://localhost:8002"
echo "========================================"
