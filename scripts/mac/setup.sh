#!/bin/bash
# DanceAnon - 一键安装脚本
set -e
cd "$(dirname "$0")/../.."

echo "========================================"
echo "  DanceAnon - 环境安装"
echo "========================================"

echo "[1/3] 检查 Python..."
python3 --version || { echo "请先安装 Python 3.10+：https://www.python.org/downloads/"; exit 1; }
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "Python 3.$PY_MINOR 版本过低，需要 3.10+"
    echo "请升级：brew install python@3.11 或访问 python.org"
    exit 1
fi

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
    brew install ffmpeg -q 2>/dev/null && echo "  ffmpeg 安装完成 ✓" || echo "  ffmpeg 安装失败"
else
    echo "  下载静态编译版 ffmpeg..."
    FFMPEG_URL="https://evermeet.cx/ffmpeg/getrelease/zip"
    curl -L -o /tmp/ffmpeg.zip "$FFMPEG_URL" 2>/dev/null
    if [ -f /tmp/ffmpeg.zip ]; then
        unzip -o /tmp/ffmpeg.zip -d /tmp/ffmpeg_extract >/dev/null 2>&1
        mkdir -p "$HOME/.local/bin" 2>/dev/null
        cp /tmp/ffmpeg_extract/ffmpeg "$HOME/.local/bin/" 2>/dev/null
        chmod +x "$HOME/.local/bin/ffmpeg" 2>/dev/null
        rm -rf /tmp/ffmpeg.zip /tmp/ffmpeg_extract
        export PATH="$HOME/.local/bin:$PATH"
        echo "  ffmpeg 安装到 $HOME/.local/bin ✓"
    else
        echo "  下载失败。可手动安装: brew install ffmpeg"
    fi
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "  启动: bash scripts/mac/run.sh"
echo "  浏览器打开 http://localhost:8002"
echo "========================================"
