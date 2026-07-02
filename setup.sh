#!/bin/bash
# 舞蹈视频智能打码 - 一键安装脚本
set -e

echo "========================================"
echo "  舞蹈视频智能打码 - 环境安装"
echo "========================================"

echo "[1/4] 检查 Python..."
python3 --version || { echo "请先安装 Python 3.10+"; exit 1; }

echo "[2/4] 创建虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "[3/4] 合并模型文件..."
if [ ! -f "sam2_hiera_tiny.pt" ] && [ -f "sam2_hiera_tiny.pt.part_aa" ]; then
    cat sam2_hiera_tiny.pt.part_* > sam2_hiera_tiny.pt
    echo "模型合并完成"
fi

echo "[4/4] 安装 Python 依赖..."
pip install -r requirements.txt -q

echo ""
echo "========================================"
echo "  安装完成！"
echo "  uvicorn api:app --host 0.0.0.0 --port 8002"
echo "========================================"
