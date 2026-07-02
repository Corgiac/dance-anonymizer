#!/bin/bash
# 舞蹈视频智能打码 - 一键安装脚本
set -e

echo "========================================"
echo "  舞蹈视频智能打码 - 环境安装"
echo "========================================"

# 检查 Python 版本
echo "[1/5] 检查 Python..."
python3 --version || { echo "请先安装 Python 3.10+"; exit 1; }

# 创建虚拟环境
echo "[2/5] 创建虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 安装 Python 依赖
echo "[3/5] 安装 Python 依赖..."
pip install -r requirements.txt -q



# 合并 SAM2 模型分卷
if [ ! -f "sam2_hiera_tiny.pt" ] && [ -f "sam2_hiera_tiny.pt.part_aa" ]; then
    echo "合并模型文件..."
    cat sam2_hiera_tiny.pt.part_* > sam2_hiera_tiny.pt
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo ""
echo "  uvicorn api:app --host 0.0.0.0 --port 8002"
echo "========================================"
