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

# 安装 SAM2
echo "[4/5] 安装 SAM2（首帧抠图精修）..."
pip install git+https://github.com/facebookresearch/sam2.git -q 2>/dev/null || echo "SAM2 安装失败，请手动安装: pip install git+https://github.com/facebookresearch/sam2.git"

# 安装 Cutie
echo "[5/5] 安装 Cutie（全片追踪引擎）..."
pip install -e vendor/Cutie -q 2>/dev/null || echo "Cutie 安装失败，请手动安装: pip install -e vendor/Cutie"

echo ""
echo "========================================"
echo "  安装完成！"
echo ""
echo "  uvicorn api:app --host 0.0.0.0 --port 8002"
echo "========================================"
