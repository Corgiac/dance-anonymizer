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

# 克隆 Cutie
echo "[5/5] 克隆 Cutie（全片追踪引擎）..."
if [ ! -d "vendor/Cutie" ]; then
    git clone https://github.com/hkchengrex/Cutie.git vendor/Cutie --depth 1
    pip install -e vendor/Cutie -q 2>/dev/null || echo "Cutie 安装失败，请手动安装: pip install -e vendor/Cutie"
else
    echo "  vendor/Cutie 已存在，跳过"
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo ""
echo "  还需要手动下载两个模型文件（约 175MB）："
echo "  1. yolo11s-seg.pt"
echo "     https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt"
echo "  2. sam2_hiera_tiny.pt"
echo "     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_tiny.pt"
echo ""
echo "  下载后放到项目根目录，然后运行："
echo "  uvicorn api:app --host 0.0.0.0 --port 8002"
echo "========================================"
