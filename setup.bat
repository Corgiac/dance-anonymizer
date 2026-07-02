@echo off
chcp 65001 >nul
echo ========================================
echo   舞蹈视频智能打码 - 一键安装 (Windows)
echo ========================================
echo.

echo [1/4] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 未找到 Python，请先安装 Python 3.10+
    echo   下载地址: https://www.python.org/downloads/
    echo   安装时务必勾选 "Add Python to PATH"
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] 创建虚拟环境...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo.

echo [3/4] 合并模型文件...
if not exist "sam2_hiera_tiny.pt" (
    if exist "sam2_hiera_tiny.pt.part_aa" (
        copy /b sam2_hiera_tiny.pt.part_aa + sam2_hiera_tiny.pt.part_ab sam2_hiera_tiny.pt
        echo 模型合并完成
    )
)
echo.

echo [4/4] 安装 Python 依赖（可能需要几分钟）...
pip install -r requirements.txt -q
echo.

echo ========================================
echo   安装完成！
echo.
echo   启动方式：
echo   1. 双击 run.bat
echo   2. 浏览器打开 http://localhost:8002
echo ========================================
pause
