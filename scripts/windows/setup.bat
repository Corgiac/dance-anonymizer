@echo off
cd /d "%~dp0..\.."
echo ========================================
echo   Dance Anonymizer - Setup (Windows)
echo ========================================
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python not found.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo.

echo [3/4] Merging model files...
if not exist "sam2_hiera_tiny.pt" (
    if exist "sam2_hiera_tiny.pt.part_aa" (
        copy /b sam2_hiera_tiny.pt.part_aa + sam2_hiera_tiny.pt.part_ab sam2_hiera_tiny.pt
        echo Model merged successfully
    )
)
echo.

echo [4/4] Installing Python packages (may take a few minutes)...
pip install -r requirements.txt -q
echo.

echo ========================================
echo   Setup complete!
echo.
echo   To start: double-click run.bat
echo   Then open http://localhost:8002
echo ========================================
pause
