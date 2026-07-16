@echo off
cd /d "%~dp0..\.."
echo ========================================
echo   DanceAnon - Setup (Windows)
echo ========================================
echo.

echo [1/3] Checking Python...
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

echo [2/3] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo.

echo [3/3] Installing Python packages (may take a few minutes)...
pip install -r requirements.txt -q
echo.

echo ========================================
echo   Setup complete!
echo.
echo   To start: double-click run.bat
echo   Then open http://localhost:8002
echo.
echo   Optional: install ffmpeg for audio support
echo   (download from https://ffmpeg.org)
echo ========================================
pause
