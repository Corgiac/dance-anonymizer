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

rem 检查 Python 版本 >= 3.10
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Python %PYVER%
for /f "tokens=2 delims=." %%a in ("%PYVER%") do set PYMINOR=%%a
if %PYMINOR% LSS 10 (
    echo.
    echo ERROR: Python %PYVER% is too old. Python 3.10 or newer required.
    echo Please download Python 3.10+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo.

echo [2/3] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo.

echo [3/3] Installing Python packages (may take a few minutes)...
echo   If you are in China, close VPN/proxy and try again if this step fails.
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo.
    echo   Retrying with Tsinghua mirror...
    pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

echo [Optional] Installing ffmpeg for audio support...
where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    echo   ffmpeg already installed
) else (
    echo   Downloading ffmpeg...
    curl -L -o "%TEMP%\ffmpeg-release-essentials.zip" "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" 2>nul
    if exist "%TEMP%\ffmpeg-release-essentials.zip" (
        mkdir "%USERPROFILE%\ffmpeg" 2>nul
        tar -xf "%TEMP%\ffmpeg-release-essentials.zip" -C "%USERPROFILE%\ffmpeg" --strip-components=1 2>nul
        echo   ffmpeg installed to %USERPROFILE%\ffmpeg
        echo   Adding to PATH for this session...
        set "PATH=%USERPROFILE%\ffmpeg\bin;%PATH%"
    ) else (
        echo   Download failed. Get ffmpeg from https://ffmpeg.org for audio support.
    )
)
echo.
echo ========================================
echo   Setup complete!
echo.
echo   To start: double-click run.bat
echo   Then open http://localhost:8002
echo ========================================
pause
