@echo off
cd /d "%~dp0..\.."
call .venv\Scripts\activate.bat
echo ========================================
echo   Dance Anonymizer
echo   Open http://localhost:8002
echo   Press Ctrl+C to stop
echo ========================================
uvicorn api:app --host 0.0.0.0 --port 8002
pause
