@echo off
title Pokemon OCR API

echo ===============================
echo Starting Pokemon OCR API
echo ===============================

REM Move to this script's directory
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Safety check
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Run API
echo.
echo API running at http://localhost:8000
echo Press CTRL+C to stop
echo.

uvicorn app:app --host 0.0.0.0 --port 8000

echo.
echo ===============================
echo API stopped
echo ===============================
pause
