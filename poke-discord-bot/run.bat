@echo off
setlocal EnableDelayedExpansion

REM === Load .env file ===
if not exist .env (
    echo ERROR: .env file not found!
    pause
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" (
        set %%A=%%B
    )
)

REM === Validate required vars ===
if "%DISCORD_TOKEN%"=="" (
    echo ERROR: DISCORD_TOKEN is not set
    pause
    exit /b 1
)

if "%API_BASE%"=="" (
    echo ERROR: API_BASE is not set
    pause
    exit /b 1
)

docker build -t pokemmo-ocr-discord-bot .

docker rm -f %BOT_NAME% >nul 2>&1

docker run -d ^
  --name %BOT_NAME% ^
  --restart unless-stopped ^
  -e DISCORD_TOKEN=%DISCORD_TOKEN% ^
  -e API_BASE=%API_BASE% ^ ^
  pokemmo-ocr-discord-bot


echo.
echo Bot started successfully.
echo API: %API_BASE%
pause
