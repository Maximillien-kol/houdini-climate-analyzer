@echo off
REM AgriShield AI — Windows Quick Start Script
REM Runs both the Python ML service and Node.js API gateway.

TITLE AgriShield AI

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║  AgriShield AI — Food Insecurity ^& Climate AI       ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM ── Step 1: Check Python ─────────────────────────────────────────────────────
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    pause & exit /b 1
)

REM ── Step 2: Install Python dependencies ──────────────────────────────────────
echo [1/5] Installing Python dependencies ...
cd /d "%~dp0python_ml"
pip install -r requirements.txt --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip install failed.
    pause & exit /b 1
)

REM ── Step 3: Train models (skip if artifacts already exist) ────────────────────
IF NOT EXIST "artifacts\rain_prediction_rf.pkl" (
    echo [2/5] Training AI models (first-time setup — this takes a few minutes) ...
    python main.py train
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Model training failed.
        pause & exit /b 1
    )
) ELSE (
    echo [2/5] Trained models found — skipping training.
)

REM ── Step 4: Start Python ML service in background ─────────────────────────────
echo [3/5] Starting Python ML service on port 5001 ...
start "AgriShield Python ML" cmd /k "cd /d "%~dp0python_ml" && python main.py serve"
timeout /t 4 /nobreak >nul

REM ── Step 5: Install Node.js dependencies ─────────────────────────────────────
echo [4/5] Installing Node.js dependencies ...
cd /d "%~dp0nodejs_api"
IF NOT EXIST ".env" (
    copy .env.example .env >nul
    echo      Copied .env.example → .env
)
npm install --silent
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm install failed. Make sure Node.js 18+ is installed.
    pause & exit /b 1
)

REM ── Step 6: Start Node.js API ─────────────────────────────────────────────────
echo [5/5] Starting Node.js API gateway on port 3000 ...
start "AgriShield Node.js API" cmd /k "cd /d "%~dp0nodejs_api" && npm start"
timeout /t 3 /nobreak >nul

echo.
echo ✓ AgriShield AI is running!
echo.
echo   Python ML Service : http://localhost:5001/health
echo   Node.js API        : http://localhost:3000/api/health
echo   Quick test         : http://localhost:3000/api/data/simulate
echo.
echo   Full prediction test:
echo   POST http://localhost:3000/api/predict/full
echo   (see README.md for request body)
echo.
pause
