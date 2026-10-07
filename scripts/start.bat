@echo off
chcp 65001 >nul 2>&1
title ملخص الاجتماعات

echo ============================================
echo       ملخص الاجتماعات العربية
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python غير مثبت! يرجى تثبيت Python 3.10+ أولاً
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js غير مثبت! يرجى تثبيت Node.js 18+ أولاً
    pause
    exit /b 1
)

REM Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg غير مثبت! يرجى تثبيته لمعالجة الصوت
    echo           https://ffmpeg.org/download.html
    echo.
)

REM Check models
echo فحص النماذج...
if not exist "%~dp0..\models\Falcon-H1-7B-Instruct-IQ4_XS.gguf" (
    echo [WARNING] نموذج Falcon غير موجود! شغّل: python scripts\download_models.py
)
if not exist "%~dp0..\models\Audar-ASR-V1-Turbo-Q8_0.gguf" (
    echo [WARNING] نموذج Audar ASR غير موجود! شغّل: python scripts\download_models.py
)
if not exist "%~dp0..\models\mmproj-Audar-ASR-V1-Turbo.gguf" (
    echo [WARNING] ملف mmproj غير موجود! شغّل: python scripts\download_models.py
)
echo.

REM Create virtual environment if needed
if not exist "%~dp0..\backend\.venv" (
    echo إنشاء بيئة Python الافتراضية...
    python -m venv "%~dp0..\backend\.venv"
)

REM Activate virtual environment
call "%~dp0..\backend\.venv\Scripts\activate.bat"

REM Install backend dependencies if needed
if exist "%~dp0..\backend\.venv\.installed" goto :deps_ready
python -c "import fastapi, uvicorn" >nul 2>&1
if not errorlevel 1 (
    echo. > "%~dp0..\backend\.venv\.installed"
    goto :deps_ready
)

echo تثبيت مكتبات Python...
pip install -r "%~dp0..\backend\requirements.txt" -q
if not errorlevel 1 echo. > "%~dp0..\backend\.venv\.installed"

:deps_ready

REM Install frontend dependencies
echo تثبيت مكتبات Frontend...
cd /d "%~dp0..\frontend"
if not exist "node_modules" (
    call npm install
)

REM Get local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo ============================================
echo   بدء تشغيل التطبيق...
echo ============================================
echo.

REM Start backend
cd /d "%~dp0..\backend"
start "Backend Server" /min cmd /c "call .venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
cd /d "%~dp0..\frontend"
start "Frontend Server" /min cmd /c "npx vite --host 0.0.0.0 --port 3000"

echo.
echo ============================================
echo   التطبيق يعمل!
echo ============================================
echo.
echo   محلياً:      http://localhost:3000
echo   على الشبكة:  http://%LOCAL_IP%:3000
echo.
echo   للإيقاف: أغلق هذه النافذة
echo ============================================
echo.
pause
