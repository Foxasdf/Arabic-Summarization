@echo off
chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CUDA_VISIBLE_DEVICES=0"
set "SHIM_MCCOMPAT=0x000000001"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
title ملخص الاجتماعات العربية


echo ========================================================
echo   ملخص الاجتماعات العربية - Arabic Meeting Summarizer
echo ========================================================
echo.

:: 1. فحص بايثون
python --version >nul 2>&1
if not errorlevel 1 goto :python_ok
echo [ERROR] لم يتم العثور على Python. يرجى تثبيت Python 3.10 أو أحدث.
echo يمكنك تحميله من: https://www.python.org/downloads/
pause
exit /b 1

:python_ok
set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%backend\.venv"

:: 2. فحص وإنشاء وتفعيل البيئة الافتراضية
if exist "%VENV_DIR%\Scripts\activate.bat" goto :venv_exists
echo [1/4] جاري إنشاء البيئة الافتراضية لأول مرة...
python -m venv --system-site-packages "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] فشل إنشاء البيئة الافتراضية.
    pause
    exit /b 1
)

:venv_exists
call "%VENV_DIR%\Scripts\activate.bat"

:: فحص هل المكتبات مثبتة مسبقاً لتجاوز التحقق البطيء عبر الإنترنت
if "%1"=="--update" goto :install_deps
if "%1"=="--install" goto :install_deps
if exist "%VENV_DIR%\.installed" goto :deps_ok

python -c "import fastapi, uvicorn" >nul 2>&1
if not errorlevel 1 (
    echo. > "%VENV_DIR%\.installed"
    goto :deps_ok
)

:install_deps
echo [2/4] جاري تثبيت وتحديث متطلبات التطبيق...
python -m pip install --upgrade pip -q
pip install -r "%PROJECT_ROOT%backend\requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 -q
if not errorlevel 1 (
    echo. > "%VENV_DIR%\.installed"
) else (
    echo [WARNING] تعذر تحديث بعض الحزم عبر الإنترنت، سيتم المتابعة بالحزم المتوفرة حالياً.
)

:deps_ok
echo [2/4] البيئة الافتراضية والمكتبات جاهزة.

:: فحص النماذج الأساسية
if exist "%PROJECT_ROOT%models\Audar-ASR-V1-Turbo-Q8_0.gguf" if exist "%PROJECT_ROOT%models\Falcon-H1-7B-Instruct-IQ4_XS.gguf" goto :models_ok
echo [!] بعض النماذج الأساسية غير متوفرة في مجلد models\.
echo جاري تشغيل سكربت تحميل النماذج تلقائياً...
python "%PROJECT_ROOT%scripts\download_models.py"
:models_ok

:: 3. فحص معالج الصوت ffmpeg
echo [3/4] فحص معالج الصوت ffmpeg...
where ffmpeg >nul 2>&1
if not errorlevel 1 goto :ffmpeg_ok

if exist "%PROJECT_ROOT%backend\bin\ffmpeg.exe" goto :ffmpeg_path_set
echo جاري استخراج ffmpeg الثابت تلقائياً...
python -c "import sys, imageio_ffmpeg, shutil, pathlib; exe = imageio_ffmpeg.get_ffmpeg_exe(); d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True); shutil.copy(exe, d / 'ffmpeg.exe'); print('تم استخراج ffmpeg بنجاح.')" "%PROJECT_ROOT%backend\bin"

:ffmpeg_path_set
set "PATH=%PROJECT_ROOT%backend\bin;%PATH%"

:ffmpeg_ok

:: 4. بناء الواجهة الأمامية إن لم تكن مبنية مسبقاً
if exist "%PROJECT_ROOT%frontend\dist\index.html" goto :frontend_ready

where npm >nul 2>&1
if not errorlevel 1 goto :build_frontend
echo [WARNING] Node.js/npm غير متوفر لبناء الواجهة، سيتم تشغيل واجهة الـ API.
goto :frontend_done

:build_frontend
echo [4/4] جاري تثبيت وبناء الواجهة الأمامية...
cd /d "%PROJECT_ROOT%frontend"
if not exist "node_modules" call npm install --silent
call npm run build
cd /d "%PROJECT_ROOT%"
goto :frontend_done

:frontend_ready
echo [4/4] ملفات الواجهة الأمامية جاهزة.

:frontend_done

:: 5. تشغيل خادم التطبيق
echo.
echo ========================================================
echo   جاري تشغيل الخادم على المنفذ 3011...
echo   افتح المتصفح على: http://localhost:3011
echo   أو الرابط البديل:  http://127.0.0.1:3011
echo   (تنبيه: لا تفتح 0.0.0.0 - وأوقف إضافة الـ VPN إن وجدت)
echo ========================================================
echo.

:: فتح المتصفح تلقائياً بمجرد بدء السيرفر
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:3011"

cd /d "%PROJECT_ROOT%backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 3011 --timeout-keep-alive 3600
pause

