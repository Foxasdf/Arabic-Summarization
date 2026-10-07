#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "        ملخص الاجتماعات العربية (Arabic Meeting Summarizer)"
echo "========================================================"
echo ""

# 1. Check Python
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] لم يتم العثور على Python! يرجى تثبيت Python 3.10+ أولاً."
    exit 1
fi

# 2. Setup virtual environment
VENV_DIR="$SCRIPT_DIR/backend/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/4] جاري إنشاء البيئة الافتراضية (venv)..."
    "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR" || {
        echo "[ERROR] فشل إنشاء البيئة الافتراضية."
        echo "على أنظمة Debian/Ubuntu يرجى التأكد من تثبيت حزمة venv:"
        echo "  sudo apt install python3-venv"
        exit 1
    }
fi

source "$VENV_DIR/bin/activate"

# Check if GPU reinstall is requested or if libraries are missing/CPU-only
NEEDS_INSTALL=0
if [ "$1" == "--update" ] || [ "$1" == "--install" ] || [ "$1" == "--reinstall-gpu" ] || [ ! -f "$VENV_DIR/.installed" ]; then
    NEEDS_INSTALL=1
elif ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
    echo "[2/4] تفعيل البيئة وتثبيت المتطلبات (مع تسريع NVIDIA CUDA)..."
    pip install --upgrade pip -q
    
    # Check if nvcc is available to build llama-cpp-python with CUDA if wheel fails
    if command -v nvcc &>/dev/null; then
        export CMAKE_ARGS="-DGGML_CUDA=on"
    fi

    pip install -r "$SCRIPT_DIR/backend/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 -q && touch "$VENV_DIR/.installed"
else
    echo "[2/4] البيئة الافتراضية والمكتبات جاهزة."
fi

# Quick GPU check warning
if command -v nvidia-smi &>/dev/null; then
    python -c "
import torch, llama_cpp
gpu_torch = torch.cuda.is_available()
gpu_llama = llama_cpp.llama_supports_gpu_offload()
if not gpu_torch or not gpu_llama:
    print('\n[تنبيه هام]: كرت الشاشة NVIDIA متوفر بالجهاز ولكن بعض المكتبات تعمل بوضع المعالج المركزي (CPU)!')
    if not gpu_torch: print('  - PyTorch (فصل المتحدثين): يعمل على CPU.')
    if not gpu_llama: print('  - LLaMA (تفريغ الصوت والـ LLM): يعمل على CPU.')
    print('  لإعادة تثبيت المكتبات مع دعم كرت الشاشة، شغّل: ./run.sh --reinstall-gpu\n')
" 2>/dev/null || true
fi

# 3. Ensure ffmpeg is available
echo "[3/4] فحص معالج الصوت (ffmpeg)..."
if ! command -v ffmpeg &>/dev/null; then
    if [ ! -f "$SCRIPT_DIR/backend/bin/ffmpeg" ]; then
        echo "جاري استخراج ffmpeg الثابت تلقائياً..."
        python -c "import sys, imageio_ffmpeg, shutil, pathlib, os; exe = imageio_ffmpeg.get_ffmpeg_exe(); d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True); shutil.copy(exe, d / 'ffmpeg'); os.chmod(d / 'ffmpeg', 0o755); print('تم استخراج ffmpeg بنجاح.')" "$SCRIPT_DIR/backend/bin"
    fi
    export PATH="$SCRIPT_DIR/backend/bin:$PATH"
fi

# 4. Build frontend if not built
if [ ! -f "$SCRIPT_DIR/frontend/dist/index.html" ]; then
    if command -v npm &>/dev/null; then
        echo "[4/4] جاري تثبيت وبناء الواجهة الأمامية (Frontend)..."
        cd "$SCRIPT_DIR/frontend"
        if [ ! -d "node_modules" ]; then
            npm install --silent
        fi
        npm run build || echo "[WARNING] فشل بناء الواجهة، سيتم تشغيل واجهة الـ API."
        cd "$SCRIPT_DIR"
    else
        echo "[WARNING] Node.js/npm غير متوفر لبناء الواجهة، سيتم تشغيل واجهة الـ API."
    fi
else
    echo "[4/4] ملفات الواجهة الأمامية جاهزة."
fi

echo ""
echo "========================================================"
echo "  جاري تشغيل الخادم..."
echo "========================================================"
echo ""

cd "$SCRIPT_DIR/backend"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
