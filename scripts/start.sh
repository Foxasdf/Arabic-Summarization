#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "      ملخص الاجتماعات العربية"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 غير مثبت! يرجى تثبيت Python 3.10+"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js غير مثبت! يرجى تثبيت Node.js 18+"
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] ffmpeg غير مثبت! يرجى تثبيته لمعالجة الصوت"
    echo "          Ubuntu: sudo apt install ffmpeg"
    echo "          macOS:  brew install ffmpeg"
    echo ""
fi

# Check models
echo "فحص النماذج..."
if [ ! -f "$PROJECT_DIR/models/Falcon-H1-7B-Instruct-IQ4_XS.gguf" ]; then
    echo "[WARNING] نموذج Falcon غير موجود! شغّل: python3 scripts/download_models.py"
fi
if [ ! -f "$PROJECT_DIR/models/Audar-ASR-V1-Turbo-Q8_0.gguf" ]; then
    echo "[WARNING] نموذج Audar ASR غير موجود! شغّل: python3 scripts/download_models.py"
fi
if [ ! -f "$PROJECT_DIR/models/mmproj-Audar-ASR-V1-Turbo.gguf" ]; then
    echo "[WARNING] ملف mmproj غير موجود! شغّل: python3 scripts/download_models.py"
fi
echo ""

# Create virtual environment if needed
if [ ! -d "$PROJECT_DIR/backend/.venv" ]; then
    echo "إنشاء بيئة Python الافتراضية..."
    python3 -m venv "$PROJECT_DIR/backend/.venv"
fi

# Activate virtual environment
source "$PROJECT_DIR/backend/.venv/bin/activate"

# Install backend dependencies if needed
if [ ! -f "$PROJECT_DIR/backend/.venv/.installed" ] && ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "تثبيت مكتبات Python..."
    pip install -r "$PROJECT_DIR/backend/requirements.txt" -q && touch "$PROJECT_DIR/backend/.venv/.installed"
else
    [ ! -f "$PROJECT_DIR/backend/.venv/.installed" ] && touch "$PROJECT_DIR/backend/.venv/.installed"
fi

# Install frontend dependencies
echo "تثبيت مكتبات Frontend..."
cd "$PROJECT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi

# Get local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")

echo ""
echo "============================================"
echo "  بدء تشغيل التطبيق..."
echo "============================================"
echo ""

# Trap to clean up background processes
cleanup() {
    echo ""
    echo "إيقاف التطبيق..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
cd "$PROJECT_DIR/backend"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
cd "$PROJECT_DIR/frontend"
npx vite --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "  التطبيق يعمل!"
echo "============================================"
echo ""
echo "  محلياً:      http://localhost:3000"
echo "  على الشبكة:  http://$LOCAL_IP:3000"
echo ""
echo "  للإيقاف: اضغط Ctrl+C"
echo "============================================"
echo ""

# Wait for background processes
wait
