import sys
import os
import warnings

# On Windows: suppress native OS modal error dialogs for any missing DLL procedure entry points
if sys.platform.startswith("win"):
    try:
        import ctypes
        # SEM_FAILCRITICALERRORS (0x0001) | SEM_NOGPFAULTERRORBOX (0x0002)
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    except Exception:
        pass

# Block broken/incompatible torchcodec from triggering OS DLL popups or warnings
sys.modules["torchcodec"] = None
warnings.filterwarnings("ignore", category=UserWarning, module=r".*pyannote.*")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("SHIM_MCCOMPAT", "0x000000001")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import logging
from contextlib import asynccontextmanager

from .config import settings
from .services.storage import init_db
from .api.routes import router

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter noisy polling requests (e.g. GET /api/meetings) from Uvicorn access logs
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if 'GET /api/meetings' in msg or 'GET /api/health' in msg:
            return False
        return True

logging.getLogger('uvicorn.access').addFilter(EndpointFilter())

def print_startup_banner():
    import socket
    from .services.hardware import get_hardware_profile
    prof = get_hardware_profile()

    local_ip = ""
    try:
        hostname = socket.gethostname()
        ips = [
            ip for ip in socket.gethostbyname_ex(hostname)[2]
            if not ip.startswith("127.") and not ip.startswith("169.254.")
        ]
        lan = [ip for ip in ips if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.")]
        local_ip = lan[0] if lan else (ips[0] if ips else "")
    except Exception:
        pass

    port = settings.PORT
    gpu_status = f"{prof.gpu_name} (مُفعّل بالكامل - VRAM: {prof.vram_free_gb:.1f}GB/{prof.vram_total_gb:.1f}GB)" if prof.gpu_name and prof.use_gpu else (prof.gpu_name or "المعالج المركزي CPU فقط")
    
    audar_ok = (settings.MODELS_DIR / settings.ASR_GGUF_FILENAME).exists() or (settings.MODELS_DIR / settings.COHERE_ASR_FILENAME).exists()
    falcon_ok = (settings.MODELS_DIR / settings.LLM_GGUF_FILENAME).exists()
    models_status = "جاهزة بالكامل ✅" if (audar_ok and falcon_ok) else "⚠️ بعض النماذج غير محملة (شغّل: python scripts/download_models.py)"

    workers_count = prof.recommended_asr_workers
    workers_str = f"×{workers_count} مقاطع بالتوازي (Parallel Speedup ×{workers_count})" if workers_count > 1 else "1 مقطع بالتسلسل"

    banner_lines = [
        "",
        "=" * 64,
        "   ملخص الاجتماعات العربية (Arabic Meeting Summarizer)",
        "=" * 64,
        f"   المعالج الرسومي (GPU):         {gpu_status}",
        f"   التفريغ المتوازي (Parallel ASR): {workers_str}",
        f"   حالة النماذج (Models):        {models_status}",
        f"   الرابط المحلي (Local):         http://localhost:{port}",
        f"   الرابط البديل (Alternative):   http://127.0.0.1:{port}",
    ]
    if local_ip:
        banner_lines.append(f"   الشبكة المحلية (Network):     http://{local_ip}:{port}")
    banner_lines.extend([
        f"   توثيق الـ API (Docs):          http://localhost:{port}/docs",
        "   [تنبيه]: لا تفتح 0.0.0.0 في المتصفح - وأوقف إضافة الـ VPN إن وجدت",
        "   للإيقاف (Stop):                اضغط Ctrl+C",
        "=" * 64,
        "",
    ])
    try:
        print("\n".join(banner_lines), flush=True)
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application...")
    await init_db()
    print_startup_banner()
    yield
    logger.info("Shutting down application...")

app = FastAPI(title="Arabic Meeting Summarizer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Mount built frontend files if present (for single-server local deployment)
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")

serve_dir = None
if os.path.exists(frontend_dist) and os.path.isfile(os.path.join(frontend_dist, "index.html")):
    serve_dir = frontend_dist
elif os.path.exists(static_dir):
    serve_dir = static_dir

if serve_dir:
    logger.info(f"Serving frontend static files from: {serve_dir}")
    app.mount("/", StaticFiles(directory=serve_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
