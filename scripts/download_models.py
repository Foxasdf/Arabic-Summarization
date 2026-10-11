"""Download required AI models from Hugging Face with robust auto-resume and infinite retry."""

import sys
import os
import time
import shutil
import subprocess
from pathlib import Path

# Fix Windows console encoding for UTF-8 and Arabic/emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODELS_DIR = Path(__file__).parent.parent / "models"

# Default Token provided by user
DEFAULT_HF_TOKEN = "REDACTED_HF_TOKEN"

# Primary Models configured for the project
MODELS = [
    {
        "repo": "audarai/Audar-ASR-V1-Turbo",
        "filename": "mmproj-Audar-ASR-V1-Turbo.gguf",
        "description": "مشفر الصوت التكميلي (Multimodal Audio Projector)",
        "expected_size": 641773856,
        "size_str": "612 MB",
        "url": "https://huggingface.co/audarai/Audar-ASR-V1-Turbo/resolve/main/mmproj-Audar-ASR-V1-Turbo.gguf",
    },
    {
        "repo": "audarai/Audar-ASR-V1-Turbo",
        "filename": "Audar-ASR-V1-Turbo-Q8_0.gguf",
        "description": "نموذج تحويل الصوت إلى نص الأساسي (Audar-ASR-V1-Turbo-Q8_0)",
        "expected_size": 2165034848,
        "size_str": "2.01 GB",
        "url": "https://huggingface.co/audarai/Audar-ASR-V1-Turbo/resolve/main/Audar-ASR-V1-Turbo-Q8_0.gguf",
    },
    {
        "repo": "unsloth/Falcon-H1-7B-Instruct-GGUF",
        "filename": "Falcon-H1-7B-Instruct-IQ4_XS.gguf",
        "description": "نموذج التلخيص العربي الأساسي (Falcon-H1-7B-Instruct-IQ4_XS)",
        "expected_size": 4173140640,
        "size_str": "3.88 GB",
        "url": "https://huggingface.co/unsloth/Falcon-H1-7B-Instruct-GGUF/resolve/main/Falcon-H1-7B-Instruct-IQ4_XS.gguf",
    },
    {
        "repo": "handy-computer/cohere-transcribe-arabic-07-2026-gguf",
        "filename": "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf",
        "description": "نموذج كوهير لتفريغ الصوت واللهجات العربية (Cohere Transcribe Arabic Q5_K_M GGUF)",
        "expected_size": 1770270112,
        "size_str": "1.65 GB",
        "url": "https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q5_K_M.gguf",
    },
]



def check_models():
    """Check which models are available locally."""
    print("=" * 68)
    print("   فحص النماذج الأساسية المتوفرة محلياً في مجلد models/")
    print("=" * 68)
    print()
    all_found = True
    for model in MODELS:
        target = MODELS_DIR / model["filename"]
        if target.exists() and target.stat().st_size >= model["expected_size"]:
            size_mb = target.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            print(f" [مكتمل] ✅ {model['description']}")
            print(f"         الملف: {model['filename']} ({size_str})")
        elif target.exists():
            all_found = False
            cur_mb = target.stat().st_size / (1024 * 1024)
            pct = (target.stat().st_size / model["expected_size"]) * 100
            print(f" [غير مكتمل] ⏳ {model['description']}")
            print(f"         الملف: {model['filename']} (تم تحميل {cur_mb:.1f} MB - {pct:.1f}%)")
        else:
            all_found = False
            print(f" [غير موجود] ❌ {model['description']}")
            print(f"         الملف المطلوب: {model['filename']} ({model['size_str']})")
        print()

    print("=" * 68)
    if all_found:
        print("✅ جميع النماذج الأساسية مكتملة وجاهزة للاستخدام بالكامل!")
    else:
        print("⚠️ بعض النماذج غير مكتملة بعد.")
        print("لتحميلها واستئناف التنزيل تلقائياً، نفّذ الأمر:")
        print("   python scripts/download_models.py")
    print("=" * 68)
    return all_found


def download_with_curl(url: str, target_file: Path, token: str, expected_size: int, max_retries: int = 50) -> bool:
    """Download using system curl with resume (-C -) and retry flags."""
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_bin:
        return False

    for attempt in range(1, max_retries + 1):
        current_size = target_file.stat().st_size if target_file.exists() else 0
        if current_size >= expected_size:
            return True

        if current_size > 0:
            pct = (current_size / expected_size) * 100
            print(f"[{attempt}/{max_retries}] استئناف التحميل من نقطة: {current_size / (1024*1024):.1f} MB ({pct:.1f}%)...")
        else:
            print(f"[{attempt}/{max_retries}] بدء التحميل...")

        cmd = [
            curl_bin,
            "-L",                  # Follow redirects
            "-C", "-",             # Resume transfer at current file size
            "-k",                  # Allow insecure SSL fallback if needed
            "--retry", "10",       # Retry on transient errors
            "--retry-delay", "3",
            "--retry-max-time", "120",
            "-H", f"Authorization: Bearer {token}",
            "-o", str(target_file),
            url
        ]

        try:
            res = subprocess.run(cmd)
            current_size = target_file.stat().st_size if target_file.exists() else 0
            if current_size >= expected_size:
                return True
            print(f"⚠️ انتهت جلسة التحميل ولم يكتمل الملف ({current_size}/{expected_size} بايت). إعادة المحاولة...")
            time.sleep(3)
        except Exception as e:
            print(f"⚠️ خطأ أثناء التحميل: {e}. إعادة المحاولة بعد 5 ثوان...")
            time.sleep(5)

    return (target_file.stat().st_size if target_file.exists() else 0) >= expected_size


def download_all(token: str = DEFAULT_HF_TOKEN):
    """Download all required models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("   بدء تحميل نماذج الذكاء الاصطناعي الأساسية للتطبيق")
    print(f"   المسار: {MODELS_DIR.resolve()}")
    print("   الموديلات: Falcon-H1-7B-Instruct + Audar-ASR-V1-Turbo")
    print("=" * 68)
    print()

    for i, model in enumerate(MODELS, 1):
        target = MODELS_DIR / model["filename"]
        if target.exists() and target.stat().st_size >= model["expected_size"]:
            size_mb = target.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            print(f"✅ [{i}/{len(MODELS)}] {model['description']}")
            print(f"   الملف مكتمل بالفعل: {target.name} ({size_str})")
            print()
            continue

        print(f"⬇️  [{i}/{len(MODELS)}] جاري تحميل: {model['description']}")
        print(f"   الملف: {model['filename']} ({model['size_str']})")
        print()

        success = download_with_curl(
            url=model["url"],
            target_file=target,
            token=token,
            expected_size=model["expected_size"],
        )

        if success:
            print(f"✅ تم تحميل {model['filename']} بنجاح!")
        else:
            print(f"❌ لم يتم اكتمال تحميل {model['filename']}. يمكنك تشغيل السكربت مجدداً للاستئناف.")
        print()

def check_and_download_pyannote(token: str = DEFAULT_HF_TOKEN):
    pyannote_base = MODELS_DIR / "pyannote"
    seg = pyannote_base / "segmentation-3.0" / "pytorch_model.bin"
    emb = pyannote_base / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin"
    cfg = pyannote_base / "speaker-diarization-3.1" / "config.yaml"
    plda_dir = pyannote_base / "plda"
    plda_1 = plda_dir / "plda.npz"
    plda_2 = plda_dir / "xvec_transform.npz"

    if seg.exists() and emb.exists() and cfg.exists() and plda_1.exists() and plda_2.exists():
        print("✅ [مكتمل] نموذج فصل المتحدثين Pyannote 3.1 مع معايرة PLDA متوفر محلياً (100% Offline)")
        return True

    print("⬇️ جاري تحميل أوزان نموذج Pyannote 3.1 وملفات المعايرة محلياً...")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        if not (seg.exists() and cfg.exists()):
            snapshot_download('pyannote/speaker-diarization-3.1', local_dir=pyannote_base / 'speaker-diarization-3.1', token=token)
            snapshot_download('pyannote/segmentation-3.0', local_dir=pyannote_base / 'segmentation-3.0', token=token)
        if not emb.exists():
            snapshot_download('pyannote/wespeaker-voxceleb-resnet34-LM', local_dir=pyannote_base / 'wespeaker-voxceleb-resnet34-LM', token=token)
        if not (plda_1.exists() and plda_2.exists()):
            plda_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download('pyannote/speaker-diarization-community-1', filename='plda/plda.npz', local_dir=pyannote_base, token=token)
            hf_hub_download('pyannote/speaker-diarization-community-1', filename='plda/xvec_transform.npz', local_dir=pyannote_base, token=token)
        print("✅ تم تحميل نموذج Pyannote 3.1 وملفات المعايرة محلياً بنجاح!")
        return True
    except Exception as e:
        print(f"⚠️ تعذر تنزيل Pyannote: {e}")
        return False


def check_and_download_whisper_dialectal(token: str = DEFAULT_HF_TOKEN):
    whisper_ct2 = MODELS_DIR / "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"
    if (whisper_ct2 / "model.bin").exists() and (whisper_ct2 / "config.json").exists():
        print("✅ [مكتمل] نموذج Whisper Large-v3 Turbo Arabic Dialectal (CTranslate2) متوفر محلياً")
        return True
    print("⬇️ جاري تحميل وتحويل نموذج Whisper Dialectal إلى CTranslate2...")
    try:
        script = Path(__file__).resolve().parent / "download_whisper.py"
        res = subprocess.run([sys.executable, str(script), token])
        return res.returncode == 0
    except Exception as e:
        print(f"⚠️ خطأ أثناء تجهيز Whisper Dialectal: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_models()
        check_and_download_pyannote()
        check_and_download_whisper_dialectal()
    else:
        custom_token = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_HF_TOKEN
        download_all(token=custom_token)
        check_and_download_pyannote(token=custom_token)
        check_and_download_whisper_dialectal(token=custom_token)
