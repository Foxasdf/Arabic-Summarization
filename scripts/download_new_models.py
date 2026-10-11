"""
Download and prepare:
1. Cohere Transcribe Arabic Q5_K_M GGUF
2. Whisper Large-v3 Turbo Arabic Dialectal (oddadmix) converted to CTranslate2 (int8_float16 / int8)

Includes auto-resume and infinite retry until complete.
"""
import sys
import os
import time
import shutil
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HF_TOKEN = os.getenv("MEETING_HF_TOKEN") or os.getenv("HF_TOKEN") or "REDACTED_HF_TOKEN"

# 1. Cohere Model Details
COHERE_FILENAME = "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"
COHERE_URL = f"https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/{COHERE_FILENAME}"
COHERE_EXPECTED_SIZE = 1770270112  # bytes (~1.65 GB)
COHERE_TARGET = MODELS_DIR / COHERE_FILENAME

# 2. Whisper Model Details
WHISPER_REPO = "oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2"
WHISPER_RAW_DIR = MODELS_DIR / "whisper-raw" / "whisper-large-v3-turbo-arabic-dialectal-v2"
WHISPER_CT2_DIR = MODELS_DIR / "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"


def download_file_curl(url: str, target_file: Path, token: str, expected_size: int, max_retries: int = 100) -> bool:
    """Download single file using curl with resume (-C -) and retry loop."""
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_bin:
        raise RuntimeError("curl binary not found in PATH")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    part_file = target_file.with_suffix(target_file.suffix + ".part")

    # If target already complete
    if target_file.exists() and target_file.stat().st_size >= expected_size:
        return True

    # If partial file exists and equals or exceeds expected size
    if part_file.exists() and part_file.stat().st_size >= expected_size:
        if target_file.exists():
            target_file.unlink()
        part_file.rename(target_file)
        return True

    print(f"[*] Starting download of {target_file.name} (Expected: {expected_size / (1024*1024):.1f} MB)...")

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        current_size = part_file.stat().st_size if part_file.exists() else 0
        pct = (current_size / expected_size * 100) if expected_size > 0 else 0
        print(f"[{attempt}/{max_retries}] Downloading {target_file.name}: {current_size/(1024*1024):.1f} MB ({pct:.1f}%)...")

        cmd = [
            curl_bin,
            "-L",
            "-C", "-",
            "-k",
            "--retry", "15",
            "--retry-delay", "3",
            "--retry-max-time", "180",
            "-H", f"Authorization: Bearer {token}",
            "-o", str(part_file),
            url,
        ]

        try:
            subprocess.run(cmd, check=False)
            current_size = part_file.stat().st_size if part_file.exists() else 0
            if current_size >= expected_size:
                if target_file.exists():
                    target_file.unlink()
                part_file.rename(target_file)
                print(f"[+] Download complete: {target_file.name} ({current_size / (1024*1024):.1f} MB)")
                return True
            print(f"[!] Session ended before completion ({current_size}/{expected_size} bytes). Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[!] Error: {e}. Retrying in 5s...")
            time.sleep(5)

    return False


def download_cohere_model(token: str = DEFAULT_HF_TOKEN) -> bool:
    """Download Cohere Transcribe Arabic GGUF."""
    print("=" * 70)
    print("1. التحقق من نموذج Cohere Transcribe Arabic Q5_K_M GGUF")
    print("=" * 70)

    if COHERE_TARGET.exists() and COHERE_TARGET.stat().st_size >= COHERE_EXPECTED_SIZE:
        print(f"[+] النموذج موجود ومكتمل بالفعل: {COHERE_TARGET.name} ({COHERE_TARGET.stat().st_size / (1024*1024):.1f} MB)")
        return True

    success = download_file_curl(
        url=COHERE_URL,
        target_file=COHERE_TARGET,
        token=token,
        expected_size=COHERE_EXPECTED_SIZE,
    )
    return success


def download_and_convert_whisper_dialectal(token: str = DEFAULT_HF_TOKEN) -> bool:
    """Download oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2 and convert to CTranslate2."""
    script_path = PROJECT_ROOT / "scripts" / "download_whisper.py"
    cmd = [sys.executable, str(script_path), token]
    res = subprocess.run(cmd)
    return res.returncode == 0



def main():
    token = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_HF_TOKEN

    print("=" * 70)
    print("بدء تحميل وتجهيز الموديلات الجديدة للتفريغ الصوتي")
    print("=" * 70)

    # 1. Download Cohere
    cohere_ok = False
    while not cohere_ok:
        try:
            cohere_ok = download_cohere_model(token=token)
        except Exception as e:
            print(f"[!] خطأ في Cohere: {e}. إعادة المحاولة بعد 5 ثوان...")
            time.sleep(5)

    # 2. Download and Convert Whisper
    whisper_ok = False
    while not whisper_ok:
        try:
            whisper_ok = download_and_convert_whisper_dialectal(token=token)
        except Exception as e:
            print(f"[!] خطأ في Whisper: {e}. إعادة المحاولة بعد 5 ثوان...")
            time.sleep(5)

    print()
    print("=" * 70)
    print("✅ تم بنجاح تحميل وتجهيز كلا النموذجين:")
    print(f"   1. Cohere GGUF: {COHERE_TARGET}")
    print(f"   2. Whisper CT2: {WHISPER_CT2_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
