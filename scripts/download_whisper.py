"""
Download and convert Whisper Large-v3 Turbo Arabic Dialectal (oddadmix) to CTranslate2.
Uses curl with resume (-C -) and infinite retries for rock-solid reliability.
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

WHISPER_REPO = "oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2"
WHISPER_RAW_DIR = MODELS_DIR / "whisper-raw" / "whisper-large-v3-turbo-arabic-dialectal-v2"
WHISPER_CT2_DIR = MODELS_DIR / "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"

# Files to download from oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2
FILES_TO_DOWNLOAD = [
    ("config.json", 1253),
    ("generation_config.json", 4489),
    ("processor_config.json", 410),
    ("tokenizer_config.json", 2105),
    ("tokenizer.json", 3931381),
    ("model.safetensors", 3235581408),
]


def download_file_curl(url: str, target_file: Path, token: str, expected_size: int, max_retries: int = 100) -> bool:
    """Download single file using curl with resume (-C -) and retry loop."""
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_bin:
        raise RuntimeError("curl binary not found in PATH")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    part_file = target_file.with_suffix(target_file.suffix + ".part")

    if target_file.exists() and target_file.stat().st_size >= expected_size:
        print(f"[+] Already downloaded: {target_file.name} ({target_file.stat().st_size / (1024*1024):.2f} MB)")
        return True

    if part_file.exists() and part_file.stat().st_size >= expected_size:
        if target_file.exists():
            target_file.unlink()
        part_file.rename(target_file)
        print(f"[+] Completed part file: {target_file.name}")
        return True

    print(f"[*] Starting download: {target_file.name} (Expected: {expected_size / (1024*1024):.2f} MB)...")

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        current_size = part_file.stat().st_size if part_file.exists() else 0
        pct = (current_size / expected_size * 100) if expected_size > 0 else 0
        print(f"[{attempt}/{max_retries}] Downloading {target_file.name}: {current_size/(1024*1024):.2f} MB ({pct:.1f}%)...")

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
                print(f"[+] Download complete: {target_file.name} ({current_size / (1024*1024):.2f} MB)")
                return True
            print(f"[!] Session ended ({current_size}/{expected_size} bytes). Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[!] Error: {e}. Retrying in 5s...")
            time.sleep(5)

    return False


def convert_to_ctranslate2():
    """Convert downloaded safetensors to CTranslate2 int8_float16."""
    print("=" * 70)
    print("بدء تحويل نموذج Whisper Dialectal إلى صيغة CTranslate2...")
    print(f"المصدر: {WHISPER_RAW_DIR}")
    print(f"الهدف: {WHISPER_CT2_DIR}")
    print("=" * 70)

    from ctranslate2.converters import TransformersConverter
    import torch

    WHISPER_CT2_DIR.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    print(f"[*] GPU Available for conversion: {use_cuda}")

    converter = TransformersConverter(
        model_name_or_path=str(WHISPER_RAW_DIR),
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        load_as_float16=use_cuda,
        low_cpu_mem_usage=True,
    )

    print("[*] جارٍ التحويل إلى int8_float16...")
    converter.convert(
        output_dir=str(WHISPER_CT2_DIR),
        quantization="int8_float16",
        force=True,
    )
    print("[+] تم التحويل بنجاح!")

    # Verify with faster-whisper
    from faster_whisper import WhisperModel
    print("[*] التحقق من تحميل النموذج عبر faster-whisper...")
    test_device = "cuda" if use_cuda else "cpu"
    test_compute = "int8_float16" if use_cuda else "int8"
    model = WhisperModel(str(WHISPER_CT2_DIR), device=test_device, compute_type=test_compute)
    del model
    print("[+] تم التحقق بنجاح من صحة النموذج المحول!")

    # Clean up raw safetensors to save disk space
    raw_st = WHISPER_RAW_DIR / "model.safetensors"
    if raw_st.exists():
        print(f"[*] تنظيف ملف safetensors الأصلي لتوفير المساحة ({raw_st.stat().st_size / (1024*1024*1024):.2f} GB)...")
        try:
            raw_st.unlink()
            print("[+] تم حذف الملف الخام المؤقت بنجاح.")
        except Exception as e:
            print(f"[!] تحذير أثناء حذف الملف: {e}")


def main():
    token = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_HF_TOKEN

    model_bin = WHISPER_CT2_DIR / "model.bin"
    if model_bin.exists() and model_bin.stat().st_size > 500 * 1024 * 1024:
        print(f"[+] نموذج Whisper CTranslate2 جاهز بالفعل في: {WHISPER_CT2_DIR}")
        return

    WHISPER_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, size in FILES_TO_DOWNLOAD:
        url = f"https://huggingface.co/{WHISPER_REPO}/resolve/main/{filename}"
        dest = WHISPER_RAW_DIR / filename
        ok = False
        while not ok:
            try:
                ok = download_file_curl(url=url, target_file=dest, token=token, expected_size=size)
            except Exception as e:
                print(f"[!] خطأ: {e}. إعادة المحاولة بعد 4 ثوان...")
                time.sleep(4)

    # All files downloaded, now convert
    convert_to_ctranslate2()
    print("=" * 70)
    print("✅ اكتمل تحميل وتحويل نموذج Whisper Dialectal بنجاح!")
    print("=" * 70)


if __name__ == "__main__":
    main()
