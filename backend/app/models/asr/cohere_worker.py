"""
Isolated worker process for Cohere Transcribe Arabic.
Runs in a dedicated clean Python process to prevent dynamic link library (DLL)
symbol collisions on Windows between llama-cpp-python and transcribe_cpp (GGML).
"""
import os
import re
import sys
import json
import logging
from pathlib import Path
import numpy as np

# Ensure backend venv site-packages is in sys.path across Windows and Linux/macOS
_backend_venv = Path(__file__).resolve().parents[4] / "backend" / ".venv"
_venv_candidates = [_backend_venv / "Lib" / "site-packages"]
if (_backend_venv / "lib").exists():
    _venv_candidates.extend((_backend_venv / "lib").glob("python*/site-packages"))
for _venv_site in _venv_candidates:
    if _venv_site.exists() and str(_venv_site) not in sys.path:
        sys.path.insert(0, str(_venv_site))


def _clean_cohere_tokens(text: str) -> str:
    """Strip internal Cohere special tokens (@@@فراغ, @@@ضحك, @@@ضوضاء, ###...)."""
    if not text:
        return ""
    text = re.sub(r"@@@\S+", " ", text)
    text = re.sub(r"#{2,}[^\s.]*\.?", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Set standard output encoding to utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

def _ensure_linux_nvidia_vulkan_icd() -> None:
    """
    In slim Linux Docker containers with NVIDIA GPUs, nvidia-container-toolkit
    mounts NVIDIA libraries into /usr/lib/x86_64-linux-gnu or /usr/local/nvidia/lib64,
    but /etc/vulkan/icd.d/nvidia_icd.json may be absent. Create a temporary ICD
    descriptor in /tmp pointing to libGLX_nvidia so ggml-vulkan can discover the NVIDIA GPU.
    """
    if not sys.platform.startswith("linux"):
        return

    # Check for existing valid ICD manifests
    standard_icds = [
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/etc/vulkan/icd.d"),
        Path("/usr/share/vulkan/icd.d"),
    ]
    found_existing = False
    for p in standard_icds:
        if p.is_file():
            found_existing = True
            break
        elif p.is_dir() and any(p.glob("*.json")):
            found_existing = True
            break

    # Search for NVIDIA GLX / Vulkan library across standard container paths
    search_dirs = [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/local/nvidia/lib"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
    ]

    found_lib = None
    for d in search_dirs:
        if not d.exists():
            continue
        # Direct check for libGLX_nvidia.so.0 or versioned variants
        candidates = list(d.glob("libGLX_nvidia.so*"))
        if candidates:
            found_lib = candidates[0]
            # Ensure the driver directory is in LD_LIBRARY_PATH so dependent libs (e.g. libnvidia-glcore) load
            lib_dir = str(d)
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in cur_ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{cur_ld}" if cur_ld else lib_dir
            break

    if found_lib and not found_existing:
        try:
            tmp_icd = Path("/tmp/nvidia_vulkan_icd.json")
            tmp_icd.write_text(
                json.dumps({
                    "file_format_version": "1.0.0",
                    "ICD": {
                        "library_path": str(found_lib.resolve()),
                        "api_version": "1.3.0",
                    },
                }),
                encoding="utf-8",
            )
            os.environ["VK_ICD_FILENAMES"] = str(tmp_icd)
            os.environ["VK_DRIVER_FILES"] = str(tmp_icd)
            sys.stderr.write(f"[cohere_worker] Created temporary Vulkan ICD pointing to {found_lib}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[cohere_worker] Failed creating Vulkan ICD: {e}\n")
            sys.stderr.flush()


def main():
    if len(sys.argv) < 2:
        sys.stdout.write(json.dumps({"status": "error", "message": "Missing model path argument"}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    model_path = sys.argv[1]
    preferred_backend = (sys.argv[2] if len(sys.argv) > 2 else "auto").strip().lower()

    _ensure_linux_nvidia_vulkan_icd()

    try:
        import transcribe_cpp
        from scipy.io import wavfile

        # Suppress verbose C++ GGML/transcribe logs to prevent pipe saturation
        try:
            transcribe_cpp.set_log_callback(lambda level, msg: None)
        except Exception:
            pass
    except ImportError as e:
        sys.stdout.write(json.dumps({"status": "error", "message": f"Import error in worker: {e}"}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    model = None
    loaded_backend = None

    # Probe backends in order of speed: GPU (cuda / vulkan) -> auto -> AVX CPU (cpu_accel) -> baseline cpu
    backends_to_try = []
    has_probe = hasattr(transcribe_cpp, "backend_available")

    def _add_backend(name: str, check_avail: bool = True):
        if name in backends_to_try:
            return
        if check_avail and has_probe:
            try:
                avail = transcribe_cpp.backend_available(name)
                if not avail:
                    sys.stderr.write(f"[cohere_worker] transcribe_cpp probe: backend '{name}' reported not available.\n")
                    sys.stderr.flush()
                    return
            except Exception as e:
                sys.stderr.write(f"[cohere_worker] transcribe_cpp probe for '{name}' error: {e}\n")
                sys.stderr.flush()
                return
        backends_to_try.append(name)

    if preferred_backend == "cpu":
        _add_backend("cpu_accel", check_avail=True)
        _add_backend("cpu", check_avail=False)
    else:
        if preferred_backend not in ("auto", "gpu", "cuda", "vulkan", "cpu_accel"):
            _add_backend(preferred_backend, check_avail=False)
        _add_backend("cuda", check_avail=True)
        _add_backend("vulkan", check_avail=True)
        _add_backend("auto", check_avail=False)
        # If vulkan was excluded by the probe, still try it directly to get C++ error or bypass probe quirk
        if "vulkan" not in backends_to_try:
            backends_to_try.append("vulkan")
        _add_backend("cpu_accel", check_avail=True)
        _add_backend("cpu", check_avail=False)

    last_err = None
    for b in backends_to_try:
        try:
            model = transcribe_cpp.Model(model_path, backend=b)
            loaded_backend = b
            break
        except Exception as e:
            last_err = e
            # Surface why a (GPU) backend was skipped instead of silently falling back to CPU
            sys.stderr.write(f"[cohere_worker] backend '{b}' failed: {e}\n")
            sys.stderr.flush()

    if model is None:
        sys.stdout.write(json.dumps({
            "status": "error",
            "message": f"Failed to load Cohere model: {last_err}"
        }) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    actual_backend = loaded_backend
    device_type = ""
    device_desc = ""
    try:
        if getattr(model, "backend", None):
            actual_backend = str(model.backend).strip().lower()
    except Exception:
        pass
    try:
        dev = getattr(model, "device", None)
        if dev is not None:
            device_type = str(getattr(dev, "device_type", "") or "").strip().lower()
            device_desc = str(getattr(dev, "description", "") or getattr(dev, "name", "") or "").strip()
    except Exception:
        pass

    is_gpu = (
        actual_backend in ("cuda", "vulkan", "metal", "rocm", "gpu")
        or device_type in ("gpu", "igpu")
    )

    # Optimal thread count: utilize CPU cores appropriately without oversubscribing
    cpu_cores = os.cpu_count() or 4
    if is_gpu:
        optimal_threads = max(2, min(8, cpu_cores // 2))
    else:
        optimal_threads = max(2, min(12, cpu_cores - 2 if cpu_cores > 4 else cpu_cores))

    # Initialize persistent session for fast sequential slice processing
    session = None
    try:
        session = model.session(n_threads=optimal_threads)
    except Exception as e:
        sys.stdout.write(json.dumps({
            "status": "error",
            "message": f"Failed to initialize transcribe_cpp session: {e}"
        }) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Signal parent that model and session are loaded and ready
    sys.stdout.write(json.dumps({
        "status": "ready",
        "backend": actual_backend,
        "is_gpu": is_gpu,
        "device": device_desc,
        "threads": optimal_threads
    }) + "\n")
    sys.stdout.flush()

    slice_count = 0
    # Request loop via stdin
    for line in iter(sys.stdin.readline, ""):
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            break
        elif cmd == "ping":
            sys.stdout.write(json.dumps({"status": "pong"}) + "\n")
            sys.stdout.flush()
            continue
        elif cmd == "transcribe_slice":
            slice_path = req.get("path")
            raw_audio = None
            audio_f32 = None
            res = None
            try:
                sr, raw_audio = wavfile.read(slice_path)
                if raw_audio.dtype == np.int16:
                    audio_f32 = (raw_audio.astype(np.float32) / 32768.0).copy()
                elif raw_audio.dtype == np.int32:
                    audio_f32 = (raw_audio.astype(np.float32) / 2147483648.0).copy()
                elif raw_audio.dtype == np.float32:
                    audio_f32 = raw_audio.copy()
                else:
                    audio_f32 = raw_audio.astype(np.float32).copy()

                if audio_f32.ndim > 1:
                    audio_f32 = audio_f32.mean(axis=1)

                # Skip completely empty or silent slices (< 0.15s or absolute peak < 1e-4)
                if len(audio_f32) < int(sr * 0.15) or np.max(np.abs(audio_f32)) < 1e-4:
                    sys.stdout.write(json.dumps({"status": "ok", "text": ""}) + "\n")
                    sys.stdout.flush()
                    continue

                # Run inference on persistent session without per-slice allocation churn
                try:
                    res = session.run(audio_f32, language="ar")
                    text = getattr(res, "text", "") or ""
                except transcribe_cpp.errors.OutputTruncated as trunc_err:
                    partial = getattr(trunc_err, "partial_result", None)
                    text = getattr(partial, "text", "") if partial else ""
                    if not text:
                        raise trunc_err

                # Strip internal Cohere silence/noise/laughter tokens like @@@فراغ, @@@ضحك, @@@ضوضاء, ###...
                text = _clean_cohere_tokens(text)
                sys.stdout.write(json.dumps({"status": "ok", "text": text}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                # If error had partial_result, extract it
                partial = getattr(e, "partial_result", None)
                partial_text = _clean_cohere_tokens(getattr(partial, "text", "") if partial else "")
                if partial_text:
                    sys.stdout.write(json.dumps({"status": "ok", "text": partial_text}) + "\n")
                    sys.stdout.flush()
                    continue

                # On session run failure, recreate session once and retry with slightly trimmed audio (removes trailing breath/silence loop)
                try:
                    if session and hasattr(session, "close"):
                        session.close()
                except Exception as close_err:
                    sys.stderr.write(f"[cohere_worker] Warning closing failed session: {close_err}\n")
                    sys.stderr.flush()
                try:
                    session = model.session(n_threads=optimal_threads)
                    # Trim 0.15s off the end if audio is long enough to avoid trailing silence loops
                    retry_audio = audio_f32
                    if len(audio_f32) > int(sr * 0.8):
                        retry_audio = audio_f32[:-int(sr * 0.15)]
                    try:
                        res = session.run(retry_audio, language="ar")
                        text = getattr(res, "text", "") or ""
                    except transcribe_cpp.errors.OutputTruncated as r_trunc:
                        r_part = getattr(r_trunc, "partial_result", None)
                        text = getattr(r_part, "text", "") if r_part else ""
                    text = _clean_cohere_tokens(text)
                    sys.stdout.write(json.dumps({"status": "ok", "text": text}) + "\n")
                    sys.stdout.flush()
                except Exception as retry_err:
                    r_part = getattr(retry_err, "partial_result", None)
                    r_text = _clean_cohere_tokens(getattr(r_part, "text", "") if r_part else "")
                    if r_text:
                        sys.stdout.write(json.dumps({"status": "ok", "text": r_text}) + "\n")
                    else:
                        sys.stdout.write(json.dumps({"status": "error", "message": str(retry_err), "text": ""}) + "\n")
                    sys.stdout.flush()
            finally:
                # Explicit cleanup of large audio arrays to prevent memory creep
                raw_audio = None
                audio_f32 = None
                res = None
                slice_count += 1
                if slice_count % 30 == 0:
                    import gc
                    gc.collect()

    try:
        if session and hasattr(session, "close"):
            session.close()
    except Exception:
        pass
    try:
        if hasattr(model, "close"):
            model.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
