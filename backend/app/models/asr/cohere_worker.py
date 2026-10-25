"""
Isolated worker process for Cohere Transcribe Arabic.
Runs in a dedicated clean Python process to prevent dynamic link library (DLL)
symbol collisions on Windows between llama-cpp-python and transcribe_cpp (GGML).
"""
import os
import sys
import json
import logging
from pathlib import Path
import numpy as np

# Ensure backend venv site-packages is in sys.path
_venv_site = Path(__file__).resolve().parents[4] / "backend" / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

# Set standard output encoding to utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

def main():
    if len(sys.argv) < 2:
        sys.stdout.write(json.dumps({"status": "error", "message": "Missing model path argument"}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    model_path = sys.argv[1]
    preferred_backend = sys.argv[2] if len(sys.argv) > 2 else "auto"

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

    # Probe backends: try cuda if available, then vulkan, then auto, then cpu
    backends_to_try = []
    if preferred_backend in ["cuda", "gpu", "auto", "vulkan"]:
        if hasattr(transcribe_cpp, "backend_available"):
            if transcribe_cpp.backend_available("cuda"):
                backends_to_try.append("cuda")
            if transcribe_cpp.backend_available("vulkan"):
                backends_to_try.append("vulkan")
        else:
            backends_to_try.append("cuda")
        if preferred_backend != "cuda":
            backends_to_try.append("auto")
    elif preferred_backend != "cpu":
        backends_to_try.append(preferred_backend)
    backends_to_try.append("cpu")

    last_err = None
    for b in backends_to_try:
        try:
            model = transcribe_cpp.Model(model_path, backend=b)
            loaded_backend = b
            break
        except Exception as e:
            last_err = e

    if model is None:
        sys.stdout.write(json.dumps({
            "status": "error",
            "message": f"Failed to load Cohere model: {last_err}"
        }) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Optimal thread count: utilize CPU cores appropriately
    cpu_cores = os.cpu_count() or 4
    if loaded_backend in ["cuda", "vulkan", "gpu"]:
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
        "backend": loaded_backend,
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

                # Strip internal Cohere silence/blank tokens like @@@فراغ
                text = text.replace("@@@فراغ", "").strip()
                sys.stdout.write(json.dumps({"status": "ok", "text": text}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                # If error had partial_result, extract it
                partial = getattr(e, "partial_result", None)
                partial_text = (getattr(partial, "text", "") if partial else "").replace("@@@فراغ", "").strip()
                if partial_text:
                    sys.stdout.write(json.dumps({"status": "ok", "text": partial_text}) + "\n")
                    sys.stdout.flush()
                    continue

                # On session run failure, recreate session once and retry with slightly trimmed audio (removes trailing breath/silence loop)
                try:
                    if session and hasattr(session, "close"):
                        session.close()
                except Exception:
                    pass
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
                    text = text.replace("@@@فراغ", "").strip()
                    sys.stdout.write(json.dumps({"status": "ok", "text": text}) + "\n")
                    sys.stdout.flush()
                except Exception as retry_err:
                    r_part = getattr(retry_err, "partial_result", None)
                    r_text = (getattr(r_part, "text", "") if r_part else "").replace("@@@فراغ", "").strip()
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
                # Periodically refresh session and garbage collect to guarantee zero memory leaks
                if slice_count % 30 == 0:
                    try:
                        if session and hasattr(session, "close"):
                            session.close()
                        session = model.session(n_threads=optimal_threads)
                    except Exception:
                        pass
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
