"""
Isolated worker process for Cohere Transcribe Arabic.
Runs in a dedicated clean Python process to prevent dynamic link library (DLL)
symbol collisions on Windows between llama-cpp-python and transcribe_cpp (GGML).
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np

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

    # Try preferred backend (auto/vulkan/gpu), fallback to CPU on error
    backends_to_try = [preferred_backend] if preferred_backend == "cpu" else [preferred_backend, "cpu"]
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

    # Signal parent that model is loaded and ready
    sys.stdout.write(json.dumps({
        "status": "ready",
        "backend": loaded_backend
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

                res = transcribe_cpp.transcribe(model, audio_f32, language="ar")
                text = getattr(res, "text", "") or ""
                # Strip internal Cohere silence/blank tokens like @@@فراغ
                text = text.replace("@@@فراغ", "").strip()
                sys.stdout.write(json.dumps({"status": "ok", "text": text}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(json.dumps({"status": "error", "message": str(e), "text": ""}) + "\n")
                sys.stdout.flush()
            finally:
                # Explicit cleanup of large audio arrays to prevent memory creep across hundreds of slices
                raw_audio = None
                audio_f32 = None
                res = None
                slice_count += 1
                if slice_count % 15 == 0:
                    import gc
                    gc.collect()

    try:
        if hasattr(model, "close"):
            model.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
