"""
Cohere Transcribe Arabic ASR Engine Implementation.
Uses transcribe_cpp (ggml runtime) for high-performance offline inference of
cohere-transcribe-arabic-07-2026 GGUF models.

Adheres strictly to BaseASREngine interface with VAD and audio slicing separated.
"""
import gc
import json
import logging
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, List

from .base import BaseASREngine
from ...config import settings

logger = logging.getLogger(__name__)


class CohereASREngine(BaseASREngine):
    """
    Cohere Transcribe Arabic Engine Implementation via transcribe.cpp (GGML).
    Specialized for dialectal Arabic and Arabic-English code-switching.
    Runs in an isolated worker process to prevent dynamic library (DLL) symbol collisions
    on Windows between llama-cpp-python and transcribe_cpp.
    """

    # Cohere runs via transcribe_cpp in an isolated subprocess.
    # Because transcribe_cpp utilizes multi-threaded CPU inference, running a single
    # dedicated worker avoids CPU contention, memory duplication, and timeouts.
    allow_secondary_workers: bool = False

    def __init__(self, model_path: Optional[Path] = None):
        default_path = settings.MODELS_DIR / getattr(
            settings, "COHERE_ASR_FILENAME", "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"
        )
        super().__init__(model_path or default_path)
        self.process: Optional[subprocess.Popen] = None
        self._worker_script = Path(__file__).resolve().parent / "cohere_worker.py"
        self._stdout_queue: queue.Queue = queue.Queue()
        self._stderr_buffer: List[str] = []
        self._drain_threads: List[threading.Thread] = []

    def _start_drain_threads(self, proc: subprocess.Popen):
        """
        Continuously read stdout into a thread-safe queue and drain stderr into a ring buffer.
        Prevents Windows anonymous pipe deadlocks where OS buffers (4-32 KB) fill up.
        """
        def _drain_stderr(pipe, buf: List[str]):
            try:
                for line in iter(pipe.readline, ""):
                    if line:
                        buf.append(line)
                        if len(buf) > 100:
                            buf.pop(0)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        def _drain_stdout(pipe, q: queue.Queue):
            try:
                for line in iter(pipe.readline, ""):
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(None)
                try:
                    pipe.close()
                except Exception:
                    pass

        t_err = threading.Thread(target=_drain_stderr, args=(proc.stderr, self._stderr_buffer), daemon=True)
        t_out = threading.Thread(target=_drain_stdout, args=(proc.stdout, self._stdout_queue), daemon=True)
        t_err.start()
        t_out.start()
        self._drain_threads = [t_err, t_out]

    def load(self):
        """Load Cohere GGUF model into memory using isolated worker process."""
        if self.is_loaded():
            return

        if not self.model_path.exists():
            msg = (
                f"ملف نموذج Cohere Transcribe غير موجود: {self.model_path.name}. "
                "يرجى تشغيل سكربت التحميل أولاً: python scripts/download_new_models.py"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading Cohere Transcribe Arabic from {self.model_path.name} in isolated worker...")

        # Hardware Resilience (Rule 3): Try auto (Vulkan/CUDA/GPU), gracefully fallback to CPU
        loaded = False
        last_error = ""
        for backend_choice in ["auto", "cpu"]:
            try:
                logger.info(f"Attempting to spawn Cohere worker with backend='{backend_choice}'...")
                self._stderr_buffer.clear()
                while not self._stdout_queue.empty():
                    try:
                        self._stdout_queue.get_nowait()
                    except Exception:
                        break

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(self._worker_script),
                        str(self.model_path.resolve()),
                        backend_choice,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )

                self._start_drain_threads(proc)

                # Wait for ready signal via queue with 180s timeout (allows cold loading from disk)
                try:
                    ready_line = self._stdout_queue.get(timeout=180.0)
                except queue.Empty:
                    ready_line = None

                if not ready_line:
                    err_out = "".join(self._stderr_buffer).strip() or "Empty response / timeout from worker"
                    last_error = err_out
                    logger.warning(f"Worker backend='{backend_choice}' failed: {last_error}")
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    continue

                resp = json.loads(ready_line)
                if resp.get("status") == "ready":
                    self.process = proc
                    loaded = True
                    actual_backend = resp.get("backend", backend_choice)
                    worker_threads = resp.get("threads", "auto")
                    logger.info(f"Cohere Transcribe Arabic worker ready using backend='{actual_backend}' (threads={worker_threads}).")
                    break
                else:
                    last_error = resp.get("message", "Unknown worker error")
                    logger.warning(f"Worker returned error for backend='{backend_choice}': {last_error}")
                    try:
                        proc.kill()
                    except Exception:
                        pass
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Failed to spawn Cohere worker with backend='{backend_choice}': {e}")

        if not loaded or self.process is None:
            self.unload()
            raise RuntimeError(f"فشل تحميل نموذج Cohere Transcribe Arabic من {self.model_path.name}: {last_error}")

    def unload(self):
        """Unload Cohere model and free memory."""
        if self.process is not None:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    try:
                        self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                        self.process.stdin.flush()
                        self.process.stdin.close()
                    except Exception:
                        pass
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        self._drain_threads.clear()
        while not self._stdout_queue.empty():
            try:
                self._stdout_queue.get_nowait()
            except Exception:
                break
        self._stderr_buffer.clear()

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Cohere Transcribe Arabic engine unloaded from memory.")

    def is_loaded(self) -> bool:
        """Check if model worker is currently running."""
        return self.process is not None and self.process.poll() is None

    def _transcribe_slice_internal(self, slice_wav_path: Path, dur_sec: float) -> str:
        """
        Transcribe a single short audio slice (<= 30s) using Cohere Transcribe Arabic worker.
        Language is forced to Arabic ('ar') inside the worker without altering any prompt.
        Protected by a 35s per-slice timeout and non-blocking background pipe readers.
        """
        if not self.is_loaded():
            self.load()

        if self.process is None or self.process.stdin is None or self.process.stdin.closed:
            logger.error("Cohere worker process not available.")
            return ""

        # Drain any residual line from queue
        while not self._stdout_queue.empty():
            try:
                self._stdout_queue.get_nowait()
            except Exception:
                break

        try:
            req = {
                "cmd": "transcribe_slice",
                "path": str(slice_wav_path.resolve()),
            }
            self.process.stdin.write(json.dumps(req) + "\n")
            self.process.stdin.flush()

            # Wait for response with generous timeout to allow CPU or complex audio processing
            slice_timeout = max(180.0, float(dur_sec) * 8.0)
            try:
                resp_line = self._stdout_queue.get(timeout=slice_timeout)
            except queue.Empty:
                recent_err = "".join(self._stderr_buffer[-10:]).strip()
                logger.error(
                    f"Cohere worker slice transcription timed out after {slice_timeout:.0f}s ({slice_wav_path.name})! "
                    f"Recent stderr: {recent_err}"
                )
                self.unload()
                return ""

            if not resp_line:
                logger.warning("Empty response from Cohere worker.")
                self.unload()
                return ""

            resp = json.loads(resp_line)
            raw_text = resp.get("text", "") or ""
            raw_text = raw_text.replace("@@@فراغ", "").strip()

            # Worker Process Recycling:
            # Every 25 slices, recycle the worker subprocess to release all OS heap memory back to Linux
            self._slice_count = getattr(self, "_slice_count", 0) + 1
            if self._slice_count % 25 == 0:
                logger.info(f"Recycling Cohere worker process after {self._slice_count} slices to prevent memory accumulation...")
                self.unload()
                self.load()

            return raw_text
        except Exception as e:
            logger.warning(f"Error during Cohere slice transcription via worker: {e}")
            return ""
