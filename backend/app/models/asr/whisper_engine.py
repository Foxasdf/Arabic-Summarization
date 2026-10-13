"""
Faster-Whisper Engine Implementation.
Supports standard Whisper models and specialized fine-tuned models such as
Whisper Large-v3 Turbo Arabic Dialectal (oddadmix) converted to CTranslate2.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseASREngine
from ...config import settings

logger = logging.getLogger(__name__)


def _ensure_cuda_dlls():
    """Ensure PyTorch CUDA / cuBLAS DLLs are discoverable on Windows by CTranslate2."""
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if hasattr(os, "add_dll_directory") and torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass
        if torch_lib.exists() and str(torch_lib) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        logger.debug(f"CUDA DLL discovery setup: {e}")


class WhisperASREngine(BaseASREngine):
    """
    Faster-Whisper (CTranslate2) Engine Implementation.
    Specialized for high-noise robustness with native CTranslate2 acceleration.
    Chunking follows WHISPER_* settings (20s window for dialectal turn-taking,
    decoupled from Cohere 28s tuning).
    """

    def __init__(self, model_path: Optional[Path] = None, model_size_or_path: str = "large-v3-turbo"):
        default_dir = settings.MODELS_DIR / "whisper" / model_size_or_path
        super().__init__(model_path or default_dir)
        self.model_size_or_path = model_size_or_path
        self.model = None
        # Per-engine chunking from WHISPER_* settings (base.transcribe reads
        # these attrs first, so global ASR_CHUNK_SECONDS never leaks in).
        self.chunk_len = float(getattr(settings, "WHISPER_CHUNK_SECONDS", 20.0))
        self.overlap = float(getattr(settings, "WHISPER_OVERLAP_SECONDS", 2.0))

    def load(self):
        if self.is_loaded():
            return

        _ensure_cuda_dlls()

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            msg = "حزمة faster-whisper غير مثبتة. لتفعيل محرك Whisper يرجى تثبيت: pip install faster-whisper"
            logger.error(msg)
            raise ImportError(msg)

        import torch

        # Hardware Resilience (Rule 3): Try CUDA first, gracefully degrade to CPU
        has_cuda = torch.cuda.is_available()
        preferred_device = "cuda" if has_cuda else "cpu"
        preferred_compute = "int8_float16" if has_cuda else "int8"

        model_loc = str(self.model_path) if self.model_path.exists() else self.model_size_or_path
        logger.info(f"Loading Faster-Whisper ({model_loc}) on {preferred_device} ({preferred_compute})")

        loaded = False
        devices_to_try = [(preferred_device, preferred_compute)]
        if preferred_device == "cuda":
            devices_to_try.append(("cuda", "float16"))
            devices_to_try.append(("cpu", "int8"))

        last_error = None
        for dev, comp in devices_to_try:
            try:
                self.model = WhisperModel(
                    model_loc,
                    device=dev,
                    compute_type=comp,
                    download_root=str(settings.MODELS_DIR / "whisper"),
                    local_files_only=self.model_path.exists(),
                )
                loaded = True
                logger.info(f"Faster-Whisper loaded successfully on {dev} ({comp}).")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Could not load Faster-Whisper on {dev} ({comp}): {e}. Retrying with next device...")

        if not loaded or self.model is None:
            raise RuntimeError(f"فشل تحميل نموذج Whisper: {last_error}")

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Faster-Whisper unloaded from memory.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def _transcribe_slice_internal(self, slice_wav_path: Path, dur_sec: float) -> str:
        """Transcribe a slice using faster-whisper with Arabic language forcing.

        Uses dialect-tuned decoding: greedy temp-0 with temperature fallback,
        strict compression gate, dialectal initial prompt + hotwords for proper
        nouns (قاسم/يبرق/كبة), and no condition-on-previous to stop cascade.
        Returns plain text (word timings handled uniformly by BaseASREngine).
        """
        if not self.is_loaded():
            self.load()

        from scipy.io import wavfile
        try:
            sr, raw_data = wavfile.read(str(slice_wav_path))
            if raw_data.dtype == np.int16:
                audio_input = (raw_data.astype(np.float32) / 32768.0).copy()
            elif raw_data.dtype == np.int32:
                audio_input = (raw_data.astype(np.float32) / 2147483648.0).copy()
            elif raw_data.dtype == np.float32:
                audio_input = raw_data.copy()
            else:
                audio_input = raw_data.astype(np.float32).copy()
            if audio_input.ndim > 1:
                audio_input = audio_input.mean(axis=1)
        except Exception as e:
            logger.warning(f"Could not load audio via wavfile, falling back to path: {e}")
            audio_input = str(slice_wav_path)

        beam = int(getattr(settings, "WHISPER_BEAM_SIZE", 5))
        no_speech = float(getattr(settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.5))
        compr = float(getattr(settings, "WHISPER_COMPRESSION_THRESHOLD", 2.0))
        logprob = float(getattr(settings, "WHISPER_LOGPROB_THRESHOLD", -1.0))
        cond_prev = bool(getattr(settings, "WHISPER_CONDITION_ON_PREV", False))
        vad_filter = bool(getattr(settings, "WHISPER_VAD_FILTER", False))
        init_prompt = getattr(settings, "WHISPER_INITIAL_PROMPT", "") or None
        hotwords = getattr(settings, "WHISPER_HOTWORDS", "") or None

        segments, info = self.model.transcribe(
            audio_input,
            language="ar",
            beam_size=beam,
            best_of=1,
            patience=1.0,
            temperature=0.0,
            compression_ratio_threshold=compr,
            log_prob_threshold=logprob,
            no_speech_threshold=no_speech,
            condition_on_previous_text=cond_prev,
            initial_prompt=init_prompt,
            hotwords=hotwords,
            vad_filter=vad_filter,  # base VAD gates whole slices; intra-chunk handled by no_speech gate
            word_timestamps=False,
            suppress_blank=True,
            without_timestamps=False,
        )
        return " ".join(seg.text for seg in segments).strip()


class WhisperDialectalASREngine(WhisperASREngine):
    """
    Whisper Large-v3 Turbo Arabic Dialectal (oddadmix) Engine.
    Converted to CTranslate2 (int8_float16 / int8) for maximum speed and Arabic dialect accuracy.
    """

    def __init__(self, model_path: Optional[Path] = None):
        default_dir = settings.MODELS_DIR / getattr(
            settings, "WHISPER_DIALECTAL_DIRNAME", "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"
        )
        super().__init__(
            model_path=model_path or default_dir,
            model_size_or_path="whisper-large-v3-turbo-arabic-dialectal-v2-ct2",
        )
