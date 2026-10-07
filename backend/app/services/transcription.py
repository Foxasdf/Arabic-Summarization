from pathlib import Path
from typing import Optional, Callable
import logging
from ..models.asr_engine import ASREngine
from ..models.registry import registry
from ..config import settings

logger = logging.getLogger(__name__)

def transcribe_audio(
    audio_path: Path,
    on_progress: Optional[Callable[[int, str], None]] = None,
    max_duration_seconds: Optional[float] = None,
    progress_range: tuple = (25, 92),
) -> dict:
    model_filename = settings.ASR_GGUF_FILENAME
    model_path = settings.MODELS_DIR / model_filename
    if not model_path.exists():
        # Fallback to any available asr model in registry
        for asr_item in registry.get_models_by_type("asr"):
            candidate = settings.MODELS_DIR / asr_item.gguf_filename
            if candidate.exists():
                model_path = candidate
                break

    mmproj_path = settings.MODELS_DIR / settings.ASR_MMPROJ_FILENAME
    engine = ASREngine(model_path, mmproj_path=mmproj_path if mmproj_path.exists() else None)
    try:
        engine.load()
        return engine.transcribe(
            audio_path,
            on_progress=on_progress,
            max_duration_seconds=max_duration_seconds,
            progress_range=progress_range,
        )
    finally:
        engine.unload()

def transcribe_with_speakers(
    audio_path: Path,
    turns: list,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> list:
    """Diarization-driven ASR: one Audar pass per speaker turn."""
    model_path = settings.MODELS_DIR / settings.ASR_GGUF_FILENAME
    mmproj_path = settings.MODELS_DIR / settings.ASR_MMPROJ_FILENAME
    engine = ASREngine(model_path, mmproj_path=mmproj_path if mmproj_path.exists() else None)
    try:
        engine.load()
        return engine.transcribe_turns(audio_path, turns, on_progress=on_progress)
    finally:
        engine.unload()
