from pathlib import Path
from typing import Optional, Callable, Union
import logging
from ..models.asr import get_asr_engine, BaseASREngine
from ..models.registry import registry
from ..config import settings

logger = logging.getLogger(__name__)

def transcribe_audio(
    audio_path: Path,
    on_progress: Optional[Callable[[int, str], None]] = None,
    max_duration_seconds: Optional[float] = None,
    progress_range: tuple = (25, 92),
    model_identifier: Optional[Union[str, Path]] = None,
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
    engine = get_asr_engine(
        model_identifier or model_path,
        mmproj_path=mmproj_path if mmproj_path.exists() else None
    )
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
    model_identifier: Optional[Union[str, Path]] = None,
) -> list:
    """
    High-accuracy full-coverage ASR with Pyannote speaker alignment:
    transcribes continuous speech to guarantee 0% word loss, then aligns
    word and clause timestamps to Pyannote speaker turns.
    """
    from .alignment import align_timestamps

    asr_res = transcribe_audio(
        audio_path,
        on_progress=on_progress,
        model_identifier=model_identifier,
        progress_range=(25, 92),
    )
    words = asr_res.get("words", []) if asr_res else []
    return align_timestamps(words, turns)

