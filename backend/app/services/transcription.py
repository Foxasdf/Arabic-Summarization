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
    progress_range: tuple = (25, 95),
) -> list:
    """
    Speaker-aware transcription pipeline:
    - Default (ASR_MODE='continuous_vad'):
      1. Performs continuous ASR over long audio contexts using VAD chunking and seam deduplication.
      2. If multiple speaker turns exist:
         - Aligns words acoustically (CTranslate2/Whisper on CUDA/CPU) with relative VAD fallback.
         - Assigns words to speakers using midpoint containment & temporal overlap.
         - Smooths spurious single-word flip-flops and builds natural conversational utterances.
      3. If no speaker turns exist, returns single-speaker utterance.
    - Legacy (ASR_MODE='per_turn'):
      Calls engine.transcribe_turns for ablation and backward compatibility.
    """
    model_path = model_identifier or settings.ASR_GGUF_FILENAME
    engine = get_asr_engine(model_path)

    # Check ASR_MODE setting (default: "continuous_vad")
    asr_mode = getattr(settings, "ASR_MODE", "continuous_vad").lower().strip()

    if asr_mode == "per_turn":
        try:
            engine.load()
            if turns:
                return engine.transcribe_turns(
                    audio_path,
                    turns,
                    on_progress=on_progress,
                    progress_range=progress_range,
                )
            else:
                asr_res = engine.transcribe(
                    audio_path,
                    on_progress=on_progress,
                    progress_range=progress_range,
                )
                text = asr_res.get("text", "").strip() if asr_res else ""
                from .audio_processor import get_audio_duration
                dur = get_audio_duration(audio_path)
                return [{
                    "speaker": "SPEAKER_00",
                    "text": text,
                    "start": 0.0,
                    "end": round(dur, 2),
                }] if text else []
        finally:
            engine.unload()

    # Continuous VAD pipeline (Production default)
    try:
        engine.load()
        lo, hi = progress_range
        asr_hi = lo + int((hi - lo) * 0.85)
        asr_res = engine.transcribe(
            audio_path,
            on_progress=on_progress,
            progress_range=(lo, asr_hi),
        )
    finally:
        engine.unload()

    text = asr_res.get("text", "").strip() if asr_res else ""
    if not text:
        return []

    from .audio_processor import get_audio_duration
    dur = get_audio_duration(audio_path)

    if not turns:
        return [{
            "speaker": "SPEAKER_00",
            "text": text,
            "start": 0.0,
            "end": round(dur, 2),
        }]

    # Phase 3 & 4: Word-Level Forced Alignment and Speaker Attribution
    from .forced_alignment import align_words_to_audio
    from .alignment import align_timestamps

    align_lo = asr_hi
    if on_progress:
        on_progress(align_lo, "جاري مطابقة الكلمات زمنياً (Forced Alignment)...")

    words = align_words_to_audio(
        audio_path,
        text,
        active_intervals=turns,
    )
    if not words and asr_res.get("words"):
        words = asr_res.get("words")

    if on_progress:
        on_progress(hi - 2, "جاري إسناد المتحدثين وتنسيق الجمل...")

    segments = align_timestamps(words, turns)
    if not segments and text:
        segments = [{
            "speaker": turns[0].get("speaker", "SPEAKER_00") if turns else "SPEAKER_00",
            "text": text,
            "start": 0.0,
            "end": round(dur, 2),
        }]

    if on_progress:
        on_progress(hi, "اكتمل التفريغ وتنسيق المتحدثين بنجاح.")

    return segments

