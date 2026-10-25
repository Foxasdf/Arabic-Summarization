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
    speaker_audio_path: Optional[Path] = None,
) -> list:
    """
    Speaker-aware transcription pipeline:
    - Default (ASR_MODE='continuous_vad'):
      1. Performs continuous ASR over long audio contexts using VAD chunking and seam deduplication.
      2. If multiple speaker turns exist:
         - Aligns words acoustically (CTranslate2/Whisper on CUDA/CPU) with relative VAD fallback.
         - Assigns words to speakers using midpoint containment & temporal overlap.
         - Smooths spurious single-word flip-flops and builds natural conversational utterances.
         - Optionally rechecks short word runs against cached speaker embeddings.
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
                turns_res = engine.transcribe_turns(
                    audio_path,
                    turns,
                    on_progress=on_progress,
                    progress_range=progress_range,
                )
                if turns_res:
                    return turns_res
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
    lo, hi = progress_range
    try:
        if on_progress:
            on_progress(lo, "جاري تحميل نموذج التفريغ الصوتي...")
        engine.load()
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
    # Prefer engine-native word timestamps (chunk-coherent, VAD-guided) which
    # preserve Cohere's text verbatim. Cross-model (Whisper) forced alignment
    # is only used as fallback when engine words are missing, because DP
    # re-alignment across different ASR hypotheses degrades timing when the
    # two models disagree on dialectal wording.
    from .alignment import align_timestamps

    has_native_words = getattr(engine, "has_word_timestamps", False)
    engine_words = (asr_res.get("words") or []) if asr_res else []
    words = engine_words if has_native_words else []
    if not words and engine_words and (dur > 1200.0 or len(engine_words) > 3000):
        logger.info(
            f"Using chunk-coherent VAD word timestamps ({len(engine_words)} words, {dur:.1f}s) "
            "directly for long-audio speaker attribution."
        )
        words = engine_words
    # Validate engine words: must cover full text and have sane durations
    if words:
        n_txt = len(text.split())
        if abs(len(words) - n_txt) > max(3, 0.05 * n_txt):
            logger.warning(
                f"Engine word count ({len(words)}) mismatches text words ({n_txt}); "
                "falling back to forced alignment."
            )
            words = []
    if not words:
        from .forced_alignment import align_words_to_audio
        align_lo = asr_hi
        if on_progress:
            on_progress(align_lo, "جاري مطابقة الكلمات زمنياً (Forced Alignment)...")
        words = align_words_to_audio(
            audio_path,
            text,
            active_intervals=turns,
        )
    else:
        align_lo = asr_hi
        if on_progress:
            on_progress(align_lo, "جاري مطابقة الكلمات زمنياً (Forced Alignment)...")

    if on_progress:
        on_progress(hi - 2, "جاري إسناد المتحدثين وتنسيق الجمل...")

    # Reuse the cached Pyannote embedding model to correct short word runs that
    # overlap a neighbouring speaker's continuous utterance. Falls back cleanly
    # when embeddings are unavailable.
    speaker_context = None
    if turns:
        from .diarization import build_speaker_embedding_context

        if on_progress:
            on_progress(hi - 1, "جاري التحقق من بصمات المتحدثين...")
        speaker_context = build_speaker_embedding_context(
            Path(speaker_audio_path or audio_path),
            turns,
        )

    segments = align_timestamps(words, turns, speaker_context=speaker_context)
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

