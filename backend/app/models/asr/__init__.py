"""
ASR Module - Pluggable Speech-to-Text Engines.
Provides unified factory for loading Audar ASR, Cohere Transcribe Arabic,
Faster-Whisper (standard & dialectal), and future speech models.
"""
from pathlib import Path
from typing import Optional, Union

from .base import BaseASREngine
from .audar_engine import AudarASREngine
from .whisper_engine import WhisperASREngine, WhisperDialectalASREngine
from .cohere_engine import CohereASREngine
from .guardrails import clean_and_guard_asr_text
from ...config import settings


def get_asr_engine(
    model_identifier: Optional[Union[str, Path]] = None,
    mmproj_path: Optional[Path] = None,
) -> BaseASREngine:
    """
    Factory function to instantiate the requested ASR engine.
    Supports:
      - 'audar' / default -> AudarASREngine (GGUF + MTMD)
      - 'cohere' -> CohereASREngine (cohere-transcribe-arabic-07-2026 GGUF via transcribe.cpp)
      - 'dialectal' / 'oddadmix' -> WhisperDialectalASREngine (CTranslate2 int8_float16)
      - 'whisper' -> WhisperASREngine (CTranslate2)
    """
    ident_str = str(model_identifier or settings.ASR_MODEL_NAME).lower()

    # 1. Cohere Transcribe Arabic
    if "cohere" in ident_str:
        if isinstance(model_identifier, Path) and model_identifier.exists():
            return CohereASREngine(model_path=model_identifier)
        cohere_path = settings.MODELS_DIR / getattr(
            settings, "COHERE_ASR_FILENAME", "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"
        )
        return CohereASREngine(model_path=cohere_path)

    # 2. Whisper Dialectal
    if "dialectal" in ident_str or "oddadmix" in ident_str:
        if isinstance(model_identifier, Path) and model_identifier.exists():
            return WhisperDialectalASREngine(model_path=model_identifier)
        dialectal_path = settings.MODELS_DIR / getattr(
            settings, "WHISPER_DIALECTAL_DIRNAME", "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"
        )
        return WhisperDialectalASREngine(model_path=dialectal_path)

    # 3. Standard Whisper
    if "whisper" in ident_str:
        return WhisperASREngine(
            model_path=Path(model_identifier) if isinstance(model_identifier, Path) else None
        )

    # 4. Audar ASR (explicit key) — must use its own GGUF, NOT ASR_GGUF_FILENAME
    # (which now points to Cohere as the default ASR).
    if "audar" in ident_str or "flash" in ident_str:
        audar_path = settings.MODELS_DIR / getattr(
            settings, "AUDAR_GGUF_FILENAME", "Audar-ASR-V1-Turbo-Q8_0.gguf"
        )
        if isinstance(model_identifier, Path) and model_identifier.exists():
            audar_path = model_identifier
        return AudarASREngine(model_path=audar_path, mmproj_path=mmproj_path)

    # 5. Default to Audar ASR engine
    if isinstance(model_identifier, Path) and model_identifier.exists():
        model_path = model_identifier
    else:
        model_path = settings.MODELS_DIR / getattr(
            settings, "AUDAR_GGUF_FILENAME", "Audar-ASR-V1-Turbo-Q8_0.gguf"
        )

    return AudarASREngine(model_path=model_path, mmproj_path=mmproj_path)


__all__ = [
    "BaseASREngine",
    "AudarASREngine",
    "WhisperASREngine",
    "WhisperDialectalASREngine",
    "CohereASREngine",
    "get_asr_engine",
    "clean_and_guard_asr_text",
]
