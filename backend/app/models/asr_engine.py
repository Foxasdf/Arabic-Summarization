"""
ASR Engine Module.
Re-exports modular components from backend.app.models.asr for complete backward compatibility.
"""
from pathlib import Path
from typing import Optional

from .asr.base import BaseASREngine
from .asr.audar_engine import AudarASREngine
from .asr.whisper_engine import WhisperASREngine, WhisperDialectalASREngine
from .asr.cohere_engine import CohereASREngine
from .asr.guardrails import clean_and_guard_asr_text
from .asr import get_asr_engine

# Retain legacy ASREngine alias pointing to AudarASREngine
class ASREngine(AudarASREngine):
    """Legacy alias for AudarASREngine."""
    pass

__all__ = [
    "ASREngine",
    "AudarASREngine",
    "BaseASREngine",
    "WhisperASREngine",
    "WhisperDialectalASREngine",
    "CohereASREngine",
    "get_asr_engine",
    "clean_and_guard_asr_text",
]
