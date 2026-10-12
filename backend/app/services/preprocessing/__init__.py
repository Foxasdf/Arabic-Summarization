"""
Audio Preprocessing Package - Production v3.1
Dual-output architecture with separate ASR, Speaker, and Playback processing pipelines.
"""

from .contracts import (
    OutputProfile,
    DenoiseMode,
    NormalizationMode,
    AudioProcessingRequest,
    AudioValidationResult,
    ChannelStrategy,
)
from .capability import probe_ffmpeg_capabilities, FFmpegCapabilities
from .presets import PRESETS, PRESET_VERSION, resolve_preset, PresetConfig
from .analyzer import AudioAnalyzer, AudioFeatures
from .loudness import LoudnessProcessor, LoudnessResult
from .engine import PreprocessingEngine, ProcessingResult
from .manifest import ManifestManager, ExecutionManifest
from .transaction import TransactionalOutputManager
from .normalizer import ArabicTextNormalizer, calculate_wer_cer

__all__ = [
    "OutputProfile",
    "DenoiseMode",
    "NormalizationMode",
    "AudioProcessingRequest",
    "AudioValidationResult",
    "ChannelStrategy",
    "probe_ffmpeg_capabilities",
    "FFmpegCapabilities",
    "PRESETS",
    "PRESET_VERSION",
    "resolve_preset",
    "PresetConfig",
    "AudioAnalyzer",
    "AudioFeatures",
    "LoudnessProcessor",
    "LoudnessResult",
    "PreprocessingEngine",
    "ProcessingResult",
    "ManifestManager",
    "ExecutionManifest",
    "TransactionalOutputManager",
    "ArabicTextNormalizer",
    "calculate_wer_cer",
]
