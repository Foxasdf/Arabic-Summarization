"""
Request contracts, data models, and validation logic for Audio Preprocessing (Production v3.1).
"""

from enum import Enum
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class OutputProfile(str, Enum):
    ASR = "asr"
    SPEAKER = "speaker"
    PLAYBACK = "playback"


class DenoiseMode(str, Enum):
    RAW = "raw"
    LIGHT = "light"
    BALANCED = "balanced"
    STRONG = "strong"
    AUTO = "auto"


class NormalizationMode(str, Enum):
    OFF = "off"
    SAFE_GAIN = "safe_gain"
    EBU_R128 = "ebu_r128"


class ChannelStrategy(str, Enum):
    AUTO_BEST = "auto_best"
    DOWNMIX_AVERAGE = "downmix_average"
    CHANNEL_0 = "channel_0"
    CHANNEL_1 = "channel_1"
    PRESERVE = "preserve"


# Track deprecated legacy alias usage
_LEGACY_ALIAS_USAGE: Dict[str, int] = {}

# Canonical mapping for legacy denoise aliases
LEGACY_DENOISE_MAP = {
    # English legacy aliases
    "none": DenoiseMode.RAW,
    "off": DenoiseMode.RAW,
    "raw": DenoiseMode.RAW,
    "low": DenoiseMode.LIGHT,
    "light": DenoiseMode.LIGHT,
    "moderate": DenoiseMode.BALANCED,
    "medium": DenoiseMode.BALANCED,
    "balanced": DenoiseMode.BALANCED,
    "high": DenoiseMode.STRONG,
    "aggressive": DenoiseMode.STRONG,
    "strong": DenoiseMode.STRONG,
    "dynamic": DenoiseMode.AUTO,
    "auto": DenoiseMode.AUTO,
    # Arabic aliases
    "لا معالجة": DenoiseMode.RAW,
    "بدون معالجة": DenoiseMode.RAW,
    "معطل": DenoiseMode.RAW,
    "خام": DenoiseMode.RAW,
    "منخفضة": DenoiseMode.LIGHT,
    "منخفض": DenoiseMode.LIGHT,
    "خفيفة": DenoiseMode.LIGHT,
    "متوسطة": DenoiseMode.BALANCED,
    "متوسط": DenoiseMode.BALANCED,
    "متوازنة": DenoiseMode.BALANCED,
    "عالية": DenoiseMode.STRONG,
    "عالي": DenoiseMode.STRONG,
    "قوية": DenoiseMode.STRONG,
    "ديناميكي": DenoiseMode.AUTO,
    "تلقائي": DenoiseMode.AUTO,
}


def map_legacy_denoise_mode(value: Optional[str]) -> tuple[DenoiseMode, Optional[str]]:
    """
    Map legacy alias to canonical DenoiseMode.
    Returns (canonical_mode, deprecation_warning_or_none).
    """
    if not value:
        return DenoiseMode.RAW, None

    cleaned = str(value).strip().lower()
    if cleaned in (DenoiseMode.RAW.value, DenoiseMode.LIGHT.value, DenoiseMode.BALANCED.value, DenoiseMode.STRONG.value, DenoiseMode.AUTO.value):
        return DenoiseMode(cleaned), None

    if cleaned in LEGACY_DENOISE_MAP:
        mapped = LEGACY_DENOISE_MAP[cleaned]
        _LEGACY_ALIAS_USAGE[cleaned] = _LEGACY_ALIAS_USAGE.get(cleaned, 0) + 1
        warning = f"Legacy denoise alias '{value}' is deprecated and will be removed in v4.0. Please use '{mapped.value}'."
        logger.warning(warning)
        return mapped, warning

    raise ValueError(f"Unknown denoise_mode '{value}'. Supported values: {[m.value for m in DenoiseMode]}")


def get_legacy_alias_usage_stats() -> Dict[str, int]:
    """Retrieve usage count of legacy aliases for telemetry."""
    return dict(_LEGACY_ALIAS_USAGE)


class AudioProcessingRequest(BaseModel):
    """
    Typed, immutable request contract for audio preprocessing.
    """
    output_profile: OutputProfile = Field(
        default=OutputProfile.ASR,
        description="Target profile: 'asr', 'speaker', or 'playback'."
    )
    denoise_mode: DenoiseMode = Field(
        default=DenoiseMode.RAW,
        description="Denoising preset: 'raw', 'light', 'balanced', 'strong', or 'auto'."
    )
    normalization: NormalizationMode = Field(
        default=NormalizationMode.OFF,
        description="Normalization mode: 'off', 'safe_gain' (ASR only), or 'ebu_r128' (playback only)."
    )
    target_sample_rate: Optional[int] = Field(
        default=None,
        description="Explicit sample rate. If None, profile default is used (16000 for asr/speaker, original for playback)."
    )
    target_channels: Optional[int] = Field(
        default=None,
        description="Target channel count. If None, profile default is used (1 for asr/speaker, preserve for playback)."
    )
    channel_strategy: ChannelStrategy = Field(
        default=ChannelStrategy.AUTO_BEST,
        description="Downmix strategy for multi-channel inputs when converting to mono."
    )
    safe_gain_db: float = Field(
        default=0.0,
        ge=0.0,
        le=12.0,
        description="Calibrated static scalar gain (dB) applied when normalization=safe_gain on ASR."
    )
    deprecation_warnings: List[str] = Field(
        default_factory=list,
        description="Recorded deprecation notices for legacy parameters."
    )

    @field_validator("output_profile", mode="before")
    @classmethod
    def validate_profile(cls, v: Any) -> OutputProfile:
        if isinstance(v, OutputProfile):
            return v
        cleaned = str(v).strip().lower()
        try:
            return OutputProfile(cleaned)
        except ValueError:
            raise ValueError(f"Invalid output_profile '{v}'. Must be one of: {[p.value for p in OutputProfile]}")

    @field_validator("denoise_mode", mode="before")
    @classmethod
    def validate_denoise(cls, v: Any) -> DenoiseMode:
        if isinstance(v, DenoiseMode):
            return v
        mode, warn = map_legacy_denoise_mode(str(v))
        return mode

    @field_validator("normalization", mode="before")
    @classmethod
    def validate_normalization(cls, v: Any) -> NormalizationMode:
        if isinstance(v, NormalizationMode):
            return v
        cleaned = str(v).strip().lower()
        try:
            return NormalizationMode(cleaned)
        except ValueError:
            raise ValueError(f"Invalid normalization '{v}'. Must be one of: {[n.value for n in NormalizationMode]}")

    @model_validator(mode="after")
    def validate_invariants(self) -> "AudioProcessingRequest":
        # Invariant 1: safe_gain is ASR-only
        if self.normalization == NormalizationMode.SAFE_GAIN and self.output_profile != OutputProfile.ASR:
            raise ValueError("Normalization 'safe_gain' is only allowed for 'asr' output profile.")

        # Invariant 2: ebu_r128 is Playback-only
        if self.normalization == NormalizationMode.EBU_R128 and self.output_profile != OutputProfile.PLAYBACK:
            raise ValueError("Normalization 'ebu_r128' is only allowed for 'playback' output profile.")

        # Set profile defaults if not specified:
        if self.output_profile == OutputProfile.PLAYBACK and self.normalization == NormalizationMode.OFF:
            # Playback normalization defaults to ebu_r128 unless caller explicitly changed or wants off
            pass

        return self


class AudioValidationResult(BaseModel):
    """
    Result of metadata probe and bounded full decode validation pass.
    """
    is_valid: bool
    error_message: Optional[str] = None
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    codec_name: str = ""
    is_clipped: bool = False
    clipped_samples_count: int = 0
    is_silent: bool = False
    rms_dbfs: float = -100.0
    true_peak_dbtp: float = -100.0
    stereo_correlation: Optional[float] = None
    anti_phase_detected: bool = False
    rumble_detected: bool = False
    rumble_energy_ratio_db: float = -100.0
    snr_proxy_db: Optional[float] = None
    snr_confidence: float = 0.0


# Custom error classes for clean HTTP mapping
class PreprocessingError(Exception):
    """Base error for preprocessing exceptions."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class RequestValidationError422(PreprocessingError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class UnsupportedMediaError415(PreprocessingError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=415, details=details)


class ServiceCapabilityError503(PreprocessingError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=503, details=details)
