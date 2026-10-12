"""
Immutable, versioned preset definitions and resolver (Production v3.1).
Freezes backend, parameters, and fallbacks for light, balanced, and strong denoisers.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from .contracts import DenoiseMode, ServiceCapabilityError503
from .capability import FFmpegCapabilities

PRESET_VERSION = "3.1.0"


class PresetConfig(BaseModel):
    """
    Immutable specification of an audio processing preset.
    """
    preset_id: str
    version: str = PRESET_VERSION
    name: str
    backend: str  # bypass, afftdn, arnndn, anlmdn
    filter_string: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    supported_sample_rates: List[int] = Field(default_factory=lambda: [8000, 16000, 22050, 44100, 48000])
    supported_channels: List[int] = Field(default_factory=lambda: [1, 2])
    model_name: Optional[str] = None
    model_hash: Optional[str] = None
    declared_fallback: Optional[str] = None
    description: str = ""


# Immutable frozen presets
PRESETS: Dict[str, PresetConfig] = {
    "raw": PresetConfig(
        preset_id="raw",
        name="Raw (Bypass)",
        backend="bypass",
        filter_string="",
        parameters={},
        declared_fallback=None,
        description="Bypass denoiser: bit-perfect pass-through of speech harmonics.",
    ),
    "light": PresetConfig(
        preset_id="light",
        name="Light FFT Denoise",
        backend="afftdn",
        filter_string="afftdn=nf=-30:nr=6:tn=0",
        parameters={"nf": -30, "nr": 6, "tn": 0},
        declared_fallback="raw",
        description="Gentle spectral denoiser (6dB reduction) preserving delicate consonants.",
    ),
    "balanced": PresetConfig(
        preset_id="balanced",
        name="Balanced FFT Denoise",
        backend="afftdn",
        filter_string="afftdn=nf=-25:nr=10:tn=0",
        parameters={"nf": -25, "nr": 10, "tn": 0},
        declared_fallback="light",
        description="Balanced spectral noise reduction (10dB reduction) without gating.",
    ),
    "strong": PresetConfig(
        preset_id="strong",
        name="Strong FFT Denoise",
        backend="afftdn",
        filter_string="afftdn=nf=-20:nr=15:tn=0",
        parameters={"nf": -20, "nr": 15, "tn": 0},
        declared_fallback="balanced",
        description="Strong spectral noise reduction (15dB reduction) for high-noise audio.",
    ),
    "anlmdn_candidate": PresetConfig(
        preset_id="anlmdn_candidate",
        name="Non-Local Means Candidate",
        backend="anlmdn",
        filter_string="anlmdn=s=7:p=0.002:r=0.006",
        parameters={"s": 7, "p": 0.002, "r": 0.006},
        declared_fallback="balanced",
        description="Candidate broadband reduction using non-local means.",
    ),
    "arnndn_speech": PresetConfig(
        preset_id="arnndn_speech",
        name="Recurrent Neural Network Denoise",
        backend="arnndn",
        filter_string="",  # populated with model path if available
        parameters={},
        model_name="rnnoise_bd",
        declared_fallback="balanced",
        description="Speech noise reduction using RNN model.",
    ),
}


def resolve_preset(
    mode: DenoiseMode,
    capabilities: Optional[FFmpegCapabilities] = None,
    allow_fallback: bool = True,
    features: Optional[Any] = None,
) -> tuple[PresetConfig, Optional[str]]:
    """
    Resolve requested DenoiseMode to a fixed PresetConfig, checking backend capability.
    Returns (resolved_preset, fallback_reason_if_any).
    Raises ServiceCapabilityError503 if required backend is unavailable and no approved fallback exists.
    """
    if mode == DenoiseMode.AUTO:
        preset, _, fallback_reason = resolve_auto_preset(
            features=features,
            capabilities=capabilities,
            allow_fallback=allow_fallback,
        )
        return preset, fallback_reason

    preset_key = mode.value
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown preset key '{preset_key}'")

    target = PRESETS[preset_key]

    # Bypass is always supported
    if target.backend == "bypass":
        return target, None

    if capabilities is None:
        return target, None

    # Check if backend filter passed live smoke test
    is_ready = capabilities.is_filter_ready(target.backend)
    if is_ready:
        return target, None

    # Backend is unavailable
    if allow_fallback and target.declared_fallback:
        fallback_target = PRESETS[target.declared_fallback]
        reason = (
            f"Backend '{target.backend}' for preset '{target.preset_id}' failed live verification; "
            f"declared fallback to '{fallback_target.preset_id}' engaged."
        )
        return fallback_target, reason

    raise ServiceCapabilityError503(
        f"Required preprocessing backend '{target.backend}' for preset '{target.preset_id}' is unavailable.",
        details={"preset": target.preset_id, "backend": target.backend},
    )


def resolve_auto_preset(
    features: Optional[Any] = None,
    capabilities: Optional[FFmpegCapabilities] = None,
    allow_fallback: bool = True,
    shadow_mode: bool = False,
) -> tuple[PresetConfig, Dict[str, Any], Optional[str]]:
    """
    Calibrated automatic preset selector (Phase 5).
    Selects exclusively among validated, immutable presets (raw, light, balanced, strong).
    Does NOT dynamically modify filter parameters.
    """
    dur = getattr(features, "duration_seconds", 0.0) if features else 0.0
    snr = getattr(features, "estimated_snr_proxy_db", None) if features else None
    conf = getattr(features, "snr_confidence", 0.0) if features else 0.0
    is_silent = getattr(features, "is_silent", False) if features else False

    decision_meta = {
        "duration_seconds": dur,
        "estimated_snr_proxy_db": snr,
        "confidence": conf,
        "shadow_mode": shadow_mode,
        "selected_preset": "raw",
        "rationale": "",
    }

    # Rule 1: Silence
    if is_silent:
        decision_meta["selected_preset"] = "raw"
        decision_meta["rationale"] = "Audio is silent; bypassing denoiser."
        preset = PRESETS["raw"]
        return preset, decision_meta, None

    # Rule 2: Clips under 5 seconds or low confidence (< 0.40)
    # Per spec: For clips under five seconds, avoid aggressive SNR-based selection; use conservative preset or bypass
    if dur < 5.0 or conf < 0.40 or snr is None:
        decision_meta["selected_preset"] = "light"
        decision_meta["rationale"] = (
            f"Short clip ({dur:.1f}s < 5.0s) or low confidence ({conf:.2f}); "
            f"selecting conservative preset 'light' to protect speech transients."
        )
        preset = PRESETS["light"]
        # Check backend readiness
        if capabilities and not capabilities.is_filter_ready(preset.backend):
            if allow_fallback and preset.declared_fallback:
                return PRESETS[preset.declared_fallback], decision_meta, f"Fallback from {preset.preset_id} to {preset.declared_fallback}"
        return preset, decision_meta, None

    # Rule 3: High SNR (>= 22 dB) -> Clean speech
    if snr >= 22.0:
        decision_meta["selected_preset"] = "raw"
        decision_meta["rationale"] = f"Clean speech detected (SNR proxy {snr:.1f} dB >= 22.0 dB); raw bypass selected."
        preset = PRESETS["raw"]
        return preset, decision_meta, None

    # Rule 4: Moderate noise (12.0 dB <= SNR < 22.0 dB) -> Balanced preset
    if snr >= 12.0:
        decision_meta["selected_preset"] = "balanced"
        decision_meta["rationale"] = f"Moderate background noise (SNR proxy {snr:.1f} dB); balanced preset selected."
        preset = PRESETS["balanced"]
        if capabilities and not capabilities.is_filter_ready(preset.backend):
            if allow_fallback and preset.declared_fallback:
                return PRESETS[preset.declared_fallback], decision_meta, f"Fallback from {preset.preset_id} to {preset.declared_fallback}"
        return preset, decision_meta, None

    # Rule 5: Heavy noise (< 12.0 dB) -> Strong preset
    decision_meta["selected_preset"] = "strong"
    decision_meta["rationale"] = f"High noise environment (SNR proxy {snr:.1f} dB < 12.0 dB); strong preset selected."
    preset = PRESETS["strong"]
    if capabilities and not capabilities.is_filter_ready(preset.backend):
        if allow_fallback and preset.declared_fallback:
            return PRESETS[preset.declared_fallback], decision_meta, f"Fallback from {preset.preset_id} to {preset.declared_fallback}"
    return preset, decision_meta, None

