"""
Voice Activity Detection (VAD) & Acoustic Energy Gate.
Prevents feeding pure silence, ambient noise, or low-level static
into generative ASR models (which would otherwise cause severe hallucinations).
"""
import logging
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


def compute_audio_rms_db(audio_data: np.ndarray) -> float:
    """Compute Root Mean Square (RMS) energy in dBFS from audio samples."""
    if audio_data is None or len(audio_data) == 0:
        return -120.0

    from .audio_processor import normalize_pcm_to_float32

    data_float = normalize_pcm_to_float32(audio_data)
    if len(data_float) == 0:
        return -120.0

    mean_square = float(np.mean(data_float ** 2))
    rms = np.sqrt(mean_square)
    return float(20.0 * np.log10(rms + 1e-9))


def is_speech_present(
    audio_input: Union[Path, str, np.ndarray],
    sr: int = 16000,
    min_rms_db: Optional[float] = None,
    min_peak: float = 0.035,
) -> Tuple[bool, float]:
    """
    Lightning-fast Acoustic Energy & Silence Gate.
    
    Args:
        audio_input: File path (WAV) or numpy array of audio samples.
        sr: Sample rate (default 16000).
        min_rms_db: Minimum RMS dBFS threshold to consider speech (default from settings: -42.0 dBFS).
        min_peak: Minimum peak amplitude threshold (to safeguard against quiet whispers).
        
    Returns:
        (has_speech: bool, rms_db: float)
    """
    if not getattr(settings, "VAD_ENABLED", True):
        return True, 0.0

    threshold_db = min_rms_db if min_rms_db is not None else getattr(settings, "VAD_MIN_RMS_DB", -42.0)

    try:
        from .audio_processor import normalize_pcm_to_float32, load_audio_16k_mono

        if isinstance(audio_input, (str, Path)):
            path = Path(audio_input)
            if not path.exists() or path.stat().st_size < 100:
                return False, -120.0
            if path.suffix.lower() == ".wav":
                from scipy.io import wavfile
                _, raw = wavfile.read(str(path))
                data_float = normalize_pcm_to_float32(raw)
            else:
                data_float = load_audio_16k_mono(path, target_sr=sr)
        else:
            data_float = normalize_pcm_to_float32(audio_input)

        if len(data_float) == 0:
            return False, -120.0

        rms_db = compute_audio_rms_db(data_float)
        peak_val = float(np.max(np.abs(data_float)))

        # If peak is above min_peak (e.g. whispered word with high dynamic range), allow it
        if rms_db >= threshold_db or peak_val >= min_peak:
            return True, rms_db

        logger.debug(f"VAD: Silence/Noise rejected (RMS: {rms_db:.1f} dBFS < {threshold_db} dBFS, Peak: {peak_val:.3f})")
        return False, rms_db

    except Exception as e:
        logger.warning(f"VAD energy check failed ({e}); defaulting to accepting slice.")
        return True, 0.0

