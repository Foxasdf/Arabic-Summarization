"""
Audio analysis and full-decode validation module (Production v3.1).
Calculates silence, clipping, stereo phase correlation, rumble, and SNR proxy features.
"""

import sys
import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pydantic import BaseModel, Field

from .contracts import AudioValidationResult, UnsupportedMediaError415
from .capability import find_ffmpeg_binary

logger = logging.getLogger(__name__)


class AudioFeatures(BaseModel):
    duration_seconds: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    codec_name: str = ""
    sample_peak_dbfs: float = -100.0
    true_peak_dbtp: float = -100.0
    rms_dbfs: float = -100.0
    is_silent: bool = False
    is_clipped: bool = False
    clipped_sample_count: int = 0
    clipping_ratio: float = 0.0
    stereo_correlation: Optional[float] = None
    anti_phase_detected: bool = False
    channel_rms_dbfs: List[float] = Field(default_factory=list)
    best_channel_index: int = 0
    rumble_detected: bool = False
    rumble_ratio_db: float = -100.0
    estimated_snr_proxy_db: Optional[float] = None
    snr_confidence: float = 0.0
    speech_ratio: float = 0.0


class AudioAnalyzer:
    """
    Performs metadata probing and bounded full floating-point PCM decode analysis.
    """

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()

    def probe_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """
        Probe stream metadata using FFmpeg/ffprobe.
        """
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return {"error": "File does not exist or is empty"}

        cmd = [self.ffmpeg_bin, "-i", str(audio_path)]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="ignore",
                timeout=60,
            )
            stderr = res.stderr
            # Check for duration
            dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr)
            duration = 0.0
            if dur_match:
                h, m, s = dur_match.groups()
                duration = int(h) * 3600 + int(m) * 60 + float(s)

            # Check audio stream: "Stream #0:0: Audio: aac (LC), 44100 Hz, stereo, fltp, 128 kb/s"
            audio_match = re.search(r"Stream #\d+:\d+.*?: Audio:\s*([a-zA-Z0-9_-]+).*?,\s*(\d+)\s*Hz,\s*([^,]+)", stderr)
            codec = ""
            sample_rate = 0
            channels = 1
            if audio_match:
                codec = audio_match.group(1)
                sample_rate = int(audio_match.group(2))
                layout = audio_match.group(3).strip()
                if "stereo" in layout:
                    channels = 2
                elif "mono" in layout:
                    channels = 1
                elif "5.1" in layout:
                    channels = 6
                elif "7.1" in layout:
                    channels = 8

            return {
                "duration": duration,
                "codec": codec,
                "sample_rate": sample_rate,
                "channels": channels,
                "file_size": audio_path.stat().st_size,
            }
        except Exception as e:
            return {"error": str(e)}

    def decode_and_analyze(
        self,
        audio_path: Path,
        max_duration_sec: float = 7200.0,
    ) -> Tuple[AudioValidationResult, AudioFeatures, Optional[np.ndarray]]:
        """
        Full decode pass into floating-point PCM intermediate.
        Computes clipping, silence, channel correlation, rumble, and SNR proxy.
        """
        if not audio_path.exists():
            val = AudioValidationResult(
                is_valid=False,
                error_message=f"Audio file '{audio_path}' does not exist.",
            )
            return val, AudioFeatures(), None

        file_size = audio_path.stat().st_size
        if file_size == 0:
            val = AudioValidationResult(
                is_valid=False,
                file_size_bytes=0,
                error_message="Uploaded audio file is 0 bytes (empty).",
            )
            return val, AudioFeatures(), None

        meta = self.probe_metadata(audio_path)
        if "error" in meta and not meta.get("codec"):
            val = AudioValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error_message=f"Corrupt audio or missing audio stream: {meta.get('error')}",
            )
            return val, AudioFeatures(), None

        # Decode full audio stream to 32-bit float PCM via stdout
        cmd = [
            self.ffmpeg_bin,
            "-v", "error",
            "-i", str(audio_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-",
        ]

        probe_dur = float(meta.get("duration", 0.0) or 0.0)
        dyn_timeout = max(600, int(probe_dur * 3.0))

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=dyn_timeout,
            )
            if proc.returncode != 0 or len(proc.stdout) == 0:
                err_msg = proc.stderr.decode("utf-8", errors="ignore")[:300] or "Decode failed"
                val = AudioValidationResult(
                    is_valid=False,
                    file_size_bytes=file_size,
                    error_message=f"Audio decode error: {err_msg}",
                )
                return val, AudioFeatures(), None

            raw_bytes = proc.stdout
        except subprocess.TimeoutExpired:
            val = AudioValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error_message="Audio decode timed out.",
            )
            return val, AudioFeatures(), None
        except Exception as e:
            val = AudioValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error_message=f"Decode exception: {e}",
            )
            return val, AudioFeatures(), None

        # Parse float32 audio
        sr = meta.get("sample_rate") or 16000
        n_ch = max(1, meta.get("channels") or 1)
        audio_data = np.frombuffer(raw_bytes, dtype=np.float32)

        total_samples = len(audio_data)
        if total_samples == 0:
            val = AudioValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error_message="Zero audio samples decoded.",
            )
            return val, AudioFeatures(), None

        # Reshape to (samples_per_ch, n_channels) if multi-channel
        rem = total_samples % n_ch
        if rem != 0:
            audio_data = audio_data[:total_samples - rem]

        if n_ch > 1:
            audio_data = audio_data.reshape(-1, n_ch)
        else:
            audio_data = audio_data.reshape(-1, 1)

        duration = len(audio_data) / float(sr)

        # 1. Clipping detection
        # Samples exceeding 0.999 or -0.999
        clip_threshold = 0.999
        clipped_mask = np.abs(audio_data) >= clip_threshold
        clipped_count = int(np.sum(clipped_mask))
        clipping_ratio = clipped_count / float(audio_data.size)
        is_clipped = clipping_ratio > 0.0005  # More than 0.05% clipped

        # 2. Overall RMS and Sample Peak
        sample_peak = float(np.max(np.abs(audio_data)))
        sample_peak_dbfs = 20.0 * np.log10(max(1e-9, sample_peak))

        # True peak estimation (oversampling proxy: 4x max estimate)
        true_peak_dbtp = min(6.0, sample_peak_dbfs + (0.5 if is_clipped else 0.1))

        sq = np.mean(audio_data ** 2)
        rms = np.sqrt(sq) if sq > 0 else 0.0
        rms_dbfs = 20.0 * np.log10(max(1e-9, rms))
        is_silent = rms_dbfs < -65.0

        # 3. Channel analysis & phase correlation
        channel_rms_dbfs = []
        for c in range(n_ch):
            c_sq = np.mean(audio_data[:, c] ** 2)
            c_rms = np.sqrt(c_sq) if c_sq > 0 else 0.0
            channel_rms_dbfs.append(20.0 * np.log10(max(1e-9, c_rms)))

        best_channel_idx = int(np.argmax(channel_rms_dbfs))

        stereo_correlation: Optional[float] = None
        anti_phase = False
        if n_ch == 2:
            left = audio_data[:, 0]
            right = audio_data[:, 1]
            # Pearson correlation
            l_std = np.std(left)
            r_std = np.std(right)
            if l_std > 1e-6 and r_std > 1e-6:
                corr = np.corrcoef(left, right)[0, 1]
                stereo_correlation = float(corr)
                if stereo_correlation < -0.2:
                    anti_phase = True
                    logger.warning(
                        f"Anti-phase stereo detected! (corr={stereo_correlation:.2f}). "
                        f"Standard averaging would cause destructive cancellation."
                    )

        # 4. Rumble detection (< 50 Hz sub-bass energy ratio)
        rumble_detected = False
        rumble_ratio_db = -100.0
        if len(audio_data) >= sr and not is_silent:
            # Measure low-frequency content via FFT or simple slice
            mono_mix = np.mean(audio_data, axis=1)
            # Take representative 5-second slice
            chunk_len = min(len(mono_mix), sr * 5)
            mono_chunk = mono_mix[:chunk_len]
            fft_mag = np.abs(np.fft.rfft(mono_chunk))
            freqs = np.fft.rfftfreq(chunk_len, d=1.0 / sr)

            sub50_mask = (freqs <= 50.0)
            speech_mask = (freqs > 50.0) & (freqs <= 4000.0)

            sub50_energy = np.sum(fft_mag[sub50_mask] ** 2) + 1e-12
            speech_energy = np.sum(fft_mag[speech_mask] ** 2) + 1e-12

            ratio = sub50_energy / speech_energy
            rumble_ratio_db = 10.0 * np.log10(ratio)
            # If sub-50Hz energy is disproportionately high (> -10 dB relative to speech)
            if rumble_ratio_db > -10.0:
                rumble_detected = True

        # 5. SNR Proxy estimation using multi-window non-speech windows
        snr_proxy_db, snr_conf, speech_ratio = self._estimate_snr_proxy(
            audio_data, sr, is_silent, duration
        )

        features = AudioFeatures(
            duration_seconds=round(duration, 3),
            sample_rate=sr,
            channels=n_ch,
            codec_name=meta.get("codec", "pcm"),
            sample_peak_dbfs=round(sample_peak_dbfs, 2),
            true_peak_dbtp=round(true_peak_dbtp, 2),
            rms_dbfs=round(rms_dbfs, 2),
            is_silent=is_silent,
            is_clipped=is_clipped,
            clipped_sample_count=clipped_count,
            clipping_ratio=round(clipping_ratio, 5),
            stereo_correlation=round(stereo_correlation, 3) if stereo_correlation is not None else None,
            anti_phase_detected=anti_phase,
            channel_rms_dbfs=[round(x, 2) for x in channel_rms_dbfs],
            best_channel_index=best_channel_idx,
            rumble_detected=rumble_detected,
            rumble_ratio_db=round(rumble_ratio_db, 2),
            estimated_snr_proxy_db=round(snr_proxy_db, 2) if snr_proxy_db is not None else None,
            snr_confidence=round(snr_conf, 2),
            speech_ratio=round(speech_ratio, 2),
        )

        val_result = AudioValidationResult(
            is_valid=True,
            file_size_bytes=file_size,
            duration_seconds=features.duration_seconds,
            sample_rate=sr,
            channels=n_ch,
            codec_name=features.codec_name,
            is_clipped=is_clipped,
            clipped_samples_count=clipped_count,
            is_silent=is_silent,
            rms_dbfs=features.rms_dbfs,
            true_peak_dbtp=features.true_peak_dbtp,
            stereo_correlation=features.stereo_correlation,
            anti_phase_detected=anti_phase,
            rumble_detected=rumble_detected,
            rumble_energy_ratio_db=features.rumble_ratio_db,
            snr_proxy_db=features.estimated_snr_proxy_db,
            snr_confidence=features.snr_confidence,
        )

        return val_result, features, audio_data

    def _estimate_snr_proxy(
        self,
        audio_data: np.ndarray,
        sr: int,
        is_silent: bool,
        duration: float,
    ) -> Tuple[Optional[float], float, float]:
        """
        Estimate SNR proxy using multiple non-speech windows and speech-active regions.
        Labels result as SNR proxy, not true SNR.
        """
        if is_silent or duration < 0.2:
            return None, 0.0, 0.0

        mono = np.mean(audio_data, axis=1)
        frame_len = int(0.05 * sr)  # 50ms frames
        n_frames = len(mono) // frame_len
        if n_frames < 4:
            return None, 0.1, 0.0

        frames = mono[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        frame_db = 20.0 * np.log10(frame_rms)

        # Distinguish speech vs noise windows
        # Speech threshold: > 60th percentile or > -40 dB
        p20 = np.percentile(frame_db, 20)
        p80 = np.percentile(frame_db, 80)

        # Robust noise estimation: take 15th to 45th percentile frames (avoiding lowest 10% digital dropouts)
        noise_frames = frame_db[(frame_db >= np.percentile(frame_db, 15)) & (frame_db <= np.percentile(frame_db, 45))]
        speech_frames = frame_db[frame_db >= np.percentile(frame_db, 70)]

        if len(noise_frames) == 0 or len(speech_frames) == 0:
            return None, 0.1, 0.0

        noise_rms_db = float(np.mean(noise_frames))
        speech_rms_db = float(np.mean(speech_frames))

        snr_proxy = speech_rms_db - noise_rms_db
        speech_ratio = len(speech_frames) / float(n_frames)

        # Confidence calculation
        # Clips < 5 seconds get lower confidence
        duration_factor = min(1.0, duration / 10.0)
        separation_factor = min(1.0, max(0.0, (p80 - p20) / 20.0))
        confidence = float(duration_factor * 0.6 + separation_factor * 0.4)

        if duration < 5.0:
            confidence = min(0.4, confidence)

        return max(0.0, snr_proxy), confidence, speech_ratio
