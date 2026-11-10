import os
import subprocess
import shutil
import re
import sys
from pathlib import Path
from typing import Optional, Dict
import logging

from ..config import settings

logger = logging.getLogger(__name__)


def _can_run_ffmpeg(binary_path: str) -> bool:
    try:
        proc = subprocess.run(
            [binary_path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


def get_ffmpeg_binary() -> str:
    """Find an ffmpeg binary that runs on *this* OS (project may be copied between Windows/Linux/macOS)."""
    is_win = sys.platform.startswith("win")
    exe = "ffmpeg.exe" if is_win else "ffmpeg"
    bin_dir = Path(__file__).resolve().parents[2] / "bin"  # backend/bin
    plat = "windows" if is_win else ("macos" if sys.platform == "darwin" else "linux")

    # 1. Bundled per-OS binary: backend/bin/<os>/ffmpeg[.exe], or legacy backend/bin/ffmpeg.exe on Windows
    for cand in (bin_dir / plat / exe, bin_dir / exe):
        if cand.exists() and (is_win or os.access(cand, os.X_OK)):
            if _can_run_ffmpeg(str(cand)):
                return str(cand)

    # 2. System PATH
    system_ffmpeg = shutil.which(exe)
    if system_ffmpeg and _can_run_ffmpeg(system_ffmpeg):
        return system_ffmpeg

    # 3. imageio-ffmpeg (pip wheel ships a binary for each OS)
    try:
        import imageio_ffmpeg
        img_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if img_exe and _can_run_ffmpeg(img_exe):
            return img_exe
    except Exception:
        pass

    return "ffmpeg"


def build_audio_filter_chain(denoise_mode: Optional[str] = None) -> str:
    """
    Build a safe, speech-preserving FFmpeg filter chain.
    CRITICAL: Never applies hard audio gates ('agate') or dynamic range compressors ('speechnorm'),
    as gates cut soft syllables and distant speakers, while speechnorm distorts acoustic embeddings.
    Supported modes: 'none' (default/recommended), 'low', 'moderate', 'high', 'dynamic'.
    """
    raw_mode = (denoise_mode or getattr(settings, "AUDIO_DENOISE_MODE", "none")).lower().strip()
    
    # Map arabic or aliases to canonical keys
    mode_map = {
        "none": "none",
        "off": "none",
        "لا معالجة": "none",
        "بدون معالجة": "none",
        "معطل": "none",
        "low": "low",
        "منخفضة": "low",
        "منخفض": "low",
        "moderate": "moderate",
        "medium": "moderate",
        "متوسطة": "moderate",
        "متوسط": "moderate",
        "high": "high",
        "aggressive": "high",
        "عالية": "high",
        "عالي": "high",
        "dynamic": "dynamic",
        "auto": "dynamic",
        "ديناميكي": "dynamic",
    }
    mode = mode_map.get(raw_mode, "none")
    
    # "none" / "لا معالجة": completely untouched audio, zero filters
    if mode == "none":
        return ""

    if mode == "low":
        # Low noise reduction: Sub-bass rumble cut (< 50 Hz) only, perfectly preserving all speech acoustics
        return "highpass=f=50"
    elif mode in ("moderate", "dynamic"):
        # Moderate / dynamic: Gentle highpass + mild FFT spectral noise reduction without gating
        return "highpass=f=60,lowpass=f=7800,afftdn=nf=-25:nr=6:tn=0"
    elif mode == "high":
        # High noise reduction: Stronger spectral denoise for noisy environments, strictly without gating
        return "highpass=f=75,lowpass=f=7500,afftdn=nf=-20:nr=10:tn=0"
    
    return ""


async def convert_to_wav(
    input_path: Path,
    output_path: Path,
    denoise_mode: Optional[str] = None,
) -> bool:
    """
    Convert any audio file to 16kHz mono WAV using the unified preprocessing engine
    with optional denoising, phase-aware channel downmixing, and transactional writing.
    """
    logger.info(f"Converting and cleaning audio {input_path} -> {output_path} (mode={denoise_mode or 'auto'})")
    try:
        from .preprocessing import (
            PreprocessingEngine,
            AudioProcessingRequest,
            OutputProfile,
            DenoiseMode,
        )

        engine = PreprocessingEngine()
        # Parse denoise mode using contract
        clean_mode = (denoise_mode or "raw").strip().lower()
        try:
            req = AudioProcessingRequest(
                output_profile=OutputProfile.ASR,
                denoise_mode=clean_mode,
            )
        except Exception:
            req = AudioProcessingRequest(
                output_profile=OutputProfile.ASR,
                denoise_mode=DenoiseMode.RAW,
            )

        job_dir = output_path.parent / f".job_{output_path.stem}"
        res = engine.process(input_path, req, job_dir)
        if res.success and "asr" in res.output_files:
            staged_asr = Path(res.output_files["asr"])
            if staged_asr.exists():
                if output_path.resolve() != staged_asr.resolve():
                    shutil.move(str(staged_asr), str(output_path))
                # Cleanup job dir if empty
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                return True

    except Exception as e:
        logger.warning(f"Unified PreprocessingEngine encountered an error: {e}. Attempting direct fallback.")

    # Direct FFmpeg fallback if needed
    try:
        ffmpeg_bin = get_ffmpeg_binary()
        temp_output = output_path.with_suffix(".tmp.wav") if input_path.resolve() == output_path.resolve() else output_path
        filter_chain = build_audio_filter_chain(denoise_mode)
        
        command = [
            ffmpeg_bin, '-y', '-i', str(input_path),
            '-ar', '16000', '-ac', '1',
        ]
        if filter_chain:
            command.extend(['-af', filter_chain])
        command.extend(['-c:a', 'pcm_s16le', str(temp_output)])

        timeout_sec = getattr(settings, "AUDIO_CONVERT_TIMEOUT", 7200)
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec
        )
        
        if process.returncode == 0:
            if temp_output != output_path:
                shutil.move(str(temp_output), str(output_path))
            return True
        else:
            logger.error(f"FFmpeg fallback failed with code {process.returncode}: {process.stderr.decode('utf-8', errors='ignore')[:300]}")
            return False
            
    except Exception as e:
        logger.error(f"FFmpeg conversion error: {e}")
        return False


def process_dual_profile_audio_sync(
    input_path: Path,
    asr_output_path: Path,
    speaker_output_path: Path,
    denoise_mode: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Execute strict profile separation (Production v3.1) synchronously:
    1. ASR Profile: 16kHz mono, phase-aware downmix, conditional rumble highpass, evaluated denoiser preset.
    2. Speaker Profile: 16kHz mono, canonical UNPROCESSED audio (no denoise, no compression, no gain).
    Both profiles are derived independently from the immutable source input_path.
    Returns (success, error_message_if_any).
    """
    try:
        from .preprocessing import (
            PreprocessingEngine,
            AudioProcessingRequest,
            OutputProfile,
            DenoiseMode,
        )

        engine = PreprocessingEngine()

        # Build requests for ASR and Speaker
        clean_mode = (denoise_mode or "raw").strip().lower()
        try:
            asr_req = AudioProcessingRequest(
                output_profile=OutputProfile.ASR,
                denoise_mode=clean_mode,
            )
        except Exception:
            asr_req = AudioProcessingRequest(
                output_profile=OutputProfile.ASR,
                denoise_mode=DenoiseMode.RAW,
            )

        spk_req = AudioProcessingRequest(
            output_profile=OutputProfile.SPEAKER,
            # Speaker defaults to RAW denoise and OFF normalization
        )

        job_dir = asr_output_path.parent / f".dual_{asr_output_path.stem}"
        res = engine.process(input_path, [asr_req, spk_req], job_dir)

        if not res.success:
            return False, "Dual profile preprocessing engine reported failure."

        # Move published artifacts to final target paths
        published_asr = Path(res.output_files.get("asr", ""))
        published_spk = Path(res.output_files.get("speaker", ""))

        if not published_asr.exists() or not published_spk.exists():
            return False, "Published artifact files were not found in staging."

        if asr_output_path.resolve() != published_asr.resolve():
            shutil.move(str(published_asr), str(asr_output_path))

        if speaker_output_path.resolve() != published_spk.resolve():
            shutil.move(str(published_spk), str(speaker_output_path))

        # Copy manifest alongside artifacts
        manifest_src = Path(res.manifest_path)
        manifest_dst = asr_output_path.parent / f"{asr_output_path.stem.replace('.16k', '')}.manifest.json"
        if manifest_src.exists():
            shutil.copy2(str(manifest_src), str(manifest_dst))

        # Cleanup temporary job dir
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

        logger.info(
            f"Dual profile processing complete: ASR={asr_output_path.name}, "
            f"Speaker={speaker_output_path.name}, Manifest={manifest_dst.name}."
        )
        return True, None
    except Exception as e:
        logger.exception("Failed during dual profile preprocessing")
        return False, str(e)


async def process_dual_profile_audio(
    input_path: Path,
    asr_output_path: Path,
    speaker_output_path: Path,
    denoise_mode: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Async wrapper that offloads heavy FFmpeg decoding to a worker thread."""
    import asyncio
    return await asyncio.to_thread(
        process_dual_profile_audio_sync,
        input_path,
        asr_output_path,
        speaker_output_path,
        denoise_mode,
    )


def get_audio_duration(path: Path) -> float:
    """
    Get duration of an audio file in seconds using ffmpeg, soundfile, or mutagen.
    """
    # 1. Try ffmpeg directly (parses Duration header from ffmpeg -i)
    ffmpeg_bin = get_ffmpeg_binary()
    try:
        cmd = [ffmpeg_bin, '-i', str(path)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore', timeout=60)
        # Match 'Duration: 00:00:00.00'
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
        if match:
            hours, minutes, seconds = match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            if duration > 0:
                return round(duration, 2)
    except Exception as e:
        logger.debug(f"ffmpeg duration failed: {e}")

    # 2. Try mutagen
    try:
        import mutagen
        audio = mutagen.File(str(path))
        if audio and audio.info and audio.info.length:
            return round(float(audio.info.length), 2)
    except Exception:
        pass

    # 3. Try soundfile
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return round(float(info.duration), 2)
    except Exception:
        pass

    return 0.0


def estimate_processing_time(duration_seconds: float) -> float:
    """
    Estimate processing time in minutes.
    Returns estimated minutes.
    """
    if duration_seconds <= 0:
        return 1.0
    minutes = (duration_seconds / 60.0) * 0.35
    return max(0.5, round(minutes, 1))


def validate_audio_file(path: Path) -> bool:
    """Check if file exists and has content."""
    return path.exists() and path.stat().st_size > 0


def normalize_pcm_to_float32(data: "np.ndarray") -> "np.ndarray":
    """
    Convert raw audio array of any dtype/channel layout into mono float32 in [-1.0, 1.0].
    CRITICAL: Normalizes integer dtypes (uint8, int16, int32) BEFORE averaging channels
    so stereo int16 arrays do not turn into unscaled float64 in [-32768, 32767] via .mean(axis=1).
    """
    import numpy as np

    arr = np.asarray(data)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    if arr.dtype == np.uint8:
        arr_f32 = (arr.astype(np.float32) - 128.0) / 128.0
    elif arr.dtype == np.int16:
        arr_f32 = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        arr_f32 = arr.astype(np.float32) / 2147483648.0
    elif np.issubdtype(arr.dtype, np.integer):
        max_val = float(max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max))
        arr_f32 = arr.astype(np.float32) / max(1.0, max_val)
    else:
        arr_f32 = arr.astype(np.float32, copy=False)

    if arr_f32.ndim > 1:
        # Handle both (samples, channels) and (channels, samples)
        if arr_f32.shape[0] <= 8 and arr_f32.shape[0] < arr_f32.shape[-1]:
            arr_f32 = arr_f32.mean(axis=0, dtype=np.float32)
        else:
            arr_f32 = arr_f32.mean(axis=1, dtype=np.float32)

    arr_f32 = np.nan_to_num(arr_f32, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(arr_f32, dtype=np.float32)


def load_audio_16k_mono(path: Path, target_sr: int = 16000) -> "np.ndarray":
    """
    Cleanly decode any audio file (.m4a, .mp3, .wav, .flac, etc.) into a
    16 kHz, mono (1D), float32 numpy array in [-1.0, 1.0].
    """
    import numpy as np

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Audio file not found or empty: {path}")

    # Fast path for standard uncompressed .wav files via scipy
    if path.suffix.lower() == ".wav":
        try:
            from scipy.io import wavfile

            sr, raw = wavfile.read(str(path))
            mono = normalize_pcm_to_float32(raw)
            if sr != target_sr and mono.size > 0:
                from scipy.signal import resample_poly
                from math import gcd

                g = gcd(int(sr), int(target_sr))
                mono = resample_poly(mono, int(target_sr) // g, int(sr) // g).astype(np.float32)
            return np.ascontiguousarray(mono, dtype=np.float32)
        except Exception as wav_err:
            logger.debug(f"scipy wavfile read failed for {path.name} ({wav_err}), falling back to ffmpeg.")

    # Universal FFmpeg decode to 16kHz mono float32 PCM (handles .m4a, AAC, MP3, Opus, etc.)
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-v", "error",
        "-i", str(path),
        "-vn",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=getattr(settings, "AUDIO_CONVERT_TIMEOUT", 7200),
    )
    if proc.returncode != 0 or len(proc.stdout) == 0:
        err_msg = proc.stderr.decode("utf-8", errors="ignore").strip()[:400]
        raise RuntimeError(f"FFmpeg failed to decode {path.name} to {target_sr}Hz mono float32: {err_msg}")

    audio_f32 = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(audio_f32, dtype=np.float32)


def ensure_16k_mono_wav(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    target_wav_path: Optional[Path] = None,
) -> Path:
    """
    Ensure an audio file is converted to a verified 16 kHz, mono PCM WAV file.
    If input_path is already a verified 16 kHz mono WAV and output_path is None, returns input_path.
    """
    import numpy as np
    from scipy.io import wavfile

    out = output_path or target_wav_path
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if input_path.suffix.lower() == ".wav" and out is None:
        try:
            sr, data = wavfile.read(str(input_path))
            channels = 1 if data.ndim == 1 else data.shape[1]
            if sr == 16000 and channels == 1 and data.dtype == np.int16:
                return input_path
        except Exception as exc:
            logger.debug(f"WAV verification failed for {input_path.name}: {exc}")

    target_wav = out or input_path.with_name(f"{input_path.stem}_16k.wav")
    target_wav.parent.mkdir(parents=True, exist_ok=True)
    audio_f32 = load_audio_16k_mono(input_path, target_sr=16000)
    pcm16 = np.clip(audio_f32 * 32768.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(str(target_wav), 16000, pcm16)
    return target_wav

