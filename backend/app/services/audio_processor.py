import subprocess
import shutil
import re
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_ffmpeg_binary() -> str:
    """Find an ffmpeg binary that runs on *this* OS (project may be copied between Windows/Linux/macOS)."""
    is_win = sys.platform.startswith("win")
    exe = "ffmpeg.exe" if is_win else "ffmpeg"
    bin_dir = Path(__file__).resolve().parents[2] / "bin"  # backend/bin
    plat = "windows" if is_win else ("macos" if sys.platform == "darwin" else "linux")

    # 1. Bundled per-OS binary: backend/bin/<os>/ffmpeg[.exe], or legacy backend/bin/ffmpeg.exe on Windows
    for cand in (bin_dir / plat / exe, bin_dir / exe):
        if cand.exists() and (is_win or os.access(cand, os.X_OK)):
            return str(cand)

    # 2. System PATH
    system_ffmpeg = shutil.which(exe)
    if system_ffmpeg:
        return system_ffmpeg

    # 3. imageio-ffmpeg (pip wheel ships a binary for each OS)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return "ffmpeg"


async def convert_to_wav(input_path: Path, output_path: Path) -> bool:
    """
    Convert any audio file to 16kHz mono WAV using ffmpeg.
    Handles same-file overwrite by converting to a temp file first.
    """
    logger.info(f"Converting {input_path} to {output_path}")
    try:
        ffmpeg_bin = get_ffmpeg_binary()
        temp_output = output_path.with_suffix(".tmp.wav") if input_path.resolve() == output_path.resolve() else output_path
        
        command = [
            ffmpeg_bin, '-y', '-i', str(input_path),
            '-ar', '16000', '-ac', '1',
            '-c:a', 'pcm_s16le',
            str(temp_output)
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800
        )
        
        if process.returncode == 0:
            if temp_output != output_path:
                shutil.move(str(temp_output), str(output_path))
            return True
        else:
            logger.error(f"FFmpeg failed with code {process.returncode}: {process.stderr.decode('utf-8', errors='ignore')[:300]}")
            return False
            
    except Exception as e:
        logger.error(f"FFmpeg conversion error: {e}")
        return False


def get_audio_duration(path: Path) -> float:
    """
    Get duration of an audio file in seconds using ffmpeg, soundfile, or mutagen.
    """
    # 1. Try ffmpeg directly (parses Duration header from ffmpeg -i)
    ffmpeg_bin = get_ffmpeg_binary()
    try:
        cmd = [ffmpeg_bin, '-i', str(path)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore', timeout=15)
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
