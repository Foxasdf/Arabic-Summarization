"""
Forced Alignment Service (Phase 3).
Aligns target ASR words to audio to produce precise word-level timestamps [start, end].
Uses acoustic alignment via CTranslate2 / Whisper when available, with a robust
relative VAD-guided temporal distribution fallback.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..config import settings
from .audio_processor import get_audio_duration

logger = logging.getLogger(__name__)

# Ensure CUDA runtime DLLs (cublas64_12.dll) from torch/lib are reachable on Windows
for _base in [
    Path(__file__).resolve().parent.parent.parent / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
    Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
    Path(__file__).resolve().parent.parent.parent / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib",
]:
    if _base.exists():
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_base))
            except Exception:
                pass
        if str(_base) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_base) + ";" + os.environ.get("PATH", "")


def _normalize_token(w: str) -> str:
    """Normalize Arabic word token for alignment comparison."""
    import re
    w = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", w)
    w = re.sub(r"[إأآٱ]", "ا", w)
    w = re.sub(r"ة", "ه", w)
    w = re.sub(r"ى", "ي", w)
    w = re.sub(r"[،,.\?!؟;:\"\'\-\(\)\[\]«»]", "", w).strip()
    return w


def _align_tokens_dp(target_words: List[str], ref_words: List[Any]) -> List[Tuple[float, float]]:
    """
    Monotonic dynamic programming alignment between target tokens and acoustic reference tokens.
    Returns a list of (start, end) timestamp tuples for target_words.
    Uses banded Sakoe-Chiba constraint when sequences are long to prevent O(N*M) quadratic CPU slowdown.
    """
    N = len(target_words)
    M = len(ref_words)
    if N == 0:
        return []
    if M == 0:
        return []
    if (N + 1) * (M + 1) > 25_000_000:
        logger.warning(f"DP alignment matrix too large ({N}x{M}), falling back to relative VAD alignment.")
        return []

    t_norm = [_normalize_token(w) for w in target_words]
    r_norm = [_normalize_token(getattr(w, "word", str(w))) for w in ref_words]

    use_band = max(N, M) > 300
    window = max(100, int(0.06 * max(N, M))) if use_band else max(N, M)
    scale = float(M) / float(max(1, N))

    dp = np.full((N + 1, M + 1), float('inf'), dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, min(window + 1 if use_band else N + 1, N + 1)):
        dp[i, 0] = float(i)
    for j in range(1, min(window + 1 if use_band else M + 1, M + 1)):
        dp[0, j] = float(j)

    for i in range(1, N + 1):
        if use_band:
            center_j = int(i * scale)
            j_min = max(1, center_j - window)
            j_max = min(M, center_j + window)
        else:
            j_min = 1
            j_max = M

        t_w = t_norm[i - 1]
        for j in range(j_min, j_max + 1):
            r_w = r_norm[j - 1]
            cost = 0.0 if t_w == r_w else 1.0
            if cost > 0 and (t_w.startswith(r_w) or r_w.startswith(t_w)):
                cost = 0.3
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost,
                dp[i - 1, j] + 0.9,
                dp[i, j - 1] + 0.9,
            )

    # Backtrack alignment
    i, j = N, M
    aligned_times: List[Optional[Tuple[float, float]]] = [None] * N
    while i > 0 and j > 0:
        cost = 0.0 if t_norm[i - 1] == r_norm[j - 1] else 1.0
        if cost > 0 and (t_norm[i - 1].startswith(r_norm[j - 1]) or r_norm[j - 1].startswith(t_norm[i - 1])):
            cost = 0.3
        diag = dp[i - 1, j - 1] + cost
        up = dp[i - 1, j] + 0.9
        left = dp[i, j - 1] + 0.9
        m_val = min(diag, up, left)
        if m_val == diag:
            ref_w = ref_words[j - 1]
            st = float(getattr(ref_w, "start", 0.0))
            en = float(getattr(ref_w, "end", st + 0.3))
            aligned_times[i - 1] = (st, en)
            i -= 1
            j -= 1
        elif m_val == up:
            i -= 1
        else:
            j -= 1

    # Interpolate unaligned words
    last_end = 0.0
    for k in range(N):
        if aligned_times[k] is None:
            next_k = next((x for x in range(k + 1, N) if aligned_times[x] is not None), None)
            if next_k is not None:
                n_st = aligned_times[next_k][0]
                step = max(0.05, (n_st - last_end) / max(1, next_k - k + 1))
                aligned_times[k] = (round(last_end, 2), round(last_end + step, 2))
                last_end = aligned_times[k][1]
            else:
                aligned_times[k] = (round(last_end, 2), round(last_end + 0.4, 2))
                last_end = aligned_times[k][1]
        else:
            last_end = aligned_times[k][1]

    return [t for t in aligned_times if t is not None]


def fallback_relative_alignment(
    words: List[str],
    total_duration: float,
    active_intervals: Optional[List[Dict[str, float]]] = None
) -> List[Dict[str, Any]]:
    """
    Robust fallback: distributes words across known active speech intervals
    or uniformly across the duration if no intervals are provided.
    """
    N = len(words)
    if N == 0:
        return []

    if not active_intervals:
        word_dur = total_duration / max(1, N)
        return [
            {
                "word": w,
                "start": round(i * word_dur, 2),
                "end": round((i + 1) * word_dur, 2),
            }
            for i, w in enumerate(words)
        ]

    # Calculate active speech time
    active_dur = sum(iv["end"] - iv["start"] for iv in active_intervals)
    if active_dur <= 0.1:
        active_dur = total_duration
        active_intervals = [{"start": 0.0, "end": total_duration}]

    # Assign words proportionally across intervals
    items = []
    w_idx = 0
    for iv in active_intervals:
        span_dur = iv["end"] - iv["start"]
        share = span_dur / active_dur
        n_in_span = max(1, int(round(share * N))) if w_idx < N else 0
        if n_in_span > 0:
            sub_words = words[w_idx:min(N, w_idx + n_in_span)]
            step = span_dur / max(1, len(sub_words))
            for s_i, sw in enumerate(sub_words):
                items.append({
                    "word": sw,
                    "start": round(iv["start"] + s_i * step, 2),
                    "end": round(iv["start"] + (s_i + 1) * step, 2),
                })
            w_idx += len(sub_words)

    # Any remaining words
    if w_idx < N:
        rem_words = words[w_idx:]
        last_t = items[-1]["end"] if items else 0.0
        step = max(0.2, (total_duration - last_t) / max(1, len(rem_words)))
        for s_i, sw in enumerate(rem_words):
            items.append({
                "word": sw,
                "start": round(last_t + s_i * step, 2),
                "end": round(last_t + (s_i + 1) * step, 2),
            })

    return items


def align_words_to_audio(
    audio_path: Path,
    transcript_text: str,
    max_duration_seconds: Optional[float] = None,
    active_intervals: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Align target transcript words to acoustic timestamps.
    Attempts GPU/CPU acoustic alignment, seamlessly falling back to relative VAD alignment.
    """
    words = transcript_text.strip().split()
    if not words:
        return []

    total_dur = get_audio_duration(audio_path)
    if max_duration_seconds and total_dur > max_duration_seconds:
        total_dur = max_duration_seconds

    whisper_dir = settings.MODELS_DIR / getattr(
        settings, "WHISPER_DIALECTAL_DIRNAME", "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"
    )

    if whisper_dir.exists():
        ref_words = []
        for device, compute_type in [("cuda", "float16"), ("cpu", "int8")]:
            try:
                import faster_whisper
                import scipy.io.wavfile

                if device == "cuda":
                    import ctranslate2
                    if ctranslate2.get_cuda_device_count() == 0:
                        continue

                audio_f32 = None
                sr = 16000
                try:
                    sr_read, raw = scipy.io.wavfile.read(str(audio_path))
                    if raw.ndim > 1:
                        raw = raw.mean(axis=1)
                    if sr_read == 16000:
                        audio_f32 = (raw.astype(np.float32) / 32768.0).copy()
                except Exception:
                    pass

                if audio_f32 is None:
                    from .audio_processor import get_ffmpeg_binary
                    import subprocess, tempfile
                    ffmpeg_bin = get_ffmpeg_binary()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                        tmp_wav_path = Path(tmp_f.name)
                    cmd = [
                        ffmpeg_bin, "-y", "-loglevel", "error",
                        "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                        str(tmp_wav_path)
                    ]
                    subprocess.run(cmd, check=True)
                    _, raw = scipy.io.wavfile.read(str(tmp_wav_path))
                    tmp_wav_path.unlink(missing_ok=True)
                    audio_f32 = (raw.astype(np.float32) / 32768.0).copy()

                if max_duration_seconds:
                    max_samples = int(max_duration_seconds * sr)
                    audio_f32 = audio_f32[:max_samples]

                logger.info(f"Running acoustic word timestamp extraction via {whisper_dir.name} on {device.upper()}...")
                model = faster_whisper.WhisperModel(str(whisper_dir), device=device, compute_type=compute_type)
                segs, _ = model.transcribe(audio_f32, language="ar", word_timestamps=True)
                ref_words = [w for s in segs for w in (s.words or [])]
                if ref_words:
                    break
            except Exception as e:
                logger.warning(f"Acoustic alignment on {device.upper()} failed ({e}); trying fallback.")
                ref_words = []
                continue
        try:
            if ref_words:
                aligned_pairs = _align_tokens_dp(words, ref_words)
                if len(aligned_pairs) == len(words):
                    clamped_pairs = []
                    for st, en in aligned_pairs:
                        st = max(0.0, float(st))
                        en = max(st + 0.05, float(en))
                        if en - st > 1.5:
                            en = st + 1.5
                        clamped_pairs.append((round(st, 2), round(en, 2)))

                    durs = [p[1] - p[0] for p in clamped_pairs]
                    avg_dur = sum(durs) / max(1, len(durs))
                    if avg_dur > 1.2 or avg_dur < 0.05:
                        logger.warning(
                            f"Acoustic alignment implausible (avg_dur={avg_dur:.2f}s); "
                            "using relative fallback to preserve speaker boundaries."
                        )
                    else:
                        logger.info(f"Acoustic forced alignment successful: aligned {len(words)} words (avg_dur={avg_dur:.2f}s).")
                        return [
                            {
                                "word": words[idx],
                                "start": clamped_pairs[idx][0],
                                "end": clamped_pairs[idx][1],
                            }
                            for idx in range(len(words))
                        ]
        except Exception as e:
            logger.warning(f"Acoustic forced alignment encountered an error ({e}); engaging relative fallback.")

    logger.info("Using relative time-proportional distribution across active speech intervals.")
    return fallback_relative_alignment(words, total_dur, active_intervals=active_intervals)
