"""Instrumented, behaviour-identical copies of the ASR pipeline stages.

Adjustment #2 (exact guardrail diffs): instead of re-running the guardrail on
already-stripped text, we re-implement ``BaseASREngine.transcribe`` and
``transcribe_slice`` with *capture points* inserted around each stage. The
slicing, chunk planning, VAD gate, inference, guardrails, seam dedup and
uniform word timing are the SAME production code paths -- the only difference
is that intermediate strings are recorded.

Behaviour parity is asserted by :func:`verify_introspection` (production
function source still matches the instrumentation anchors) and by
:func:`diff_against_production` (same text for the same audio) before the
numbers are trusted.
"""
from __future__ import annotations

import gc
import inspect
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as C


# ---------------------------------------------------------------------------
# Introspection guard: fail loudly if production code drifts from our copy
# ---------------------------------------------------------------------------
ANCHORS = {
    "uniform_word_timing": "word_dur = dur_sec / max(1, len(words))",
    "seam_dedup": "overlap_k = self._find_word_overlap(tail_words, head_words)",
    "vad_gate": "raw_text = self._transcribe_slice_internal(slice_wav_path, dur_sec=dur_sec)",
    "guardrail": "clean_and_guard_asr_text(raw_text, dur_sec=dur_sec)",
}


def verify_introspection() -> Dict[str, bool]:
    """Confirm the production code still contains every anchor we replicate."""
    from app.models.asr.base import BaseASREngine

    src = {
        "transcribe": inspect.getsource(BaseASREngine.transcribe),
        "transcribe_slice": inspect.getsource(BaseASREngine.transcribe_slice),
    }
    joined = "\n".join(src.values())
    return {name: (anchor in joined) for name, anchor in ANCHORS.items()}


# ---------------------------------------------------------------------------
# Harness-time VAD probing
# ---------------------------------------------------------------------------
def rms_db_of(path: Path) -> float:
    """RMS dBFS of a PCM slice, using production's own normalisation rules."""
    from app.services.vad import compute_audio_rms_db
    from scipy.io import wavfile

    _, data = wavfile.read(str(path))
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return round(compute_audio_rms_db(data), 2)


def transcribe_slice_captured(engine, slice_wav_path: Path, dur_sec: float) -> Dict[str, Any]:
    """Production ``transcribe_slice`` with every intermediate string captured."""
    from app.config import settings
    from app.services.vad import is_speech_present

    rec: Dict[str, Any] = {
        "vad_passed": None,
        "rms_db": None,
        "raw": "",
        "guarded": "",
        "guardrail_changed": False,
    }

    has_speech, rms_db = is_speech_present(slice_wav_path)
    rec["vad_passed"] = bool(has_speech)
    rec["rms_db"] = round(float(rms_db), 2)
    if not has_speech:
        rec["raw"] = ""
        rec["guarded"] = ""
        return rec

    raw_text = engine._transcribe_slice_internal(slice_wav_path, dur_sec=dur_sec)
    rec["raw"] = raw_text or ""

    if getattr(settings, "ASR_GUARDRAILS_ENABLED", True):
        from app.models.asr.guardrails import clean_and_guard_asr_text

        guarded = clean_and_guard_asr_text(raw_text, dur_sec=dur_sec)
    else:  # pragma: no cover - config default is True
        guarded = (raw_text or "").strip()

    rec["guarded"] = guarded or ""
    rec["guardrail_changed"] = rec["guarded"].strip() != rec["raw"].strip()
    return rec


# ---------------------------------------------------------------------------
# Timeline helpers (module-level so they are picklable and swappable)
# ---------------------------------------------------------------------------
def uniform_timing(start_sec: float, dur_sec: float, kept_words: List[str], offset: int, total_words: int) -> List[Dict[str, Any]]:
    """Exact copy of production uniform word timing (base.py transcribe).

    ``offset``/``total_words`` are unique to this slice (no dedup happened), so
    timing is relative to the slice itself, exactly like production.
    """
    word_dur = dur_sec / max(1, total_words)
    out = []
    for j, word in enumerate(kept_words):
        k = offset + j
        out.append(
            {
                "word": word,
                "start": round(start_sec + k * word_dur, 2),
                "end": round(start_sec + (k + 1) * word_dur, 2),
            }
        )
    return out


def bisection_timing(start_sec: float, dur_sec: float, kept_words: List[str], offset: int, total_words: int,
                     gap: float = 0.06, min_word: float = 0.12, min_gap: float = 0.10) -> List[Dict[str, Any]]:
    """Arm B: bisect the slice on the largest silence, recurse, then split text.

    Pure numeric substitution for word timestamps: no extra audio processing,
    no introspection, no model, no new downloads. If the timing headroom is real
    it must show up here; if not, only an acoustic aligner can help.
    """
    spans = _bisect_spans(start_sec, dur_sec, max(0, total_words - offset), gap, min_word, min_gap)
    out = []
    for j, word in enumerate(kept_words):
        st, en = spans[j] if j < len(spans) else (start_sec, start_sec + dur_sec)
        out.append({"word": word, "start": round(st, 2), "end": round(en, 2)})
    return out


def _bisect_spans(start: float, end: float, n: int, gap: float, min_word: float, min_gap: float) -> List[tuple]:
    dur = end - start
    if n <= 1:
        return [(start, start + dur)]
    est = n * (min_word + gap / 2.0)
    if dur >= 2 * est and dur >= 2 * (min_word + min_gap):
        q = _largest_silence_midpoint(start, end)
        if not (start + 0.05 < q < end - 0.05):
            q = start + dur / 2.0
        n_left = max(1, min(n - 1, n // 2))
        return _bisect_spans(start, q, n_left, gap, min_word, min_gap) + \
            _bisect_spans(q, end, n - n_left, gap, min_word, min_gap)
    # Text-only equal split inside this span (production's behaviour, localised)
    lo = start + min_gap / 3.0
    span = max(n * 0.05, dur - 2 * (min_gap / 3.0))
    step = span / n
    return [
        (lo + i * step, max(lo + i * step + min_word, lo + (i + 1) * step))
        for i in range(n)
    ]


def _largest_silence_midpoint(start: float, end: float, min_sep_s: float = 0.25, frame_s: float = 0.05) -> float:
    """Locate the centroid of the lowest-energy frame strictly inside ``(start, end)``.

    Uses a 50 ms RMS frame track over the in-memory speaker WAV, which is the
    same signal (16 kHz mono PCM) production slices from, so a split point here
    is a genuine pause in this recording rather than a text-length artifact.
    """
    if _AUDIO is None:
        return (start + end) / 2.0
    sr = 16000
    n = len(_AUDIO)
    lo_i = int(max(0.0, start + min_sep_s) * sr)
    hi_i = int(min(n, (end - min_sep_s) * sr))
    if hi_i - lo_i < sr // 20:
        return (start + end) / 2.0
    data = _AUDIO[lo_i:hi_i]
    if data.dtype.kind in "iu":
        data = data.astype("float32") / float(32768.0 if data.dtype.itemsize == 2 else 2147483648.0)
    f_len = int(frame_s * sr)
    n_frames = len(data) // f_len
    if n_frames < 3:
        return (start + end) / 2.0
    frames = data[: n_frames * f_len].reshape(n_frames, f_len).astype("float64")
    # Smooth with a 3-frame box so a single click cannot win
    energy = (frames ** 2).mean(axis=1)
    if len(energy) >= 3:
        energy = (energy[:-2] + energy[1:-1] + energy[2:]) / 3.0
    k = int(energy.argmin())
    return lo_i / sr + (k + 1) * frame_s


_AUDIO = None  # set by run_transcription() for bisection_timing


# ---------------------------------------------------------------------------
# Instrumented continuous transcription (mirrors BaseASREngine.transcribe)
# ---------------------------------------------------------------------------
async def run_transcription(
    engine,
    audio_path: Path,
    timing: str = "uniform",
    on_progress=None,
) -> Dict[str, Any]:
    """Run production-equivalent continuous transcription, capturing everything.

    Returns ``{"text", "words", "slices", "words_pre_guardrail", "total_wall_s"}``.
    """
    global _AUDIO
    import numpy as np
    from scipy.io import wavfile

    from app.config import settings
    from app.models.asr.base import BaseASREngine
    from app.services.audio_processor import get_audio_duration, get_ffmpeg_binary

    t0 = time.perf_counter()
    if not engine.is_loaded():
        engine.load()

    total_duration = get_audio_duration(audio_path)
    audio_data = None
    sr = 16000
    try:
        sr_read, raw_data = wavfile.read(str(audio_path))
        if sr_read == 16000:
            audio_data = raw_data
            sr = sr_read
    except Exception:
        audio_data = None

    _AUDIO = audio_data

    chunk_len = float(getattr(engine, "chunk_len", getattr(settings, "ASR_CHUNK_SECONDS", 25.0)))
    overlap = float(getattr(engine, "overlap", getattr(settings, "ASR_OVERLAP_SECONDS", 2.5)))
    min_silence_ms = float(getattr(settings, "ASR_MIN_SILENCE_MS", 300))
    ffmpeg_bin = get_ffmpeg_binary()

    slice_info = BaseASREngine.plan_vad_chunks(
        audio_data=audio_data,
        sr=sr,
        total_duration=total_duration,
        target_chunk_len=chunk_len,
        min_chunk_len=max(8.0, round(chunk_len * 0.72, 1)),
        max_chunk_len=min(30.0, round(chunk_len + 5.0, 1)),
        overlap=overlap,
        min_silence_sec=min_silence_ms / 1000.0,
    )

    results: List[Dict[str, Any]] = []
    slice_paths: List[Optional[Path]] = []

    tmp_dir = tempfile.mkdtemp(prefix="diag133306_")
    tmp_path = Path(tmp_dir)
    try:
        for item in slice_info:
            i, start, dur = item["idx"], item["start"], item["dur"]
            slice_wav = tmp_path / f"slice_{i:04d}.wav"
            created = False
            if audio_data is not None:
                try:
                    s_idx = int(start * sr)
                    e_idx = min(len(audio_data), int((start + dur) * sr))
                    if e_idx > s_idx:
                        wavfile.write(str(slice_wav), sr, audio_data[s_idx:e_idx])
                        created = slice_wav.exists()
                except Exception:
                    created = False
            if not created:
                cmd = [
                    ffmpeg_bin, "-y", "-loglevel", "error",
                    "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(slice_wav),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                created = proc.returncode == 0 and slice_wav.exists()
            slice_paths.append(slice_wav if created else None)

        for idx_k, item in enumerate(slice_info):
            i, start, dur = item["idx"], item["start"], item["dur"]
            p = slice_paths[idx_k]
            rec = {"idx": i, "start": start, "dur": dur, "path": str(p) if p else None}
            if p is not None and p.exists():
                # VAD probe first (same order of evaluation as production)
                rec["rms_db"] = rms_db_of(p)
                cap = transcribe_slice_captured(engine, p, dur)
                rec.update(cap)
                try:
                    p.unlink()
                except OSError:
                    pass
            results.append(rec)
            if on_progress and (idx_k % 3 == 0 or idx_k == len(slice_info) - 1):
                on_progress(idx_k + 1, len(slice_info))

        del audio_data
        gc.collect()
    finally:
        for p in slice_paths:
            if p and p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    # --- Production sequential overlap dedup + word timing -----------------
    all_words: List[Dict[str, Any]] = []
    all_text_parts: List[str] = []
    pre_guardrail_text: List[str] = []
    seam_events: List[Dict[str, Any]] = []

    for rec in results:
        slice_text = rec.get("guarded") or ""
        if rec.get("raw"):
            pre_guardrail_text.append(rec["raw"])
        if not slice_text:
            continue
        words = slice_text.split()
        if not words:
            continue

        words_to_add = words
        overlap_k = 0
        if rec["idx"] > 0 and all_words:
            tail_words = [w["word"] for w in all_words[-15:]]
            head_words = words[:15]
            overlap_k = engine._find_word_overlap(tail_words, head_words)
            if overlap_k > 0:
                words_to_add = words[overlap_k:]
        seam_events.append(
            {
                "idx": rec["idx"],
                "start": rec["start"],
                "overlap_k": overlap_k,
                "dropped": " ".join(words[:overlap_k]),
            }
        )
        if not words_to_add:
            continue

        offset_idx = len(words) - len(words_to_add)
        if timing == "uniform":
            chunk_words = uniform_timing(rec["start"], rec["dur"], words_to_add, offset_idx, len(words))
        elif timing == "bisection":
            chunk_words = bisection_timing(rec["start"], rec["dur"], words_to_add, offset_idx, len(words))
        else:  # pragma: no cover
            raise ValueError(timing)

        if chunk_words:
            all_text_parts.append(" ".join(cw["word"] for cw in chunk_words))
            all_words.extend(chunk_words)

    return {
        "text": " ".join(all_text_parts).strip(),
        "pre_guardrail_text": " ".join(pre_guardrail_text).strip(),
        "words": all_words,
        "slices": results,
        "seam_events": seam_events,
        "total_wall_s": round(time.perf_counter() - t0, 2),
        "timing": timing,
        "slice_plan": slice_info,
        "audio_duration": total_duration,
    }


def word_slice_index(words: List[Dict[str, Any]], slice_plan: List[Dict[str, Any]]) -> List[int]:
    """Map each word to the slice whose start/end window contains its midpoint."""
    out = []
    for w in words:
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        hit = -1
        for sp in slice_plan:
            if sp["start"] <= mid < sp["start"] + sp["dur"]:
                hit = sp["idx"]
                break
        out.append(hit)
    return out
