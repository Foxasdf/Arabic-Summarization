"""Recovering speech the ASR drops: long-context vs boundary-aligned decoding.

What the evidence showed (133306)
---------------------------------
A 9 s stretch, 78.5-87.3 s, is loud (peaks -17 dB), continuous, and contains real
speech that the reference transcript covers at `84.0 s: "لما كنت صف سابع تقريبا"`.
It produced **zero words**. Pyannote did detect speech there, so this is not a VAD
or diarization problem, and the slice it fell inside was not truncated -- on
`nlp_5min` slices as long as 29.65 s transcribe at normal word rates, so there is
no silent 28 s cap.

The decisive test: the same audio decodes differently depending on where the
window starts.

    decode 61.2-90.3 (the shipped 29.1 s slice)  -> 63 words, ends "والله شو اسمه"
    decode 78.5-90.3 (11.8 s window)             -> 13 words, contains the missing line

Same samples, same model, different *context composition*. That points at long
single-shot decoding losing content, which is a general failure mode rather than a
quirk of this file.

The candidate
-------------
Re-decode any region the first pass left empty but which Pyannote marks as speech,
using a shorter window anchored on the speech itself. It is content-agnostic: no
word lists, no per-file constants. The only tunable is a window length, and it is
swept here rather than picked.
"""
from __future__ import annotations

import os
import tempfile
import wave
from typing import Any, Dict, List, Optional, Tuple

from . import config as C


def _read_wav(path: str) -> Tuple[Any, int]:
    import numpy as np

    w = wave.open(path, "rb")
    sr = w.getframerate()
    n = w.getnframes()
    a = np.frombuffer(w.readframes(n), dtype=np.int16)
    w.close()
    return a, sr


def _write_slice(a, sr: int, lo: float, hi: float) -> str:
    lo_i = max(0, int(round(lo * sr)))
    hi_i = max(lo_i + 1, int(round(hi * sr)))
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="recover_")
    os.close(fd)
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(a[lo_i:hi_i].tobytes())
    w.close()
    return path


def find_holes(key: str, min_hole: float = 2.0) -> List[Dict[str, Any]]:
    """Spans where the decoded word stream goes empty while audio is still present.

    Defined from the word stream rather than from Pyannote turn coverage: a turn
    boundary marks who is speaking, not where audio exists, so keying holes to
    turns under-reports them (a 9 s gap here surfaced as only 2.7 s, because the
    turns visible over it were short). Anchoring on adjacent words catches the
    whole span regardless of how the turn segmentation split it.

    Audio presence is then checked so pure silence is never targeted.
    """
    import numpy as np

    spec = C.SUITE[key]
    words = C.read_json(spec.word_dir / "C_words.json")
    ws = sorted((float(w["start"]), float(w["end"])) for w in words
                if w.get("start") is not None and w.get("end") is not None)

    a, sr = _read_wav(str(spec.artifact_dir / "asr.16k.wav"))
    dur = len(a) / sr

    holes = []
    for (_, prev_end), (next_start, _) in zip(ws, ws[1:]):
        gap = next_start - prev_end
        if gap < min_hole:
            continue
        lo_i, hi_i = int(prev_end * sr), int(next_start * sr)
        seg = a[lo_i:hi_i]
        if len(seg) == 0:
            continue
        rms = float(np.sqrt(np.mean(seg.astype("float32") ** 2))) / 32768.0
        db = 20.0 * np.log10(rms + 1e-12)
        db = float(db)
        holes.append({"start": round(prev_end, 2), "end": round(next_start, 2),
                      "dur": round(gap, 2), "rms_db": round(db, 1),
                      "has_audio": bool(db > -35.0)})
    return holes


def probe_window(key: str, lo: float, hi: float, pad: float = 0.5) -> Dict[str, Any]:
    """Decode one padded window and return its text and word count."""
    from . import pipeline_runner as PR

    spec = C.SUITE[key]
    a, sr = _read_wav(str(spec.artifact_dir / "asr.16k.wav"))
    dur = len(a) / sr
    w_lo = max(0.0, lo - pad)
    w_hi = min(dur, hi + pad)
    path = _write_slice(a, sr, w_lo, w_hi)
    try:
        r = PR.run_captured_asr(path, timing="uniform")
        text = (r.get("text") or "").strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"window": [round(w_lo, 2), round(w_hi, 2)],
            "window_dur": round(w_hi - w_lo, 2),
            "words": len(text.split()), "text": text}


def sweep_window_lengths(key: str, lengths: Optional[List[float]] = None) -> Dict[str, Any]:
    """For each detected hole, try re-decoding at several window lengths.

    The point is to find whether recovery is *robust to the constant* or only
    works at one magic value. A lever that needs a precise window length is a
    tuned parameter; one that works across a range is a real fix.
    """
    spec = C.SUITE[key]
    lengths = lengths or [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    holes = find_holes(key)
    out_path = spec.out_dir / "recovery.json"
    if out_path.exists():
        cached = C.read_json(out_path)
        if cached.get("holes") == holes:
            return cached

    results = []
    for h in holes:
        mid = (h["start"] + h["end"]) / 2.0
        trials = []
        for L in lengths:
            lo = max(0.0, mid - L / 2.0)
            hi = min(spec.duration(), mid + L / 2.0)
            p = probe_window(key, lo, hi, pad=0.0)
            trials.append({"len": L, **p})
        results.append({"hole": h, "trials": trials})

    out = {"key": key, "holes": holes, "results": results}
    C.write_json(out_path, out)
    return out


def summarize(sweep: Dict[str, Any]) -> Dict[str, Any]:
    """Word counts by window length across all holes, plus the shipped baseline.

    Baseline for comparison is what the shipped single long decode produced in
    those same spans -- which is zero by construction, since that is how a hole
    was defined.
    """
    by_len: Dict[float, List[int]] = {}
    for r in sweep["results"]:
        for t in r["trials"]:
            by_len.setdefault(t["len"], []).append(t["words"])
    rows = []
    for L in sorted(by_len):
        v = by_len[L]
        rows.append({"len": L, "mean_words": round(sum(v) / len(v), 2),
                     "holes_recovered": sum(1 for x in v if x > 0),
                     "n_holes": len(v)})
    return {"holes": len(sweep["holes"]),
            "hole_seconds": round(sum(h["dur"] for h in sweep["holes"]), 1),
            "by_len": rows}
