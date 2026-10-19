"""End-to-end gate for the hole-recovery candidate.

`recovery.py` established that word-stream holes contain recoverable speech and
that recovery is robust to the window constant (13/13 and 3/3 holes recovered at
every length swept). This module asks the only question that matters for shipping:
**does inserting the recovered words improve DER and agreement on both files?**

The mechanism is deliberately simple and content-agnostic:

1. Decode normally (the shipped path) -> words with timings.
2. Find holes: gaps in the word stream that still contain audio.
3. Re-decode each hole with a short window centred on it.
4. Place the recovered words inside the hole, spaced across it, and rebuild the
   turn segmentation.

No word lists, no per-file constants. The window length is a single knob, and the
sweep above says the result is insensitive to it, so it is held at the value that
matched the reference best (16 s) and the others are reported for transparency.
"""
from __future__ import annotations

import os
import tempfile
import wave
from typing import Any, Dict, List, Optional

from . import config as C


def _score(spec: C.FileSpec, segs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from . import scoring
    from . import scoring_spk as SS

    ref = spec.ref()
    dur = float(C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
                .get("audio_duration") or spec.duration())
    st = SS.score_diarization(ref, segs, dur)["stages"]["ref_tiles"]
    ag = SS.segment_agreement_weighted(ref, segs)
    txt = scoring.score_text(ref, segs)
    return {
        "n_segments": len(segs), "der": st["der"], "conf_s": st["conf_s"],
        "miss_s": st["miss_s"], "agree": ag["pct_ok"],
        "agree_word": ag["agree_word"], "agree_duration": ag["agree_duration"],
        "wer_norm": txt["wer_norm"], "wer_dialect_norm": txt["wer_dialect_norm"],
        "hyp_words": txt["hyp_words"],
    }


def _decode_words(path: str) -> List[Dict[str, Any]]:
    from . import pipeline_runner as PR

    r = PR.run_captured_asr(path, timing="uniform")
    return r.get("words") or []


def _norm(w: str) -> str:
    import re

    return re.sub(r"[^\w\u0600-\u06FF]", "", w or "")


def _trim_overlap(decoded: List[Dict[str, Any]],
                  before: List[str], after: List[str],
                  max_trim: int = 8) -> List[Dict[str, Any]]:
    """Drop decoded words that merely repeat what is already on either side.

    A short hole forces a wider decode window for context, and that window can
    extend past the hole into speech already transcribed. Left in place those
    repeats become insertion errors and inflate WER while DER barely notices,
    which is exactly the failure the first version of this gate showed. Trimming
    against the neighbouring words is cheap and needs no reference text.
    """
    toks = [_norm(w.get("word", "")) for w in decoded]
    pre = [_norm(x) for x in before]
    post = [_norm(x) for x in after]

    lo = 0
    for k in range(min(max_trim, len(toks)), 0, -1):
        if toks[:k] and pre[-k:] == toks[:k]:
            lo = k
            break

    hi = len(toks)
    for k in range(min(max_trim, len(toks) - lo), 0, -1):
        if toks[hi - k:] and post[:k] == toks[hi - k:]:
            hi -= k
            break

    return decoded[lo:hi]


def build_recovered_words(key: str, window: float = 16.0,
                          min_hole: float = 2.0) -> Dict[str, Any]:
    """Return the shipped word stream with recovered hole words spliced in."""
    from . import recovery as R

    spec = C.SUITE[key]
    words = C.read_json(spec.word_dir / "C_words.json")
    base = [dict(w) for w in words]

    import numpy as np
    a, sr = R._read_wav(str(spec.artifact_dir / "asr.16k.wav"))
    holes = [h for h in R.find_holes(key, min_hole) if h["has_audio"]]

    inserted: List[Dict[str, Any]] = []
    for h in holes:
        mid = (h["start"] + h["end"]) / 2.0
        lo = max(0.0, mid - window / 2.0)
        hi = min(len(a) / sr, mid + window / 2.0)
        path = R._write_slice(a, sr, lo, hi)
        try:
            rw = _decode_words(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if not rw:
            continue

        # Words already transcribed on either side of the hole. A wide context
        # window can re-decode them, so trim the repeats before inserting.
        before = [w.get("word", "") for w in base
                  if float(w.get("end", 0)) <= h["start"] + 0.01][-10:]
        after = [w.get("word", "") for w in base
                 if float(w.get("start", 0)) >= h["end"] - 0.01][:10]
        rw = _trim_overlap(rw, before, after)
        if not rw:
            continue

        # space the recovered words across the hole itself, not the wider window,
        # so they land where the speech actually is
        n = len(rw)
        span = max(0.2, h["dur"])
        step = span / n
        for i, w in enumerate(rw):
            inserted.append({
                "word": w.get("word", ""),
                "start": round(h["start"] + i * step, 2),
                "end": round(h["start"] + (i + 1) * step, 2),
            })

    merged = sorted(base + inserted, key=lambda w: (w["start"], w["end"]))
    return {"words": merged, "n_base": len(base), "n_inserted": len(inserted),
            "holes": holes, "window": window}


def run_gate(key: str, window: float = 16.0, force: bool = False) -> Dict[str, Any]:
    """Compare shipped segmentation against hole-recovered segmentation."""
    from app.services.alignment import align_timestamps

    spec = C.SUITE[key]
    out_path = spec.out_dir / "recovery_gate.json"
    if out_path.exists() and not force:
        cached = C.read_json(out_path)
        if cached.get("window") == window:
            return cached

    turns = C.read_json(spec.artifact_dir / "turns_auto.json")

    base_words = C.read_json(spec.word_dir / "C_words.json")
    baseline = _score(spec, align_timestamps(base_words, turns))

    rec = build_recovered_words(key, window=window)
    recovered = _score(spec, align_timestamps(rec["words"], turns))

    out = {
        "key": key, "window": window,
        "baseline": baseline, "recovered": recovered,
        "n_base_words": rec["n_base"], "n_inserted_words": rec["n_inserted"],
        "n_holes": len(rec["holes"]),
        "hole_seconds": round(sum(h["dur"] for h in rec["holes"]), 1),
        "d_der": round(recovered["der"] - baseline["der"], 4),
        "d_agree_word": round(recovered["agree_word"] - baseline["agree_word"], 2),
        "d_miss_s": round(recovered["miss_s"] - baseline["miss_s"], 2),
        "d_conf_s": round(recovered["conf_s"] - baseline["conf_s"], 2),
    }
    C.write_json(out_path, out)
    return out
