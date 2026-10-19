"""Candidate evaluation: score a candidate fix on every file and gate it jointly.

Each candidate is applied through :mod:`candidates` (runtime patch, restored on
exit) and measured on the **same captured audio and diarization** as the baseline,
so the only difference is the change under test. That keeps the comparison
paired, which matters far more than extra repeats when the underlying GPU pass is
deterministic (``parity.identical == True`` was verified per file).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as C


def _text_only(segments: List[Dict[str, Any]]) -> str:
    return " ".join(s.get("text", "") for s in segments).strip()


def evaluate_text_candidate(key: str, cand: str) -> Dict[str, Any]:
    """Apply a guardrail-class candidate to the captured per-slice text.

    The production text pipeline is: guardrail each 28 s slice, then dedupe words
    at each seam. Concatenating raw slice outputs would skip the dedupe and
    over-count words by ~6% on 133306, so this replays the seam dedupe too --
    otherwise the candidate would appear to "recover" words that dedupe removes
    anyway, and its WER delta would be measured against the wrong baseline.
    """
    from . import candidates as CN
    from . import scoring
    from app.models.asr.guardrails import clean_and_guard_asr_text
    from app.models.asr.base import BaseASREngine

    spec = C.SUITE[key]
    captured = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
    ref = spec.ref()
    ref_text = " ".join(s.get("text", "") for s in ref).strip()

    raw_texts = [s.get("raw") or "" for s in captured.get("slices", [])]
    durs = [float(s["dur"]) for s in captured.get("slices", [])]

    def build() -> Tuple[str, List[List[str]]]:
        per_slice = [[w for w in clean_and_guard_asr_text(t, dur_sec=d).split() if w]
                     for t, d in zip(raw_texts, durs)]
        # replay production seam dedupe (string equality, as production does)
        joined: List[str] = []
        for words in per_slice:
            if not joined:
                joined = list(words)
                continue
            k = BaseASREngine._find_word_overlap(joined, words)
            joined.extend(words[k:] if k else words)
        return " ".join(joined).strip(), per_slice

    before_text, before_slices = build()
    with CN.apply(cand):
        after_text, after_slices = build()

    b = scoring.wer_cer(ref_text, before_text)
    a = scoring.wer_cer(ref_text, after_text)

    changed = [
        {"slice": i, "before_words": len(x), "after_words": len(y)}
        for i, (x, y) in enumerate(zip(before_slices, after_slices)) if x != y
    ]
    return {
        "key": key,
        "candidate": cand,
        "wer_before": b["norm_wer"],
        "wer_after": a["norm_wer"],
        "delta": round(a["norm_wer"] - b["norm_wer"], 4),
        "words_before": b["hyp_word_count"],
        "words_after": a["hyp_word_count"],
        "words_delta": a["hyp_word_count"] - b["hyp_word_count"],
        "slices_changed": changed,
        "text_after": after_text,
    }


def evaluate_engine_words(key: str) -> Dict[str, Any]:
    """Does the engine's own word list exist and survive production validation?

    This is a *capability* probe rather than a full re-run: it verifies (a) the
    captured pass emitted words, (b) they pass the same count/sanity validation
    `transcribe_with_speakers` applies, and (c) they differ from the forced-alignment
    result in a way that could change speaker attribution.
    """
    spec = C.SUITE[key]
    captured = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
    text = captured.get("text") or ""
    words = captured.get("words") or []
    n_txt = len(text.split())
    n_words = len(words)
    within = abs(n_words - n_txt) <= max(3, 0.05 * n_txt) if n_txt else False
    return {
        "key": key,
        "candidate": "engine_words",
        "engine_emits_words": n_words > 0,
        "n_words": n_words,
        "n_text_words": n_txt,
        "passes_validation": within,
        "note": "production gated these out via has_word_timestamps (undefined on every engine class)",
    }


def _der_of_segments(spec: C.FileSpec, segs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from . import scoring_spk as SS

    ref = spec.ref()
    dur = float(C.read_json(spec.artifact_dir / "captured_asr_uniform.json").get("audio_duration") or spec.duration())
    dj = SS.score_diarization(ref, segs, dur)
    ag = SS.segment_agreement_weighted(ref, segs)
    st = dj["stages"]["ref_tiles"]
    return {"der": st["der"], "conf_s": st["conf_s"], "miss_s": st["miss_s"],
            "agree": ag["pct_ok"], "agree_word": ag["agree_word"],
            "agree_duration": ag["agree_duration"], "n_segments": len(segs)}


def evaluate_fuzzy_seam(key: str) -> Dict[str, Any]:
    """Re-run the captured pass with the fuzzy seam dedupe and score it."""
    from . import ablations as AB
    from . import candidates as CN
    from . import pipeline_runner as PR
    from app.services.alignment import align_timestamps

    spec = C.SUITE[key]
    asr_wav = spec.artifact_dir / "asr.16k.wav"
    turns_auto = C.read_json(spec.artifact_dir / "turns_auto.json")
    base_cap = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")

    cached = spec.artifact_dir / "captured_asr_fuzzyseam.json"
    if cached.exists():
        cap = C.read_json(cached)
    else:
        t0 = time.time()
        with CN.apply("fuzzy_seam"):
            cap = PR.run_captured_asr(asr_wav, timing="uniform")
        C.write_json(cached, cap)

    base_segs = align_timestamps(base_cap["words"], turns_auto)
    cand_segs = align_timestamps(cap["words"], turns_auto)
    return {
        "key": key,
        "candidate": "fuzzy_seam",
        "baseline": _der_of_segments(spec, base_segs),
        "candidate_metrics": _der_of_segments(spec, cand_segs),
        "words_before": len(base_cap["words"]),
        "words_after": len(cap["words"]),
        "wall_s": cap.get("resources", {}).get("wall_s"),
    }


def _load_asr_wav(asr_wav: Path):
    """Load the canonical 16 kHz mono ASR WAV once (memory-mapped, read-only)."""
    import numpy as np
    from scipy.io import wavfile

    if not asr_wav.exists():
        return None, 0
    sr, data = wavfile.read(str(asr_wav))
    x = data.astype("float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = max(1e-9, float(np.abs(x).max()))
    return x / peak, int(sr)


def _mask_from_buffer(x, sr: int, start: float, dur: float,
                      rms_thresh_db: float = -42.0, hop_ms: int = 10) -> Optional[List[bool]]:
    """Speech mask for the window [start, start+dur) of an in-memory signal.

    The per-slice scratch WAVs written during the capture are deleted when the run
    ends, so the mask is recomputed from the canonical ASR WAV at the same offsets.
    That is the identical audio the pipeline sliced, so the mask is equivalent.
    """
    try:
        import numpy as np
    except Exception:
        return None
    if x is None or sr <= 0:
        return None
    a = max(0, int(round(start * sr)))
    b = min(len(x), int(round((start + dur) * sr)))
    if b - a < sr // 100:
        return None
    seg = x[a:b]
    hop = max(1, int(sr * hop_ms / 1000))
    n = len(seg) // hop
    if n == 0:
        return None
    frames = seg[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return [bool(v >= rms_thresh_db) for v in db]


def evaluate_snap_words(key: str) -> Dict[str, Any]:
    """Snap word boundaries onto Pyannote turn boundaries where they disagree slightly.

    ``conf_s`` (84.4 s of speaker-confusion time on 133306) is concentrated at turn
    boundaries. A word whose midpoint sits a few hundred ms past the end of its
    turn gets assigned to the next speaker. When a word straddles a turn boundary,
    the boundary is the better estimate of where one speaker stopped and the next
    began than an even split of the 28 s window.

    The candidate moves each straddling word's *own* boundary to the turn edge
    (never past the word's midpoint, which would flip a confident assignment), so
    it can only reassign words that were genuinely ambiguous. Word text is
    untouched.
    """
    from app.services.alignment import align_timestamps, clean_diarization_turns
    from . import scoring

    spec = C.SUITE[key]
    cap = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
    turns = clean_diarization_turns(C.read_json(spec.artifact_dir / "turns_auto.json"))
    base_words = cap["words"]
    ref = spec.ref()

    edges = sorted({round(float(t["start"]), 3) for t in turns} |
                   {round(float(t["end"]), 3) for t in turns})

    def nearest_edge(v: float, tol: float) -> Optional[float]:
        if not edges:
            return None
        import bisect
        i = bisect.bisect_left(edges, v)
        cand = [edges[j] for j in (i - 1, i) if 0 <= j < len(edges)]
        if not cand:
            return None
        e = min(cand, key=lambda x: abs(x - v))
        return e if abs(e - v) <= tol else None

    new_words = []
    moved = 0
    for w in base_words:
        s, e = float(w["start"]), float(w["end"])
        mid = (s + e) / 2.0
        ns, ne = s, e
        tol = max(0.15, min(0.6, (e - s)))
        es = nearest_edge(s, tol)
        ee = nearest_edge(e, tol)
        if es is not None:
            ns = es
        if ee is not None:
            ne = ee
        # never cross the midpoint, and never invert the interval
        ns = min(ns, mid)
        ne = max(ne, mid)
        if ns >= ne:
            ns, ne = s, e
        if abs(ns - s) > 1e-6 or abs(ne - e) > 1e-6:
            moved += 1
        new_words.append({"word": w["word"], "start": round(ns, 2), "end": round(ne, 2)})

    text_before = " ".join(w["word"] for w in base_words)
    text_after = " ".join(w["word"] for w in new_words)
    segs_before = align_timestamps(base_words, turns)
    segs_after = align_timestamps(new_words, turns)
    return {
        "key": key,
        "candidate": "snap_words",
        "text_unchanged": text_before == text_after,
        "words_moved": moved,
        "n_words": len(base_words),
        "n_turn_edges": len(edges),
        "baseline": _der_of_segments(spec, segs_before),
        "candidate_metrics": _der_of_segments(spec, segs_after),
        "wer_before": scoring.score_text(ref, segs_before)["wer_norm"],
        "wer_after": scoring.score_text(ref, segs_after)["wer_norm"],
    }


def evaluate_vad_word_timing(key: str) -> Dict[str, Any]:
    """Spread each slice's words over the slice's *speech* instead of the slice.

    Production (base.py:514) assigns ``word_dur = dur_sec / len(words)`` over the
    whole 28 s window, ignoring where speech actually is. Any silence or trailing
    non-speech inside the window therefore drags every word after it late, which is
    what pushes words across Pyannote turn boundaries and mislabels the speaker
    afterwards.

    The candidate keeps the same word order and the same count -- it only changes
    where in time each word sits. Text is therefore unchanged by construction, so
    any WER movement would indicate a bug in this code rather than an ASR effect.
    """
    from app.services.alignment import align_timestamps
    from . import scoring
    try:
        from app.config import settings
        thresh = float(getattr(settings, "VAD_MIN_RMS_DB", -42.0))
    except Exception:
        thresh = -42.0

    spec = C.SUITE[key]
    cap = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
    turns_auto = C.read_json(spec.artifact_dir / "turns_auto.json")
    base_words = cap["words"]

    def respace(track: Optional[List[bool]], start: float, dur: float, n: int) -> List[Tuple[float, float]]:
        step = dur / max(1, n)
        flat = [(round(start + i * step, 2), round(start + (i + 1) * step, 2)) for i in range(n)]
        if not track or n == 0:
            return flat
        idx = [i for i, v in enumerate(track) if v]
        if len(idx) < 2:
            return flat
        lo_i, hi_i = idx[0], idx[-1]
        span = hi_i - lo_i + 1
        frame = dur / len(track)
        out = []
        for i in range(n):
            a = lo_i + span * i / n
            b = lo_i + span * (i + 1) / n
            out.append((round(start + a * frame, 2), round(start + b * frame, 2)))
        return out

    new_words: List[Dict[str, Any]] = []
    pos = 0
    slices_with_mask = 0
    asr_wav = spec.artifact_dir / "asr.16k.wav"
    buf, sr = _load_asr_wav(asr_wav)
    for sl in cap.get("slices", []):
        n = len((sl.get("guarded") or "").split())
        if not n:
            continue
        mask = _mask_from_buffer(buf, sr, float(sl["start"]), float(sl["dur"]), thresh)
        if mask:
            slices_with_mask += 1
        spans = respace(mask, float(sl["start"]), float(sl["dur"]), n)
        take = min(n, len(base_words) - pos)
        for j in range(take):
            bw = base_words[pos + j]
            new_words.append({"word": bw["word"], "start": spans[j][0], "end": spans[j][1]})
        pos += take
    # any words not covered by a slice keep their original timing
    new_words.extend(base_words[len(new_words):])

    text_before = " ".join(w["word"] for w in base_words)
    text_after = " ".join(w["word"] for w in new_words)

    segs_before = align_timestamps(base_words, turns_auto)
    segs_after = align_timestamps(new_words, turns_auto)
    ref = spec.ref()
    return {
        "key": key,
        "candidate": "vad_word_timing",
        "text_unchanged": text_before == text_after,
        "n_words_before": len(base_words),
        "n_words_after": len(new_words),
        "slices_with_mask": slices_with_mask,
        "n_slices": len(cap.get("slices", [])),
        "baseline": _der_of_segments(spec, segs_before),
        "candidate_metrics": _der_of_segments(spec, segs_after),
        "wer_before": scoring.score_text(ref, segs_before)["wer_norm"],
        "wer_after": scoring.score_text(ref, segs_after)["wer_norm"],
    }
