"""Hole recovery with acoustic word timing.

Rounds 4-5 established the blocker precisely: re-decoded hole text was being
placed with *synthetic* timestamps, spread across the hole. Inserted words then got
attributed to whichever speaker the interpolation happened to put them next to,
which is what the WER regression was actually measuring. The 8.8 s hole on 133306
shows the mechanism — its speech starts at 84.0 s but uniform spacing put the words
at 78.5 s, five seconds early.

Arm C already supplies acoustic word timing for the main transcript. This module
asks whether the same mechanism rescues hole recovery: align the decoded hole text
to the hole's audio so the words know *when* they were spoken, then let the existing
speaker-assignment stage place them.

`align_words_to_audio` exposes `active_intervals`, which is the natural handle for
the pause problem: restricting alignment to the voiced portion of the hole keeps
words off the silence and, crucially, off the laughter that confused the
energy-anchor attempt.
"""
from __future__ import annotations

import os
import tempfile
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


def _voiced_intervals(a, sr: int, lo: float, hi: float,
                      rel_db: float = -18.0, min_gap: float = 0.12) -> List[Dict[str, float]]:
    """Voiced sub-spans inside [lo, hi], in absolute audio time.

    Frames are 50 ms and the threshold is relative to the hole's own peak, so it
    adapts per recording rather than assuming a level. Gaps shorter than
    ``min_gap`` are bridged to avoid dropping word onsets.
    """
    import numpy as np

    lo_i = max(0, int(lo * sr))
    hi_i = min(len(a), int(hi * sr))
    if hi_i <= lo_i:
        return [{"start": lo, "end": hi}]

    seg = a[lo_i:hi_i].astype("float32") / 32768.0
    win = max(1, int(0.05 * sr))
    n = len(seg) // win
    if n < 1:
        return [{"start": lo, "end": hi}]

    frames = seg[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20.0 * np.log10(rms)
    voiced = db >= (float(db.max()) + rel_db)

    spans: List[List[float]] = []
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j + 1 < n and voiced[j + 1]:
                j += 1
            spans.append([lo + i * win / sr, lo + (j + 1) * win / sr])
            i = j + 1
        else:
            i += 1

    if not spans:
        return []

    merged: List[List[float]] = [spans[0]]
    for s in spans[1:]:
        if s[0] - merged[-1][1] <= min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [{"start": round(s, 3), "end": round(e, 3)} for s, e in merged]


def build_timed(key: str, window: float = 16.0, min_hole: float = 2.0,
                gate_rule: Optional[str] = None,
                use_intervals: bool = True) -> Dict[str, Any]:
    """Recover hole words with acoustic timings instead of synthetic ones."""
    from . import recovery as R
    from . import recovery_gate as RG
    from . import discriminate as D
    from app.services.forced_alignment import align_words_to_audio

    spec = C.SUITE[key]
    base = [dict(w) for w in C.read_json(spec.word_dir / "C_words.json")]
    a, sr = R._read_wav(str(spec.artifact_dir / "asr.16k.wav"))
    audio_dur = len(a) / sr
    holes = [h for h in R.find_holes(key, min_hole) if h["has_audio"]]
    rule = RG._RULE_FNS.get(gate_rule) if gate_rule else None

    inserted: List[Dict[str, Any]] = []
    filled: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for h in holes:
        mid = (h["start"] + h["end"]) / 2.0
        lo = max(0.0, mid - window / 2.0)
        hi = min(audio_dur, mid + window / 2.0)
        path = R._write_slice(a, sr, lo, hi)
        try:
            rw = RG._decode_words(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if not rw:
            continue

        before = [w.get("word", "") for w in base
                  if float(w.get("end", 0)) <= h["start"] + 0.01][-10:]
        after = [w.get("word", "") for w in base
                 if float(w.get("start", 0)) >= h["end"] - 0.01][:10]
        rw = RG._trim_overlap(rw, before, after)
        if not rw:
            continue

        if rule is not None:
            decoded = [D._norm(w.get("word", "")) for w in rw if D._norm(w.get("word", ""))]
            feats = D.features_for_hole(key, h, decoded, base)
            if not rule(feats):
                continue

        text = " ".join(w.get("word", "") for w in rw)
        # Align the recovered text to the *hole's* audio, not the wide context
        # window: the window exists only to give the decoder context.
        hole_path = R._write_slice(a, sr, h["start"], h["end"])
        try:
            intervals = _voiced_intervals(a, sr, h["start"], h["end"]) if use_intervals else None
            aligned = align_words_to_audio(hole_path, text, active_intervals=intervals)
        except Exception as e:  # noqa: BLE001 - fall back rather than lose the hole
            aligned = []
            diagnostics.append({"hole": h, "error": repr(e)})
        finally:
            try:
                os.unlink(hole_path)
            except OSError:
                pass

        if not aligned:
            continue

        filled.append(h)
        # `aligned` stamps are relative to the hole slice, so they must be shifted
        # by the hole start. They must NOT then be clamped to [start, end]: the
        # aligner spreads a sentence across the voiced part of the hole, and some
        # words legitimately land just outside it. Clamping silently collapsed
        # several words onto the same boundary instant, which the downstream
        # utterance builder then dropped -- leaving one- and two-word fragments
        # like "يسبب" where a full clause had been decoded. Shift only, and let
        # the surrounding words order the sequence.
        for w in aligned:
            st = float(w.get("start", 0.0)) + h["start"]
            en = float(w.get("end", st - h["start"])) + h["start"]
            inserted.append({
                "word": w.get("word", ""),
                "start": round(st, 2),
                "end": round(max(en, st + 0.05), 2),
            })

    merged = sorted(base + inserted, key=lambda w: (w["start"], w["end"]))
    return {"words": merged, "n_base": len(base), "n_inserted": len(inserted),
            "holes": filled, "n_holes_total": len(holes), "window": window,
            "gate_rule": gate_rule, "use_intervals": use_intervals,
            "diagnostics": diagnostics}


def run_gate(key: str, window: float = 16.0, gate_rule: Optional[str] = None,
             use_intervals: bool = True, force: bool = False) -> Dict[str, Any]:
    """Score timed hole recovery against the shipped segmentation."""
    from app.services.alignment import align_timestamps

    spec = C.SUITE[key]
    tag = f"timed_{gate_rule or 'all'}_int{int(use_intervals)}".replace("<", "lt").replace(">", "gt")
    out_path = spec.out_dir / f"recovery_gate_{tag}.json"
    if out_path.exists() and not force:
        cached = C.read_json(out_path)
        if cached.get("window") == window:
            return cached

    turns = C.read_json(spec.artifact_dir / "turns_auto.json")
    base_words = C.read_json(spec.word_dir / "C_words.json")
    baseline = _score(spec, align_timestamps(base_words, turns))

    rec = build_timed(key, window=window, gate_rule=gate_rule, use_intervals=use_intervals)
    recovered = _score(spec, align_timestamps(rec["words"], turns))

    out = {
        "key": key, "window": window, "gate_rule": gate_rule,
        "use_intervals": use_intervals,
        "baseline": baseline, "recovered": recovered,
        "n_base_words": rec["n_base"], "n_inserted_words": rec["n_inserted"],
        "n_holes": rec["n_holes_total"], "n_holes_filled": len(rec["holes"]),
        "d_der": round(recovered["der"] - baseline["der"], 4),
        "d_wer": round(recovered["wer_norm"] - baseline["wer_norm"], 4),
        "d_agree_word": round(recovered["agree_word"] - baseline["agree_word"], 2),
    }
    C.write_json(out_path, out)
    return out
