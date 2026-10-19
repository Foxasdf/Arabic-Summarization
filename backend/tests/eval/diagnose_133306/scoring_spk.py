"""Speaker scoring: DER/JER with a fixed 0.25 s collar, decomposed into stages.

Three quantities are kept distinct because they answer different questions:

* ``ref_tiles``     - reference segments are back-to-back utterances (not an RTTM);
                      raw Pyannote has small gaps. Calls that score as MISS are
                      partly an artifact of that convention.
* ``speech_compare`` - compare only where the optimiser says a hypothesis frame
                      should be speech (hyp-speech sets ref speech to False).
                      Production-realistic "did we put the right words on the
                      right person" number.
* ``strict``        - pure single-speaker RTTM, with and without collar. The
                      number to quote in papers; keep the artifact comparison honest.

Overlap is resolved to a single label with the same midpoint rule production uses
(``clean_diarization_turns``) so that DER is not inflated by simultaneous labels.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

from . import config as C

FRAME = 0.01


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------
def _frames(turns: List[Dict[str, Any]], total: float) -> List[Optional[str]]:
    n = max(1, int(round(total / FRAME)))
    lab: List[Optional[str]] = [None] * n
    for t in sorted(turns, key=lambda x: float(x["start"])):
        s = max(0, int(round(float(t["start"]) / FRAME)))
        e = min(n, int(round(float(t["end"]) / FRAME)))
        for i in range(s, e):
            lab[i] = t["speaker"]
    return lab


def resolve_midpoints(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve overlapping turns to exactly one label per instant.

    Sweep-line over turn boundaries: at every instant the active turns compete
    and the *shortest* wins, so a long background turn cannot swallow a brief
    interjection. This partitions the timeline cleanly (production's midpoint
    rule expressed as a stream) instead of dropping or truncating segments.
    """
    if not turns:
        return []
    reps = [{"start": round(float(t["start"]), 2), "end": round(float(t["end"]), 2),
             "speaker": str(t["speaker"])} for t in turns]
    reps = [r for r in reps if r["end"] - r["start"] > 0.02]
    if not reps:
        return []
    bounds = sorted({b for r in reps for b in (r["start"], r["end"])})

    out: List[Dict[str, Any]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 0.005:
            continue
        mid = (a + b) / 2.0
        active = [r for r in reps if r["start"] <= mid <= r["end"]]
        if not active:
            continue
        winner = min(active, key=lambda r: (r["end"] - r["start"], r["start"]))
        if out and out[-1]["speaker"] == winner["speaker"] and abs(out[-1]["end"] - a) <= 0.05:
            out[-1]["end"] = b
        else:
            out.append({"speaker": winner["speaker"], "start": a, "end": b})
    return out


def _score_labelling(ref_f: List[Optional[str]], hyp_f: List[Optional[str]], total: float,
                     mode: str, collar: float) -> Dict[str, Any]:
    """Frame-level DER with optimal (greedy-assignment) speaker mapping."""
    n = max(1, int(round(total / FRAME)))
    ref_f = (ref_f + [None] * n)[:n]
    hyp_f = (hyp_f + [None] * n)[:n]

    # collar mask from reference label boundaries
    excl = [False] * n
    if collar > 0:
        prev = None
        bounds = []
        for i, v in enumerate(ref_f):
            if v != prev:
                bounds.append(i)
                prev = v
        half = int(round(collar / FRAME))
        for b in bounds:
            for i in range(max(0, b - half), min(n, b + half)):
                excl[i] = True

    if mode == "speech_compare":
        hyp_speech = [v is not None for v in hyp_f]
        ref_f = [r if hyp_speech[i] else None for i, r in enumerate(ref_f)]

    ref_speech = [v is not None for v in ref_f]
    hyp_speech = [v is not None for v in hyp_f]
    # count baseline: reference speech frames (post-rewrite for speech_compare)
    total_ref = sum(1 for i, r in enumerate(ref_speech) if r and not excl[i])
    miss = sum(1 for i in range(n) if ref_speech[i] and not hyp_speech[i] and not excl[i])
    fa = sum(1 for i in range(n) if hyp_speech[i] and not ref_speech[i] and not excl[i])

    # confusion: frames where both have a label but they disagree under best mapping
    pairs: Dict[Tuple[str, str], int] = {}
    for i in range(n):
        if excl[i] or not ref_speech[i] or not hyp_speech[i]:
            continue
        pairs[(ref_f[i], hyp_f[i])] = pairs.get((ref_f[i], hyp_f[i]), 0) + 1
    ref_spk = sorted({r for (r, _), c in pairs.items() if c})
    hyp_spk = sorted({h for (_, h), c in pairs.items() if c})
    best_conf = 0
    mapping: Dict[str, str] = {}
    if ref_spk and hyp_spk:
        k = max(len(ref_spk), len(hyp_spk))
        names_h = hyp_spk + [f"__H{i}" for i in range(k - len(hyp_spk))]
        names_r = ref_spk + [f"__R{i}" for i in range(k - len(ref_spk))]
        best_conf = None
        for perm in itertools.permutations(names_h, len(names_r)):
            mp = dict(zip(names_r, perm))
            c = sum(cnt for (r, h), cnt in pairs.items() if mp.get(r) != h)
            if best_conf is None or c < best_conf:
                best_conf, mapping = c, mp
        best_conf = best_conf or 0
    else:
        best_conf = sum(pairs.values())

    speech_s = round(total_ref * FRAME, 2)
    der = (miss + fa + best_conf) / max(1, total_ref)
    return {
        "der": round(der, 4),
        "miss_s": round(miss * FRAME, 2),
        "fa_s": round(fa * FRAME, 2),
        "conf_s": round(best_conf * FRAME, 2),
        "scored_speech_s": speech_s,
        "n_ref_spk": len({v for v in ref_f if v}),
        "n_hyp_spk": len({v for v in hyp_f if v}),
        "collar": collar,
        "mode": mode,
        "mapping": {k: v for k, v in mapping.items() if not v.startswith("__")},
        "pairs": {
            "|".join(k): round(v * FRAME, 2)
            for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]
        },
    }


def jer(ref_turns: List[Dict[str, Any]], hyp_turns: List[Dict[str, Any]], total: float, collar: float = C.COLLAR) -> float:
    r = resolve_midpoints(ref_turns)
    h = resolve_midpoints(hyp_turns)
    rf = _frames(r, total)
    hf = _frames(h, total)
    excl = [False] * len(rf)
    prev = None
    for i, v in enumerate(rf):
        if v != prev:
            prev = v
            for k in range(max(0, i - int(round(collar / FRAME))), min(len(rf), i + int(round(collar / FRAME)))):
                excl[k] = True
    vals = []
    for spk in sorted({t["speaker"] for t in r}):
        best = 0.0
        for hsp in sorted({t["speaker"] for t in h}):
            inter = union = 0
            for i in range(len(rf)):
                if excl[i]:
                    continue
                ri = rf[i] == spk
                hi = hf[i] == hsp
                if ri and hi:
                    inter += 1
                if ri or hi:
                    union += 1
            best = max(best, inter / max(1, union))
        vals.append(1.0 - best)
    return round(sum(vals) / max(1, len(vals)), 4)


def score_diarization(ref_segments: List[Dict[str, Any]], hyp_turns: List[Dict[str, Any]],
                      total: float) -> Dict[str, Any]:
    ref_turns = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in ref_segments]
    ref_r = resolve_midpoints(ref_turns)
    hyp_r = resolve_midpoints(hyp_turns)
    rf = _frames(ref_r, total)
    hf = _frames(hyp_r, total)
    out: Dict[str, Any] = {
        "n_turns_hyp": len(hyp_turns),
        "n_turns_hyp_resolved": len(hyp_r),
        "n_turns_ref": len(ref_segments),
        "jer": jer(ref_turns, hyp_turns, total),
        "stages": {
            "ref_tiles": _score_labelling(rf, hf, total, "ref_tiles", C.COLLAR),
            "speech_compare": _score_labelling(rf, hf, total, "speech_compare", C.COLLAR),
            "strict_collar25": _score_labelling(rf, hf, total, "strict", C.COLLAR),
            "strict_nocollar": _score_labelling(rf, hf, total, "strict", 0.0),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Speaker agreement between two segment lists (element-level)
# ---------------------------------------------------------------------------
def segment_agreement(ref_segments: List[Dict[str, Any]], hyp_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    ref_turns = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in ref_segments]
    rf = _frames(resolve_midpoints(ref_turns), max(s["end"] for s in ref_segments) + 1.0)
    ok = unknown = total = 0
    wrong = []
    for h in hyp_segments:
        s = max(0, int(round(h["start"] / FRAME)))
        e = min(len(rf), int(round(h["end"] / FRAME)))
        if e <= s:
            unknown += 1
            continue
        votes: Dict[str, int] = {}
        for i in range(s, e):
            v = rf[i]
            if v:
                votes[v] = votes.get(v, 0) + 1
        total += 1
        if not votes:
            unknown += 1
            continue
        major = max(votes.items(), key=lambda kv: kv[1])[0]
        if major == h["speaker"]:
            ok += 1
        else:
            wrong.append({"start": round(h["start"], 2), "end": round(h["end"], 2),
                          "hyp": h["speaker"], "ref_major": major,
                          "text": (h.get("text") or "")[:70]})
    return {
        "segments_checked": total,
        "segments_ok": ok,
        "segments_wrong": len(wrong),
        "segments_unknown_ref": unknown,
        "pct_ok": round(100.0 * ok / max(1, total), 1),
        "wrong_segments": wrong,
    }


def segment_agreement_weighted(ref_segments: List[Dict[str, Any]],
                               hyp_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Speaker agreement weighted by words and by duration.

    Unweighted agreement counts a 0.26 s one-word backchannel ("طيب") exactly like a
    37 s sixty-word turn, which hides real progress: a system can fix every long
    turn while the *count* of segments stays dominated by short interjections that
    are intrinsically hard to attribute correctly.

    Word-weighted  = share of hypothesis words whose segment carries the correct
                     majority reference speaker.
    Duration-weighted = same, weighted by each segment's speech seconds.
    Both are computed over the same per-segment correctness decision as
    ``segment_agreement`` so the three numbers cannot contradict each other.
    """
    base = segment_agreement(ref_segments, hyp_segments)
    ref_turns = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in ref_segments]
    rf = _frames(resolve_midpoints(ref_turns), max(s["end"] for s in ref_segments) + 1.0)

    words_ok = words_total = 0
    dur_ok = dur_total = 0.0
    for h in hyp_segments:
        n_words = len((h.get("text") or "").split())
        dur = max(0.0, float(h["end"]) - float(h["start"]))
        s = max(0, int(round(h["start"] / FRAME)))
        e = min(len(rf), int(round(h["end"] / FRAME)))
        votes: Dict[str, int] = {}
        for i in range(s, e):
            v = rf[i]
            if v:
                votes[v] = votes.get(v, 0) + 1
        if not votes:
            continue  # no reference speech under this segment: excluded from both
        major = max(votes.items(), key=lambda kv: kv[1])[0]
        good = major == h["speaker"]
        words_total += n_words
        dur_total += dur
        if good:
            words_ok += n_words
            dur_ok += dur

    return {
        "pct_ok": base["pct_ok"],
        "agree_word": round(100.0 * words_ok / max(1, words_total), 1),
        "agree_duration": round(100.0 * dur_ok / max(1e-6, dur_total), 1),
        "words_counted": words_total,
        "duration_counted_s": round(dur_total, 2),
        "segments_checked": base["segments_checked"],
        "wrong_segments": base["wrong_segments"],
    }
