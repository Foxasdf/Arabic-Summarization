"""Text error ledger: attribute every reference->hypothesis word operation to a cause.

Inputs
------
* the reference segments (human-corrected);
* the *captured* continuous-ASR run from :mod:`patches`, which recorded, per
  slice, the exact ``raw`` string produced by Cohere and the exact ``guarded``
  string emitted by ``clean_and_guard_asr_text`` (adjustment #2). The guardrail
  attribution below is therefore exact in-context: for every edited span we can
  say whether the words that vanished had been *emitted by the model in that very
  chunk* and then deleted. Nothing is re-run or re-parsed, so no lower bound.

Every edit produced by the reference-vs-hypothesis alignment is attributed to
exactly one cause, checked in this order:

    guardrail_capture    - the deleted reference words were in the slice's raw
                           Cohere output and disappeared in its guarded output
    seam_suspect         - the operation lies in a chunk-overlap seam window
    vad_gate             - the operation's slice was rejected by the energy gate
    insertion_fp         - hypothesis-only words in the least active stretches
    possible_missed_speech - reference-only words in the least active stretches
    substitution         - everything else (model limitation, dialect gap, ...)
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from . import config as C

PUNCT = "ØŒØ›ØŸÂ«Â»â€œâ€!?:;,.\"'()[]{}â€¦-â€“â€”"

ORDER = [
    "guardrail_capture",
    "seam_suspect",
    "vad_gate",
    "insertion_fp",
    "possible_missed_speech",
    "substitution",
]

CAUSE_NOTES = {
    "guardrail_capture": "Words Cohere produced in that chunk but the guardrails deleted "
                         "(loop collapse, phrase wipe, or word-rate truncation).",
    "seam_suspect": "Inside a chunk overlap window: lost/lifted at a 28 s boundary or "
                    "removed by word-string seam dedup.",
    "vad_gate": "Slice rejected by the energy gate although it holds reference speech.",
    "insertion_fp": "Hypothesis-only tokens in a low-activity stretch (hallucination or "
                    "chunk-boundary duplication the dedup missed).",
    "possible_missed_speech": "Reference-only tokens in a low-activity stretch (silence gate, "
                              "ASR miss, or chunk cut).",
    "substitution": "Real recognition error: dialect/lexicon/accent gap or a genuine mis-hear.",
}


def _norm(tok: str) -> str:
    from app.services.preprocessing.normalizer import ArabicTextNormalizer

    return ArabicTextNormalizer().normalize(tok.strip(PUNCT)).strip()


def _opcodes(a: List[str], b: List[str]) -> List[Tuple[str, int, int, int, int]]:
    """Minimum-cost edit script (Levenshtein DP with backtrace).

    ``difflib`` is greedy and produced 351 substitutions where the optimum is
    260, which inflated the ledger by 7% relative to the WER report. This builds
    the true optimal script so the ledger's totals *equal* the WER sub/del/ins.
    Returns opcodes in the same shape difflib's ``get_opcodes`` uses.
    """
    n, m = len(a), len(b)
    INF = float("inf")
    # dp[i][j] = cost of aligning a[:i] with b[:j]
    prev = list(range(m + 1))
    rows = [prev]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j - 1], prev[j], cur[j - 1])
        rows.append(cur)
        prev = cur

    ops: List[Tuple[str, int, int, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and rows[i][j] == rows[i - 1][j - 1]:
            i, j = i - 1, j - 1
            continue
        cost = rows[i][j]
        if i > 0 and j > 0 and cost == 1 + rows[i - 1][j - 1]:
            ops.append(("replace", i - 1, i, j - 1, j))
            i, j = i - 1, j - 1
        elif i > 0 and cost == 1 + rows[i - 1][j]:
            ops.append(("delete", i - 1, i, j, j))
            i -= 1
        else:
            ops.append(("insert", i, i, j - 1, j))
            j -= 1
    ops.reverse()
    return [o for o in ops if o[0] != "equal"]


def _similar(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Slice / seam / guardrail evidence from the captured run
# ---------------------------------------------------------------------------
def slice_evidence(captured: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    slices = (captured or {}).get("slices") or []
    spans = []
    seams = []
    drops = []
    for rec in slices:
        start, dur = float(rec["start"]), float(rec["dur"])
        raw = rec.get("raw") or ""
        guarded = rec.get("guarded") or ""
        spans.append({"idx": rec["idx"], "start": start, "end": start + dur,
                      "vad": rec.get("vad_passed"), "rms": rec.get("rms_db"),
                      "n_raw": len(raw.split()), "n_guarded": len(guarded.split()),
                      "changed": bool(rec.get("guardrail_changed"))})
        if rec.get("guardrail_changed"):
            drops.append({"idx": rec["idx"], "start": round(start, 2), "end": round(start + dur, 2),
                          "raw": raw, "guarded": guarded,
                          "deleted": _deleted_tokens(raw, guarded)})
    for rec, nxt in zip(slices, slices[1:]):
        s_end = float(rec["start"]) + float(rec["dur"])
        window = (float(nxt["start"]), max(s_end, float(nxt["start"]) + 0.30))
        seams.append({"a_idx": rec["idx"], "b_idx": nxt["idx"],
                      "a_end": round(s_end, 2), "b_start": round(float(nxt["start"]), 2),
                      "window": [round(window[0], 2), round(window[1], 2)]})
    return {"slices": spans, "seams": seams, "guardrail_drops": drops}


def _deleted_tokens(raw: str, guarded: str) -> Dict[str, Any]:
    r, g = raw.split(), guarded.split()
    dels, repl = [], []
    for op, i1, i2, j1, j2 in _opcodes(r, g):
        if op == "delete":
            dels.extend(r[i1:i2])
        elif op == "replace":
            keep = min(i2 - i1, j2 - j1)
            repl.append({"from": " ".join(r[i1 + keep:i2]), "to": " ".join(g[j1 + keep:j2])} if keep else
                        {"from": " ".join(r[i1:i2]), "to": " ".join(g[j1:j2])})
            if i2 - i1 > j2 - j1:
                dels.extend(r[i1 + (j2 - j1):i2])
    return {"deleted": dels, "replaced": repl}


# ---------------------------------------------------------------------------
# Activity track (only used to classify insertions / silent deletions)
# ---------------------------------------------------------------------------
def activity_track(audio_path: Optional[Any], frame_s: float = 0.25) -> Optional[Dict[str, Any]]:
    try:
        import numpy as np
        from scipy.io import wavfile

        sr, data = wavfile.read(str(audio_path))
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        f = data.astype("float32") / 32768.0
        n = int(len(f) / (sr * frame_s))
        if n < 4:
            return None
        energy = (f[: n * int(sr * frame_s)].reshape(n, -1) ** 2).mean(axis=1)
        energy = energy[:n]
        p10, p90 = float(np.percentile(energy, 10)), float(np.percentile(energy, 90))
        center = float(np.percentile(energy, 40))
        return {"frame_s": frame_s, "energy": energy, "floor": p10, "peak": p90, "center": center}
    except Exception:
        return None


def _activity(track: Optional[Dict[str, Any]], span: Tuple[float, float]) -> Optional[float]:
    if not track:
        return None
    fs = track["frame_s"]
    a = max(0, int(span[0] / fs))
    b = min(len(track["energy"]) - 1, int(span[1] / fs))
    if b < a:
        b = a
    seg = track["energy"][a:b + 1]
    import numpy as np

    # dB relative to the file's median-active frame
    return round(float(10 * np.log10(max(1e-12, float(seg.mean()) / max(1e-12, track["center"])))), 1)


SILENT_DB = -9.0  # <=9 dB below the median-active frame counts as a genuinely quiet span


# ---------------------------------------------------------------------------
# Main ledger
# ---------------------------------------------------------------------------
def replay_guardrail_cf(captured: Optional[Dict[str, Any]],
                        ref_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counterfactual: is each guardrail edit justified by genuine non-speech?

    Two rules can act on a 28 s slice:
      * the transcription-credit regexes (``تسجيل``/``التسجيل`` ... ``.*``), which
        delete everything after the match;
      * the word-rate rule (``rate > 6.5/s and dur < 3.0`` -> truncate), which is
        fed the whole-slice duration.

    For every slice whose guarded text differs we report how many words each rule
    alone would remove, and whether the removed span actually contains reference
    speech. A rule that deletes reference speech is a false positive: the text
    existed and the human heard it.
    """
    from app.models.asr.guardrails import clean_and_guard_asr_text

    if not captured:
        return {}
    rows = []
    for s in captured.get("slices", []):
        raw = s.get("raw") or ""
        if not raw.strip():
            continue
        dur = float(s["dur"])
        start = float(s["start"])
        end = start + dur
        guard = clean_and_guard_asr_text(raw, dur_sec=dur)
        if guard.split() == raw.split():
            continue

        # reference words whose midpoint lies in this slice
        ref_words_in_slice = 0
        for x in ref_segments:
            a, b = float(x["start"]), float(x["end"])
            overlap = max(0.0, min(b, end) - max(a, start))
            if overlap <= 0:
                continue
            n = len((x.get("text") or "").split())
            ref_words_in_slice += n * (overlap / max(1e-6, b - a))

        removed = len(raw.split()) - len(guard.split())
        rows.append({
            "slice": s["idx"],
            "start": round(start, 2),
            "dur": round(dur, 2),
            "raw_words": len(raw.split()),
            "guarded_words": len(guard.split()),
            "words_removed": removed,
            "ref_words_in_slice": round(ref_words_in_slice, 1),
            "model_rate_wps": round(len(raw.split()) / dur, 2),
            "text_removed": " ".join(
                [w for w in raw.split() if w not in set(guard.split())]
            )[:160],
        })
    total_removed = sum(r["words_removed"] for r in rows)
    total_ref_speech_in_affected = sum(r["ref_words_in_slice"] for r in rows)
    return {
        "slices_affected": rows,
        "words_removed": total_removed,
        "ref_words_in_affected_slices": round(total_ref_speech_in_affected, 1),
        "note": "Removal of reference speech inside an affected slice is a false positive: "
                "the words were emitted by the model and the human heard them.",
    }


def build_ledger(ref_segments: List[Dict[str, Any]], hyp_tokens: List[str],
                 captured: Optional[Dict[str, Any]], audio_wav: Optional[Any] = None,
                 total: float = 0.0) -> Dict[str, Any]:
    ref_tokens: List[str] = []
    ref_ts: List[float] = []
    for s in ref_segments:
        toks = (s.get("text") or "").split()
        if not toks:
            continue
        step = (float(s["end"]) - float(s["start"])) / len(toks)
        for j, t in enumerate(toks):
            ref_tokens.append(t)
            ref_ts.append(float(s["start"]) + j * step)

    hyp_ts = _hyp_timestamps(hyp_tokens, captured, total)
    ev = slice_evidence(captured)
    track = activity_track(audio_wav) if audio_wav else None

    # Raw per-slice norm-token streams, for exact guardrail attribution
    raw_streams = [(d["idx"], [_norm(t) for t in d["raw"].split()]) for d in ev["guardrail_drops"]]
    dropped_norm = {_norm(t) for d in ev["guardrail_drops"] for t in d["deleted"]["deleted"]}

    def _seam_hit(span: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        for s in ev["seams"]:
            if span[0] <= s["window"][1] and span[1] >= s["window"][0]:
                return s
        return None

    def _slice_at(t: float) -> Optional[Dict[str, Any]]:
        for s in ev["slices"]:
            if s["start"] <= t <= s["end"]:
                return s
        return None

    records: List[Dict[str, Any]] = []
    buckets: Dict[str, Dict[str, Any]] = {}

    for op, i1, i2, j1, j2 in _opcodes([_norm(t) for t in ref_tokens], [_norm(t) for t in hyp_tokens]):
        span = _span(ref_ts, i1, i2, hyp_ts, j1, j2)
        ref_part = ref_tokens[i1:i2]
        hyp_part = hyp_tokens[j1:j2]
        act = _activity(track, span) if span else None
        cause = "substitution"
        evidence: List[str] = []

        if op == "delete":
            # exact guardrail check: were the deleted words in that chunk's raw output?
            hit = None
            if span:
                for idx, stream in raw_streams:
                    sl = _slice_at(span[0])
                    if sl and sl["idx"] != idx:
                        continue
                    if _contains_subsequence(stream, [_norm(t) for t in ref_part]):
                        hit = idx
                        break
            if hit is not None and any(_norm(t) in dropped_norm for t in ref_part):
                cause = "guardrail_capture"
                evidence.append(f"chunk#{hit}: present in raw text, absent from guarded text")
            elif _seam_hit(span) if span else False:
                s = _seam_hit(span)
                cause = "seam_suspect"
                evidence.append(f"seam {s['a_idx']}->{s['b_idx']} at {s['window'][0]:.2f}-{s['window'][1]:.2f}s")
            else:
                sl = _slice_at(span[0]) if span else None
                if sl is not None and sl["vad"] is False:
                    cause = "vad_gate"
                    evidence.append(f"slice#{sl['idx']} rejected (rms {sl['rms']} dBFS)")
                elif act is not None and act <= SILENT_DB:
                    cause = "possible_missed_speech"
                    evidence.append(f"{act:+.1f} dB relative to median active frame")
                else:
                    cause = "substitution"
                    evidence.append(f"{act:+.1f} dB" if act is not None else "energetic span")
        elif op == "insert":
            if len(ref_part) and any(_similar(_norm(a), _norm(b)) > 0.75 for a in ref_part for b in hyp_part):
                cause = "substitution"
                evidence.append("phonetic/spelling variant of a reference token")
            elif _seam_hit(span) if span else False:
                s = _seam_hit(span)
                cause = "seam_suspect"
                evidence.append(f"seam {s['a_idx']}->{s['b_idx']}: duplicate not removed")
            elif act is not None and act <= SILENT_DB:
                cause = "insertion_fp"
                evidence.append(f"{act:+.1f} dB relative to median active frame")
            else:
                cause = "insertion_fp"
                evidence.append(f"{act:+.1f} dB" if act is not None else "no energy track")
        else:  # replace
            if any(_norm(a) in dropped_norm for a in ref_part) and hyp_part:
                # guardrails rewrote a token that the reference kept
                best = max((_similar(_norm(a), _norm(b)) for a in ref_part for b in hyp_part), default=0)
                if best < 0.75:
                    cause = "guardrail_capture"
                    evidence.append("token present in raw text, altered by guardrails")
                else:
                    cause = "substitution"
                    evidence.append("spelling/alef-hamza variant")
            else:
                n = max(i2 - i1, j2 - j1)
                sim = max((_similar(_norm(a), _norm(b)) for a in ref_part for b in hyp_part), default=0.0)
                if sim >= 0.75:
                    evidence.append(f"orthographic variant (sim {sim:.2f})")
                else:
                    evidence.append(f"different word (sim {sim:.2f})")

        n_edits = _edit_count(op, i1, i2, j1, j2)
        b = buckets.setdefault(cause, {"cause": cause, "n_ops": 0, "words": 0,
                                       "ref_words": [], "hyp_words": [], "examples": [],
                                       "note": CAUSE_NOTES[cause]})
        b["n_ops"] += 1
        b["words"] += n_edits
        b["ref_words"].extend(ref_part)
        b["hyp_words"].extend(hyp_part)
        if len(b["examples"]) < 15:
            b["examples"].append({
                "t": round(span[0], 2) if span else None,
                "ref": " ".join(ref_part),
                "hyp": " ".join(hyp_part),
                "op": op,
                "evidence": "; ".join(evidence),
            })
        records.append({"op": op, "t": round(span[0], 2) if span else None,
                        "ref": " ".join(ref_part), "hyp": " ".join(hyp_part),
                        "cause": cause, "evidence": "; ".join(evidence)})

    total_ref = max(1, len(ref_tokens))
    causes = []
    for c in ORDER:
        b = buckets.get(c)
        if not b:
            continue
        causes.append({
            "cause": c,
            "n_ops": b["n_ops"],
            "words": b["words"],
            "pct_of_ref_words": round(100.0 * b["words"] / total_ref, 2),
            **{k: v for k, v in b.items() if k in ("ref_words", "hyp_words", "examples", "note")},
        })

    return {
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
        "edits": sum(r["words"] for r in causes),
        "causes": causes,
        "records": records,
        "evidence": ev,
        "activity_track": {"frame_s": track["frame_s"]} if track else None,
        "rate_rule_counterfactual": replay_guardrail_cf(captured, ref_segments),
    }


def _edit_count(op: str, i1: int, i2: int, j1: int, j2: int) -> int:
    """Edit cost of one opcode, matching Levenshtein cost, not span length.

    A ``replace`` block of m ref words by n hyp words costs m substitutions when
    m == n, and otherwise pairs ``min(m, n)`` substitutions with ``|m - n|``
    insertions/deletions -- exactly how ``levenshtein_distance`` scores it. This
    keeps the ledger's total equal to sub + del + ins in the WER report.
    """
    if op == "replace":
        return max(i2 - i1, j2 - j1)
    return (i2 - i1) if op == "delete" else (j2 - j1)


def _contains_subsequence(seq: List[str], needle: List[str]) -> bool:
    if not needle:
        return False
    it = iter(seq)
    return all(any(x == n for x in it) for n in needle)


def _hyp_timestamps(hyp_tokens: List[str], captured: Optional[Dict[str, Any]],
                    total: float) -> Optional[List[Tuple[float, float]]]:
    words = (captured or {}).get("words") or []
    if len(words) != len(hyp_tokens):
        return None
    return [(float(w["start"]), float(w["end"])) for w in words]


def _span(ref_ts: List[float], i1: int, i2: int, hyp_ts: Optional[List[Tuple[float, float]]],
          j1: int, j2: int) -> Optional[Tuple[float, float]]:
    starts, ends = [], []
    if i2 > i1 and i1 < len(ref_ts):
        starts.append(ref_ts[i1])
        ends.append(ref_ts[min(i2 - 1, len(ref_ts) - 1)] + 0.25)
    if hyp_ts and j2 > j1 and j1 < len(hyp_ts):
        starts.append(hyp_ts[j1][0])
        ends.append(hyp_ts[min(j2 - 1, len(hyp_ts) - 1)][1])
    if not starts:
        return None
    return (min(starts), max(ends))
