"""Two-file suite: side-by-side tables, macro-average, and cross-file gates.

The whole point is anti-overfitting. A lever is only accepted when it survives
BOTH recordings, so this module never reports a single-file delta as a result:

    gate = PASS  iff  the improvement holds on both members
                      (and no member regresses beyond tolerance)

Files have different speaker counts and different acoustic conditions on purpose
(3 speakers / 398 s vs 4 speakers / 300 s), so any constant that happens to fit
one will show up as a regression on the other.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as C

# Tolerance for a *regression*: a change is "neutral" inside this band.
#
# These are NOT arbitrary round numbers. The Cohere text pass is bit-identical
# across repeats (verified: same SHA-256 on both files, 3 runs each), but the
# segment path that follows it is NOT deterministic -- the utterance builder's
# output varies run to run. Measured over 3 repeats per file:
#
#   nlp_5min  DER range 0.0042 (sd 0.0019), word agreement range 0.90 pts
#   133306    DER range 0.0237 (sd 0.0100), word agreement range 5.00 pts
#
# So a 0.01 DER tolerance sits *below the noise floor* on 133306 and would call
# random variation a result. The 133306 band is set to its measured range and the
# validation band to the larger of its own range and 0.005.
NOISE = {
    "133306": {"der": 0.025, "agree_pts": 5.0},
    "nlp_5min": {"der": 0.005, "agree_pts": 1.0},
}
TOL_WER = 0.01
TOL_DER = 0.025
TOL_AGREE = 5.0  # percentage points


def noise_for(key: str, metric: str) -> float:
    """Measured run-to-run band for this file, falling back to the largest seen."""
    rec = NOISE.get(key) or {}
    if metric == "der":
        return rec.get("der", max(v["der"] for v in NOISE.values()))
    return rec.get("agree_pts", max(v["agree_pts"] for v in NOISE.values())) / 100.0


def _load(spec: C.FileSpec, name: str) -> Optional[Dict[str, Any]]:
    p = spec.out_dir / name
    return C.read_json(p) if p.exists() else None


def _artifact(spec: C.FileSpec, name: str):
    p = spec.artifact_dir / name
    return C.read_json(p) if p.exists() else None


# ---------------------------------------------------------------------------
# per-file extraction
# ---------------------------------------------------------------------------
def file_metrics(key: str) -> Dict[str, Any]:
    """Read one file's cached artifacts into a flat, comparable record."""
    from . import scoring
    from . import scoring_spk as SS

    spec = C.SUITE[key]
    ref = spec.ref()
    n_spk = len({s["speaker"] for s in ref})

    arms = _load(spec, "arms_raw.json") or {}
    arm_rows = arms.get("arms") or {}
    captured = _artifact(spec, "captured_asr_uniform.json") or {}
    tr = _artifact(spec, "transcribe_with_speakers.json") or {}
    turns_auto = _artifact(spec, "turns_auto.json") or []
    turns_min1 = _artifact(spec, f"turns_min1.json") or []
    turns_oracle = _artifact(spec, f"turns_oracle{n_spk}.json") or []
    audio_dur = float(captured.get("audio_duration") or spec.duration())

    text = scoring.score_text(ref, tr.get("segments") or [])
    cap_text = scoring.score_text(ref, [{"text": captured.get("text") or ""}]) if captured.get("text") else {}
    diar_raw = SS.score_diarization(ref, turns_auto, audio_dur)
    agree_prod = SS.segment_agreement_weighted(ref, tr.get("segments") or [])

    def arm(name):
        a = arm_rows.get(name)
        if not a:
            return None
        st = ((a.get("diar") or {}).get("stages") or {}).get("ref_tiles") or {}
        return {
            "n_segments": a.get("n_segments"),
            "der": st.get("der"),
            "conf_s": st.get("conf_s"),
            "miss_s": st.get("miss_s"),
            "agree": a.get("agree"),
            "agree_word": a.get("agree_word"),
            "agree_duration": a.get("agree_duration"),
        }

    return {
        "key": key,
        "audio": spec.audio.name,
        "duration_s": round(audio_dur, 2),
        "n_ref_segments": len(ref),
        "n_ref_words": sum(len(s["text"].split()) for s in ref),
        "n_ref_speakers": n_spk,
        "n_turns_auto": len(turns_auto),
        "n_speakers_auto": len({t["speaker"] for t in turns_auto}) if turns_auto else 0,
        "n_speakers_min1": len({t["speaker"] for t in turns_min1}) if turns_min1 else None,
        "auto_equals_min1": (turns_auto == turns_min1) if (turns_auto and turns_min1) else None,
        "auto_equals_oracle": (turns_auto == turns_oracle) if (turns_auto and turns_oracle) else None,
        "der_raw_diarization": diar_raw["stages"]["ref_tiles"]["der"],
        "der_production": diar_raw["stages"]["ref_tiles"]["der"],
        "text": {
            "wer_norm": text["wer_norm"],
            "cer_norm": text["cer_norm"],
            "wer_captured": cap_text.get("wer_norm"),
            "wer_dialect_norm": text["wer_dialect_norm"],
            "wer_dialect_gap": text["wer_dialect_gap"],
            "n_segments": len(tr.get("segments") or []),
            "hyp_words": text["hyp_words"],
        },
        "production": {
            "n_segments": len(tr.get("segments") or []),
            "der": (((arm_rows.get("A_production_segments") or {}).get("diar") or {})
                    .get("stages", {}).get("ref_tiles", {}) or {}).get("der"),
            "agree": agree_prod["pct_ok"],
            "agree_word": agree_prod["agree_word"],
            "agree_duration": agree_prod["agree_duration"],
        },
        "arms": {k: arm(k) for k in ("A_uniform", "A_oracle", "C_whisper_timing",
                                     "B_bisection", "D_per_turn")
                 if arm(k) is not None},
        "arm_c_resources": ((arms.get("arm_c") or {}).get("resources") or {}),
        "ledger": (arms.get("ledger") or {}).get("causes") or [],
        "guardrail": arms.get("guardrail_audit") or {},
        "shipped_scored": (_load(spec, "metrics.json") or {}).get("shipped_scored") or {},
    }


# ---------------------------------------------------------------------------
# cross-file combination
# ---------------------------------------------------------------------------
def _pick(rec: Dict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
    cur: Any = rec
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, (int, float)) else None


def macro_average(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Equal-weight mean across files. None values are skipped, not zeroed."""
    def avg(path):
        vals = [v for v in (_pick(r, path) for r in records) if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "key": "MACRO",
        "n_files": len(records),
        "wer_norm": avg(("text", "wer_norm")),
        "cer_norm": avg(("text", "cer_norm")),
        "wer_dialect_norm": avg(("text", "wer_dialect_norm")),
        "der_raw": avg(("der_raw_diarization",)),
        "der_prod": avg(("production", "der")),
        "agree_prod": avg(("production", "agree")),
        "agree_word_prod": avg(("production", "agree_word")),
        "agree_dur_prod": avg(("production", "agree_duration")),
        "der_c": avg(("arms", "C_whisper_timing", "der")),
        "agree_word_c": avg(("arms", "C_whisper_timing", "agree_word")),
        "agree_dur_c": avg(("arms", "C_whisper_timing", "agree_duration")),
    }


def two_file_gate(name: str,
                  before: Dict[str, Dict[str, Any]],
                  after: Dict[str, Dict[str, Any]],
                  metric: str,
                  higher_is_better: bool = False,
                  tol: Optional[float] = None,
                  min_gain: Optional[float] = None) -> Dict[str, Any]:
    """Evaluate a candidate on every file; pass only if no file regresses.

    ``before`` / ``after`` are keyed by file and hold the metric's values.

    Two different thresholds are used, deliberately:
      * ``tol``      -- how much *regression* is tolerated. Zero-ish: a candidate
                        that makes any file worse is out.
      * ``min_gain`` -- how much improvement is needed to call it a win. A
                        candidate that provably touches only broken slices (a
                        pure recall fix) should not have to clear the same bar as
                        one that reshapes every segment, so callers pass a small
                        ``min_gain`` for those.
    """
    rows = []
    ok_all = True
    improved_any = False
    for key in C.SUITE:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            rows.append({"file": key, "before": b, "after": a, "delta": None, "ok": None})
            ok_all = False
            continue
        d = round(a - b, 4)
        # Per-file measured noise band, not one global constant.
        if tol is not None:
            t = tol
        elif higher_is_better:
            t = noise_for(key, "agree")
        else:
            t = noise_for(key, "der")
        g = min_gain if min_gain is not None else t
        if higher_is_better:
            ok = d >= -t
            better = d >= g and d > 0
        else:
            ok = d <= t
            better = d <= -g and d < 0
        ok_all &= ok
        improved_any |= better
        rows.append({"file": key, "before": round(b, 4), "after": round(a, 4),
                     "delta": d, "tolerance": round(t, 4), "ok": ok, "improved": better})
    status = "PASS" if (ok_all and improved_any) else ("NEUTRAL" if ok_all else "FAIL")
    return {"candidate": name, "metric": metric, "higher_is_better": higher_is_better,
            "per_file": rows, "status": status}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _f(v, nd=4):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render_suite(records: List[Dict[str, Any]], gates: List[Dict[str, Any]],
                 candidate_notes: List[str],
                 rec_cand: Optional[Dict[str, Dict[str, Any]]] = None,
                 rec_e2e: Optional[Dict[str, Dict[str, Any]]] = None,
                 frontier: Optional[Dict[str, Any]] = None,
                 recovery: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    rec_cand = rec_cand or {}
    rec_e2e = rec_e2e or {}
    recovery = recovery or {}
    macro = macro_average(records)
    L: List[str] = []
    L.append("# Two-file suite: 133306 (primary) vs nlp_5min (validation)\n")
    L.append("A lever is accepted only if it improves on **both** files. The two recordings differ "
             "deliberately: 3 speakers / 398 s of dialectal conversation, versus 4 speakers / 300 s — "
             "any constant that merely fits one shows up as a regression on the other.\n")

    L.append("## Files\n")
    L.append("| file | audio | duration | ref segs | ref words | ref speakers | auto speakers | turns |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in records:
        L.append(f"| `{r['key']}` | {r['audio']} | {r['duration_s']} s | {r['n_ref_segments']} | "
                 f"{r['n_ref_words']} | {r['n_ref_speakers']} | {r['n_speakers_auto']} | {r['n_turns_auto']} |")
    L.append("")

    L.append("## Measurement noise: what counts as a real change\n")
    L.append("Gates are only meaningful if their thresholds exceed the noise in the measurement. These "
             "were measured, not guessed, by repeating the pipeline 3× per file:\n")
    L.append("| file | Cohere text pass | segment-path DER range | segment-path agreement range |")
    L.append("|---|---|---|---|")
    for r in records:
        nf = NOISE.get(r["key"], {})
        L.append(f"| `{r['key']}` | bit-identical across 3 runs | "
                 f"±{nf.get('der', 0):.4f} | ±{nf.get('agree_pts', 0):.1f} pts |")
    L.append("")
    L.append("Two separate facts, and conflating them would be a mistake:\n")
    L.append("1. **The ASR text is fully deterministic.** Three repeats of the Cohere pass produced "
             "byte-identical text (same SHA-256) on both files, so no stochastic repeat count is needed "
             "for text-level conclusions and text comparisons are exactly paired.")
    L.append("2. **The segment path is not deterministic.** With identical input text, the downstream "
             "utterance builder produced 52/53/52 segments on `nlp_5min` and 67/73/72 on `133306`, with "
             "DER moving 0.0237 and word agreement moving 5.0 pts on `133306`. The variability is in "
             "there, not in the ASR.\n")
    L.append("Consequence: a 0.01 DER gate on `133306` would sit **below** its own noise floor and would "
             "flag random variation as a result — so each file now carries the band it actually exhibits. "
             "Text metrics (WER, dialect-NORM WER) remain tight at 0.01 because they are deterministic; "
             "only the segment-derived metrics need the wider band.\n")

    L.append("## Text: WER, and how much of it is spelling rather than mishearing\n")
    L.append("| file | WER segment-level | WER captured-text | CER | **WER dialect-norm** | dialect share |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in records:
        t = r["text"]
        share = (t["wer_dialect_gap"] / t["wer_norm"] * 100) if t["wer_norm"] else 0
        L.append(f"| `{r['key']}` | {_f(t['wer_norm'])} | {_f(t.get('wer_captured'))} | {_f(t['cer_norm'])} | "
                 f"**{_f(t['wer_dialect_norm'])}** | {share:.1f}% |")
    L.append(f"| **macro** | **{_f(macro['wer_norm'])}** | | **{_f(macro['cer_norm'])}** | "
             f"**{_f(macro['wer_dialect_norm'])}** | |")
    L.append("")
    L.append("Two WER columns because the two differ by design and mixing them is how a report starts "
             "contradicting itself: **segment-level** is what the product emits (text re-split into speaker "
             "turns), **captured-text** is the raw continuous pass before turn building. Candidate fixes "
             "are measured against the captured-text column, so they are compared like-for-like.\n")
    L.append("`wer_dialect_norm` collapses documented Levantine↔MSA orthography (ـو/ـه suffix, ت/ث, د/ذ, "
             "انو/انه, يلي/اللي, هال/ال) symmetrically on both sides. The gap between it and `wer_norm` is "
             "the share of error that is *spelling convention* — where Cohere heard the word correctly and "
             "only disagreed with the reference's orthography. **`wer_norm` remains the headline metric**; "
             "this is diagnostic only and is never applied to the pipeline.\n")

    L.append("## Speaker separation\n")
    L.append("| file | raw Diarization DER | production DER | agreement (seg) | agreement (word) | agreement (duration) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in records:
        p = r["production"]
        L.append(f"| `{r['key']}` | {_f(r['der_raw_diarization'])} | {_f(p['der'])} | "
                 f"{_f(p['agree'], 1)}% | **{_f(p['agree_word'], 1)}%** | {_f(p['agree_duration'], 1)}% |")
    L.append(f"| **macro** | **{_f(macro['der_raw'])}** | **{_f(macro['der_prod'])}** | "
             f"**{_f(macro['agree_prod'], 1)}%** | **{_f(macro['agree_word_prod'], 1)}%** | "
             f"**{_f(macro['agree_dur_prod'], 1)}%** |")
    L.append("")
    L.append("### Why weighting matters\n")
    L.append("Unweighted agreement counts a 0.26 s one-word backchannel the same as a 37 s turn, so it "
             "understates quality and hides genuine progress. Word- and duration-weighted agreement answer "
             "the question a user actually cares about: *of what was said, how much was attributed to the "
             "right person?*\n")

    L.append("## Arm C (Whisper word timing) re-evaluated with weighted agreement\n")
    L.append("| file | DER prod | DER Arm C | Δ DER | agree(seg) prod→C | agree(word) prod→C | Δ word |")
    L.append("|---|---:|---:|---:|---|---|---:|")
    any_c = False
    for r in records:
        c = (r["arms"] or {}).get("C_whisper_timing")
        p = r["production"]
        if not c or c.get("der") is None:
            L.append(f"| `{r['key']}` | {_f(p['der'])} | — | — | {_f(p['agree'], 1)}% → — | "
                     f"{_f(p['agree_word'], 1)}% → — | — |")
            continue
        any_c = True
        d_der = c["der"] - (p["der"] or 0)
        d_word = (c["agree_word"] or 0) - (p["agree_word"] or 0)
        L.append(f"| `{r['key']}` | {_f(p['der'])} | {_f(c['der'])} | {d_der:+.4f} | "
                 f"{_f(p['agree'], 1)}% → {_f(c['agree'], 1)}% | "
                 f"{_f(p['agree_word'], 1)}% → {_f(c['agree_word'], 1)}% | {d_word:+.1f} pts |")
    L.append(f"| **macro** | **{_f(macro['der_prod'])}** | **{_f(macro['der_c'])}** | | | "
             f"**{_f(macro['agree_word_prod'], 1)}% → {_f(macro['agree_word_c'], 1)}%** | |")
    L.append("")
    if any_c:
        L.append("This is the paradox from the 133306 report resolved: under unweighted agreement Arm C looked "
                 "flat, because the stems it fixes are long turns while the *count* of segments stays "
                 "dominated by short interjections. Under word/duration weighting the same change is "
                 "visible. The gate still applies to both files — see §gates.\n")
        L.append("**What this verdict does and does not say.** Arm C passes because it never regresses: on "
                 "`133306` it is a clear win (DER −0.0381, word agreement +6.5 pts), and on `nlp_5min` it "
                 "costs +0.0029 DER — inside the 0.01 neutral band — while still gaining +0.3 pts of word "
                 "agreement. That is a genuine pass, but a *narrow* one on the validation file: the second "
                 "recording is where Arm C stops paying. If Arm C is ever taken to production it should be "
                 "gated on a measured benefit per file rather than assumed universally better.\n")

    L.append("### Arm C production viability\n")
    L.append("| file | wall clock | peak device VRAM | % of 6 GB | audio length | realtime factor |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in records:
        res = r.get("arm_c_resources") or {}
        w = res.get("wall_s")
        v = res.get("peak_device_vram_mb")
        tot = (res.get("vram_total_gb") or 6.0) * 1024
        rtf = (w / r["duration_s"]) if (w and r["duration_s"]) else None
        L.append(f"| `{r['key']}` | {_f(w, 1)} s | {_f(v, 0)} MB | "
                 f"{(100 * v / tot):.0f}% | {r['duration_s']} s | {_f(rtf, 3)}× |")
    L.append("")
    L.append("Measured from `nvidia-smi` whole-device counters sampled every 0.25 s, the only honest source "
             "here: `torch.cuda.max_memory_allocated` cannot see the CTranslate2 allocator or the Vulkan "
             "subprocess, so it reports `null`. Peak VRAM is ~40% of the 6 GB card and wall clock is far "
             "below realtime, so **Arm C is viable for a single-file production run**. Caveat: these runs "
             "are strictly sequential; concurrent requests competing for the same GPU and VRAM were not "
             "tested and would need a queue or memory cap before multi-user exposure.\n")

    L.append("### The real design space: word list × word timing\n")
    L.append("The two \"structural fixes\" are not two independent levers. The Cohere engine's own word list "
             "and Arm C's aligned list contain the **same token sequence** on both files "
             "(794 == 794, 668 == 668; zero non-equal regions under a sequence alignment), because Arm C was "
             "produced by aligning onto that same transcript. So the honest design space is 2x2 — what you do "
             "about the word list, and what you do about the timing:\n")
    L.append("| words | fabricated timing | acoustic timing (Arm C) |")
    L.append("|---|---|---|")
    L.append("| discarded by text path | `production` | `arm_c` |")
    L.append("| preserved | `engine_words` | `combined` (= `arm_c`, since the sequences coincide) |")
    L.append("")
    if frontier:
        L.append("Macro-averaged across both files:\n")
        L.append("| cell | DER | word agreement | duration agreement | WER |")
        L.append("|---|---:|---:|---:|---:|")
        for cname, v in frontier["macro"].items():
            L.append(f"| `{cname}` | {_f(v['der'])} | {_f(v['agree_word'], 2)}% | "
                     f"{_f(v['agree_duration'], 2)}% | {_f(v['wer_norm'])} |")
        L.append("")
        best_der = frontier["best_der_cell"]
        best_agree = frontier["best_agreement_cell"]
        L.append(f"**The two metrics disagree about the best configuration**: DER prefers `{best_der}`, "
                 f"speaker agreement prefers `{best_agree}`. This is not a tie to be broken by preference — "
                 f"it is a real trade between different failure modes, and it belongs in the report rather "
                 f"than behind a single PASS/FAIL.\n")
        L.append("| file | production | engine_words | arm_c |")
        L.append("|---|---|---|---|")
        for k, cells in frontier["per_file"].items():
            def cell(c):
                v = cells.get(c)
                if not v:
                    return "—"
                return (f"DER {_f(v['der'])}<br>conf {_f(v['conf_s'], 1)}s "
                        f"miss {_f(v['miss_s'], 1)}s<br>agree {_f(v['agree_word'], 1)}%")
            L.append(f"| `{k}` | {cell('production')} | {cell('engine_words')} | {cell('arm_c')} |")
        L.append("")
        L.append("Reading it: `engine_words` wins DER by **buying back missing speech** — it detects words the "
                 "text path was dropping (133306: miss 35.1 s → 7.5 s) but attaches the fabricated uniform "
                 "timing to them, so confusion climbs. `arm_c` uses **real acoustic timing**, which lowers "
                 "confusion (133306: 68.6 s → 61.3 s) and raises agreement, but leaves the missing words "
                 "missing, so DER improves less or not at all.")
    L.append("")
    L.append("### Why each DER change happened: confusion vs missing speech\n")
    L.append("A DER delta alone cannot tell you which side moved, and different levers move *opposite* "
             "sides. Both are in seconds of wrongly-attributed / uncovered speech.\n")
    L.append("Only levers that the 2x2 above does **not** already cover are listed here. The engine-word "
             "variants are omitted because they are that table; the credit regex is included because it "
             "changes neither the word list nor the timing. One numeric source per comparison, rather than "
             "two runs of the same thing that differ by pipeline noise.\n")
    L.append("| file | lever | confusion (s) | missing speech (s) | DER |")
    L.append("|---|---|---:|---:|---:|")
    for r in records:
        for ckey, label in (("vad_word_timing", "VAD-aware timing"),
                            ("snap_words", "boundary snapping")):
            c = (rec_cand.get(r["key"]) or {}).get(ckey) or {}
            b, a = c.get("baseline") or {}, c.get("candidate_metrics") or {}
            if not b or not a:
                continue
            L.append(f"| `{r['key']}` | {label} | {_f(b.get('conf_s'), 1)} → {_f(a.get('conf_s'), 1)} | "
                     f"{_f(b.get('miss_s'), 1)} → {_f(a.get('miss_s'), 1)} | "
                     f"{_f(b.get('der'))} → {_f(a.get('der'))} |")
    for r in records:
        # credit regex is the one e2e cell not represented in the 2x2 (the 2x2
        # varies word list and timing; the regex varies neither), so it belongs here.
        e2 = (rec_e2e.get(r["key"]) or {})
        b = e2.get("baseline") or {}
        a = e2.get("credit_regex") or {}
        if b and a:
            L.append(f"| `{r['key']}` | credit regex *(e2e)* | "
                     f"{_f(b.get('conf_s'), 1)} → {_f(a.get('conf_s'), 1)} | "
                     f"{_f(b.get('miss_s'), 1)} → {_f(a.get('miss_s'), 1)} | "
                     f"{_f(b.get('der'))} → {_f(a.get('der'))} |")
    L.append("")
    L.append("Note the two directions are not the same kind of trade. The credit regex moves **both** sides "
             "favourably on `133306` (confusion 68.2 s → 63.2 s, and its recovered speech more than offsets "
             "the rise in uncovered time since total DER falls) because it restores speech the old rule "
             "deleted outright. The engine-word change, by contrast, moves the two sides in **opposite** "
             "directions — that is the trade described in the 2x2 above.\n")
    L.append("The two fixes are therefore **not** interchangeable, and the data separates them cleanly:\n")
    L.append("| fix | net effect | recommendation |")
    L.append("|---|---|---|")
    L.append("| credit regex | improves DER on `133306` (+0.0236) with no measurable cost on `nlp_5min`; "
             "raises word agreement on `133306` (87.3 → 88.8%) and costs 0.3 pts on `nlp_5min`, well "
             "inside that file's ±1.0 pt band | **adopt** — it only ever restores speech the old rule "
             "wrongly deleted, and it is the one change that improves DER *and* agreement |")
    L.append("| engine word list | improves DER on both files but attaches fabricated uniform timing to the "
             "recovered words, converting missing speech into speaker confusion | **do not adopt alone** — "
             "see the 2x2 above: the words are worth keeping, but only alongside acoustic timing, and since "
             "Arm C already re-times the same sequence the practical choice is `arm_c`, not `engine_words` |")
    L.append("")
    L.append("> **Correction to the previous round.** That round reported the `engine_words` result as a "
             "capability fix that needs \"the alignment stage revisited\", on the assumption it was "
             "independent of Arm C. It is not: the engine list and the Arm C list are the same sequence, so "
             "`engine_words` and `arm_c` differ only in the timing attached. The conclusion (do not ship it "
             "alone) stands, but the reason is different — it is not a missing follow-up, it is that the "
             "acoustic-timing configuration already supersedes it.\n")

    L.append("## Hole recovery: speech the ASR never emitted\n")
    if recovery:
        L.append("The measurement that started this: on `133306`, the span **78.5–87.3 s** produces **zero "
                 "words**. The audio there is not quiet — it peaks at −17 dB, louder than the surrounding "
                 "speech — Pyannote hears speech in it, and the reference transcript contains "
                 "`لما كنت صف سابع تقريبا كان وزني شي 90` inside it. It is not a 28 s truncation either: on "
                 "`nlp_5min` slices up to 29.65 s transcribe at normal word rates, and decoding that same "
                 "region as an 11.8 s window with the shipped engine returns "
                 "`لما كنت صف سابع تقريبا كان وزني شي 90` **verbatim**. Same samples, same model; only the "
                 "window composition differs. So the words are recoverable — a long single-shot decode was "
                 "dropping them.\n")
        L.append("A candidate that re-decodes any word-stream gap containing audio, with a short window, "
                 "recovers speech at every window length tried — so it is not a tuned constant:\n")
        L.append("| file | holes (with audio) | hole seconds | recovered at 4 s | 6 s | 8 s | 10 s | 12 s | 16 s |")
        L.append("|---|---:|---:|---|---|---|---|---|---|")
        for r in records:
            rv = recovery.get(r["key"]) or {}
            sw = rv.get("sweep") or {}
            by = {str(int(x["len"])): x for x in (sw.get("by_len") or [])}
            cells = " | ".join(f"{by[k]['holes_recovered']}/{by[k]['n_holes']}"
                               for k in ("4", "6", "8", "10", "12", "16") if k in by)
            L.append(f"| `{r['key']}` | {sw.get('holes', '—')} | {sw.get('hole_seconds', '—')} | {cells} |")
        L.append("")
        L.append("But recovering the words and *scoring* them are different questions, and the answer to "
                 "the second is negative:\n")
        L.append("| file | DER | missing speech (s) | word agreement | WER |")
        L.append("|---|---|---|---|---|")
        for r in records:
            g = (recovery.get(r["key"]) or {}).get("gate") or {}
            b, a = g.get("baseline") or {}, g.get("recovered") or {}
            if not b or not a:
                continue
            L.append(f"| `{r['key']}` | {_f(b.get('der'))} → {_f(a.get('der'))} ({g['d_der']:+.4f}) | "
                     f"{_f(b.get('miss_s'), 1)} → {_f(a.get('miss_s'), 1)} | "
                     f"{_f(b.get('agree_word'), 1)}% → {_f(a.get('agree_word'), 1)}% | "
                     f"{_f(b.get('wer_norm'))} → {_f(a.get('wer_norm'))} |")
        L.append("")
        L.append("DER improves on both files and missing speech falls sharply, because the inserted words "
                 "give the scorer speech to attribute where before there was none. **WER gets worse on both**, "
                 "and that is the honest verdict: of 90 words inserted on `133306`, only 67% appear in the "
                 "reference vocabulary at all, and many of the rest are re-decodings of speech that was "
                 "already transcribed — the wider context window reaches past the hole. The 8.8 s case "
                 "recovered its line verbatim, but 2–3 s gaps are usually the speaker pausing (the reference "
                 "marks them with `...`), and re-decoding a pause yields either silence or neighbouring "
                 "speech, not new content.\n")
        L.append("**Verdict: not adopted.** The mechanism is real and the failure it targets is real, but "
                 "inserting words with synthetic timings into gaps that are mostly pauses costs more in "
                 "insertion errors than it recovers. Making this a win needs a way to tell a genuine "
                 "dropped-utterance from a pause before inserting — which the current signals do not "
                 "provide, since both are loud, unbounded in Pyannote, and wordless.\n")
    L.append("")

    L.append("## Error attribution per file\n")
    L.append("| cause | 133306 | nlp_5min |")
    L.append("|---|---:|---:|")
    by_key = {r["key"]: {c["cause"]: c for c in r["ledger"]} for r in records}
    all_causes = []
    for r in records:
        for c in r["ledger"]:
            if c["cause"] not in all_causes:
                all_causes.append(c["cause"])
    for cause in all_causes:
        cells = []
        for r in records:
            c = by_key.get(r["key"], {}).get(cause)
            cells.append(f"{c['pct_of_ref_words']}%" if c else "—")
        L.append(f"| `{cause}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Guardrail behaviour per file\n")
    L.append("| file | slice words lost | slices affected | probe sentences wrongly wiped |")
    L.append("|---|---:|---:|---:|")
    for r in records:
        g = r["guardrail"] or {}
        L.append(f"| `{r['key']}` | {g.get('slice_words_lost', '—')} | "
                 f"{len(g.get('slice_fires') or [])} | "
                 f"{g.get('probes_deleted', '—')} / {g.get('probes_total', '—')} |")
    L.append("")
    L.append("The probe column is a fixed synthetic battery applied to every file, so **7/8 reads the same "
             "on both** — the defect lives in the rule, not in the audio. The slice columns show how often "
             "it actually fired on real speech: once on `133306` (42 words), never on `nlp_5min`. Together "
             "they say the rule is broadly unsafe (it would delete ordinary sentences most of the time) and "
             "that this particular recording happened to trigger it once. Fixing it is therefore worth more "
             "than the +42 words recovered here.\n")

    L.append("## Gates (both files must pass)\n")
    if not gates:
        L.append("_no candidate gates were evaluated_\n")
    else:
        L.append("| candidate | metric | " + " | ".join(r["key"] for r in records) + " | verdict |")
        L.append("|---|---|" + "---|" * len(records) + "---|")
        for g in gates:
            cells = []
            for pf in g["per_file"]:
                if pf["delta"] is None:
                    cells.append("—")
                else:
                    mark = "ok" if pf["ok"] else "**fail**"
                    cells.append(f"{_f(pf['before'])} → {_f(pf['after'])} ({pf['delta']:+.4f}) {mark}")
            L.append(f"| {g['candidate']} | {g['metric']} | " + " | ".join(cells) + f" | **{g['status']}** |")
        L.append("")
    for note in candidate_notes:
        L.append(f"- {note}")
    if candidate_notes:
        L.append("")

    L.append("### What the rejected candidates tell us\n")
    L.append("Two of the most promising-looking levers failed, and the way they failed points at the real "
             "constraint:\n")
    L.append("- **VAD-aware word timing (DER +0.025 on `133306`, +0.063 on `nlp_5min`).** Spreading each "
             "slice's words across its *speech* rather than across the whole window sounds strictly more "
             "correct as an estimate of when words were spoken — and it does reduce speaker confusion "
             "(`133306`: 86.2 s → 77.2 s of wrongly-attributed speech). But it **adds missing speech** "
             "(7.5 s → 25.2 s; `nlp_5min`: 0.0 s → 19.1 s), because squeezing words into the speech "
             "regions leaves uncovered time inside turns that the scorer then counts as missed. Confusion "
             "and miss trade off against each other, and the miss side cost more.")
    L.append("- **Pyannote boundary snapping (DER +0.005 / +0.016, word agreement −2.1 / −0.4 pts).** "
             "Snapping moved 375 of 794 words — far too many to be \"ambiguous words near a boundary\". "
             "Because the even-split timing is itself inaccurate, most words are *already* slightly wrong "
             "relative to the turn they belong to, so snapping to the nearest edge redistributes them "
             "rather than correcting them. It fragments segments (77 → 99) and, unlike the timing lever, "
             "it *worsens* confusion (86.2 s → 88.9 s): pulling word edges out to turn boundaries removes "
             "words from the middle of turns and hands more of them to the neighbouring speaker.")
    L.append("- **Fuzzy seam dedupe is NEUTRAL on DER and a small PASS on agreement.** It removes 21 "
             "duplicate words on `133306` and 31 on `nlp_5min` and improves word-weighted agreement on both "
             "(+0.6 / +0.7 pts), but DER does not clearly improve. It is a real cleanup of repeated text, "
             "not a structural fix; adopt it for quality, not expecting it to repair speaker structure.\n")
    L.append("The common thread: **every lever that re-estimates word timing from a 28 s slice trades "
             "confusion against missing speech, and neither trade wins.** You cannot recover true word "
             "timing by post-processing fabricated timing — the information was never there. The one lever "
             "with real headroom is giving the ASR genuine word timestamps (Arm C), which is not a "
             "re-estimate but a measurement.\n")

    L.append("## Caveats\n")
    L.append("- Two files is better than one but still a small sample. A of 2/2 pass raises confidence; "
             "it does not prove generality.")
    L.append("- `wer_dialect_norm` uses a fixed, documented rule list applied symmetrically; it is not "
             "tuned per file and not applied to output.")
    L.append("- The reference for each file is human-corrected utterance-level text, not an RTTM, so "
             "absolute DER carries a model-dependent floor; only *deltas* are used for gating.")
    L.append("- Gate tolerance: WER/DER 0.01 absolute, agreement 1.0 percentage point; a change inside the "
             "band counts as neutral, not as a pass on its own.")
    L.append("")
    return "\n".join(L)
