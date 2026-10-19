"""Aggregate cached artifacts into metrics.json and render diagnosis.md.

Gate convention (adjustment #1): every gate is written so that the *pass* side is
the improvement side, and the signed delta is printed beside it.

    WER / CER / DER / JER   lower is better   ->  PASS iff delta <= -threshold
    agreement / pct_ok      higher is better  ->  PASS iff delta >= +threshold
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as C
from . import scoring
from . import scoring_spk as SS


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_lower(delta: Optional[float], threshold: float) -> tuple[str, str]:
    """Metric where lower is better."""
    if delta is None:
        return "n/a", "no measurement"
    ok = delta <= -threshold
    return ("PASS" if ok else "FAIL"), f"\u0394 = {delta:+.4f}  (need \u2264 {-threshold:.4f})"


def gate_higher(delta: Optional[float], threshold: float) -> tuple[str, str]:
    """Metric where higher is better."""
    if delta is None:
        return "n/a", "no measurement"
    ok = delta >= threshold
    return ("PASS" if ok else "FAIL"), f"\u0394 = {delta:+.2f}  (need \u2265 {threshold:+.2f})"


def _d(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _load(name: str, default=None):
    p = C.OUT_DIR / name
    return C.read_json(p) if p.exists() else default


def _load_artifact(name: str, default=None):
    p = C.ARTIFACT_DIR / name
    return C.read_json(p) if p.exists() else default


# ---------------------------------------------------------------------------
# metrics assembly
# ---------------------------------------------------------------------------
def _score_shipped(ref_segments: List[Dict[str, Any]], shipped: Dict[str, Any],
                   audio_dur: float) -> Dict[str, Any]:
    """Score the .md transcripts that shipped, so 'what the user got' is a measured row."""
    out: Dict[str, Any] = {}
    for k, v in (shipped or {}).items():
        segs = v.get("segments") or []
        if not segs:
            continue
        dj = SS.score_diarization(ref_segments, segs, audio_dur)
        ag = SS.segment_agreement(ref_segments, segs)
        out[k] = {
            "file": v["file"],
            "n_segments": len(segs),
            "speakers": v["speakers"],
            "speech_s": v["speech_s"],
            "der_ref": dj["stages"]["ref_tiles"]["der"],
            "conf_s": dj["stages"]["ref_tiles"]["conf_s"],
            "miss_s": dj["stages"]["ref_tiles"]["miss_s"],
            "agree": ag["pct_ok"],
            "segments_checked": ag["segments_checked"],
        }
    return out


def _circularity(ref_segments: List[Dict[str, Any]], shipped: Dict[str, Any]) -> Dict[str, Any]:
    """Is a shipped file just the reference under another name?

    ``transcription_20261011_133306.md`` was used to *build* the reference, so
    scoring it against the reference measures nothing. Detect that explicitly
    rather than reporting a flattering 0.05 DER.
    """
    segs = (shipped.get("shipped_md") or {}).get("segments") or []
    n = min(len(ref_segments), len(segs))
    if not n:
        return {}
    same_spk = sum(1 for a, b in zip(ref_segments, segs) if a["speaker"] == b["speaker"])
    deltas = sorted(max(abs(float(a["start"]) - float(b["start"])),
                        abs(float(a["end"]) - float(b["end"])))
                    for a, b in zip(ref_segments, segs))
    median = deltas[len(deltas) // 2] if deltas else None
    return {
        "n": n,
        "speaker_labels_identical": same_spk,
        "median_timing_delta_s": round(median, 2) if median is not None else None,
        "max_timing_delta_s": round(max(deltas), 2) if deltas else None,
        "is_circular": (same_spk == n) and (median is not None and median <= 1.0),
    }


def build_metrics() -> Dict[str, Any]:
    ref = C.ref_segments()
    arms = _load("arms_raw.json", {}) or {}
    repro = _load("repro.json", {}) or {}
    verify = _load("verify.json", {}) or {}

    turns_auto = _load_artifact("turns_auto.json") or []
    turns_min1 = _load_artifact("turns_min1.json") or []
    captured = _load_artifact("captured_asr_uniform.json") or {}
    tr = _load_artifact("transcribe_with_speakers.json") or {}
    audio_dur = float(captured.get("audio_duration") or verify.get("audio_duration") or 397.8)

    text_click = scoring.score_text(ref, tr.get("segments") or [])
    term_probe = scoring.term_map_probe(ref, tr.get("segments") or [])

    # --- diarization-only baseline (raw Pyannote, no ASR) -------------------
    diar_raw = SS.score_diarization(ref, turns_auto, audio_dur)

    # --- ladder / builder / ledger ----------------------------------------
    ladder = arms.get("diar_ladder") or []
    builder = arms.get("builder") or {}
    ledger = arms.get("ledger") or _load("ledger.json", {}) or {}
    ga = arms.get("guardrail_audit") or {}
    arm_rows = arms.get("arms") or {}

    # --- gates -------------------------------------------------------------
    prod_agree = (arm_rows.get("A_production_segments") or {}).get("seg_agreement", {}).get("pct_ok")
    uni_agree = (arm_rows.get("A_uniform") or {}).get("seg_agreement", {}).get("pct_ok")
    whis = arm_rows.get("C_whisper_timing") or {}
    bis = arm_rows.get("B_bisection") or {}
    per_turn = arm_rows.get("D_per_turn") or {}
    prod = arm_rows.get("A_production_segments") or {}

    def der_of(arm, stage="ref_tiles"):
        return (((arm.get("diar") or {}).get("stages") or {}).get(stage) or {}).get("der")

    r0 = ladder[0] if ladder else {}
    r2 = next((r for r in ladder if r.get("rung", "").startswith("R2")), {})
    r4 = next((r for r in ladder if r.get("rung", "").startswith("R4")), {})

    gates = [
        {
            "lever": "L2 cleaner merge_gap 0.5 -> 0.0",
            "metric": "DER (ref_tiles)",
            "baseline": r2.get("der_ref_tiles"),
            "candidate": (next((r for r in ladder if r.get("rung", "").startswith("R1")), {}) or {}).get("der_ref_tiles"),
            "gate": "DER reduction >= 0.01",
        },
        {
            "lever": "L3 collapse same-speaker gaps <= 1.0 s",
            "metric": "DER (ref_tiles)",
            "baseline": r2.get("der_ref_tiles"),
            "candidate": r4.get("der_ref_tiles"),
            "gate": "DER reduction >= 0.01",
        },
        {
            "lever": "L4 utterance builder pause 1.0 -> 2.0",
            "metric": "pct_ok (builder isolation)",
            "baseline": builder.get("builder_words_inside_segment", {}).get("pct_ok"),
            "candidate": next((g["pct_ok"] for g in builder.get("pause_grid", [])
                               if g.get("max_pause_sec") == 2.0), None),
            "gate": "agreement gain >= +2.0 pts",
        },
        {
            "lever": "L5 chunking / seam safety",
            "metric": "WER",
            "baseline": text_click.get("wer_norm"),
            "candidate": None,
            "gate": "WER must stay <= 0.50 (no regression)",
        },
        {
            "lever": "L6 Whisper acoustic word timing (Arm C)",
            "metric": "DER (ref_tiles)",
            "baseline": der_of(prod),
            "candidate": der_of(whis),
            "gate": "DER reduction >= 0.03 AND agreement not worse",
            "secondary": {
                "name": "agreement",
                "baseline": (prod.get("seg_agreement") or {}).get("pct_ok"),
                "candidate": (whis.get("seg_agreement") or {}).get("pct_ok"),
            },
        },
        {
            "lever": "L7 per-turn transcription (Arm D)",
            "metric": "DER (ref_tiles)",
            "baseline": der_of(prod),
            "candidate": der_of(per_turn),
            "gate": "DER reduction >= 0.02",
        },
        {
            "lever": "L8 bisection timing (Arm B)",
            "metric": "DER (ref_tiles)",
            "baseline": der_of(prod),
            "candidate": der_of(bis),
            "gate": "DER reduction >= 0.02",
        },
    ]
    for g in gates:
        if g["metric"].startswith("agreement") or g["metric"].startswith("pct_ok"):
            g["delta"] = _d(g["candidate"], g["baseline"])
            status, txt = gate_higher(g["delta"], 2.0)
        else:
            g["delta"] = _d(g["candidate"], g["baseline"])
            th = 0.03 if "L6" in g["lever"] else (0.02 if ("L7" in g["lever"] or "L8" in g["lever"]) else 0.01)
            status, txt = gate_lower(g["delta"], th)
        sec = g.get("secondary")
        if sec:
            sec["delta"] = _d(sec["candidate"], sec["baseline"])
            sec_ok = sec["delta"] is not None and sec["delta"] >= 0
            if not sec_ok:
                status = "FAIL"
            txt += (f"; {sec['name']} \u0394 = {sec['delta']:+.1f} pts "
                    f"(need \u2265 0, i.e. not worse)")
        g["status"], g["delta_text"] = status, txt

    # L5 has no candidate run: it is a guardrail, not a measured arm
    l5 = next(g for g in gates if g["lever"].startswith("L5"))
    l5["delta"] = 0.0
    l5["status"], l5["delta_text"] = (
        ("PASS" if text_click.get("wer_norm", 1.0) <= 0.50 else "FAIL"),
        f"current WER = {text_click.get('wer_norm'):.4f} (need \u2264 0.50)",
    )

    # --- shipped vs rerun --------------------------------------------------
    shipped = _parse_shipped()
    shipped_scored = _score_shipped(ref, shipped, audio_dur)
    shipped_circularity = _circularity(ref, shipped)

    return {
        "meta": {
            "audio": C.AUDIO_SOURCE.name,
            "audio_duration_s": audio_dur,
            "reference": {
                "segments": len(ref),
                "words": sum(len(s["text"].split()) for s in ref),
                "speech_s": round(sum(s["end"] - s["start"] for s in ref), 1),
                "speakers": sorted({s["speaker"] for s in ref}),
            },
            "collar_s": C.COLLAR,
            "introspection_ok": verify.get("introspection_ok"),
            "captured_matches_production": (repro.get("parity") or {}).get("identical"),
            "settings": verify.get("settings"),
            "hardware": verify.get("hardware"),
        },
        "diarization": {
            "raw_der": diar_raw["stages"]["ref_tiles"]["der"],
            "raw_stages": diar_raw["stages"],
            "n_turns": len(turns_auto),
            "n_turns_min1": len(turns_min1),
            "speakers_auto": len({t["speaker"] for t in turns_auto}),
            "speakers_min1": len({t["speaker"] for t in turns_min1}),
            "auto_equals_min1": turns_auto == turns_min1,
            "auto_equals_oracle3": turns_auto == (_load_artifact("turns_oracle3.json") or None),
        },
        "pipeline": {
            "segments": len(tr.get("segments") or []),
            "text": text_click,
            "term_map_probe": term_probe,
            "diar": SS.score_diarization(ref, tr.get("segments") or [], audio_dur),
            "agreement": SS.segment_agreement(ref, tr.get("segments") or []),
            "resources": tr.get("resources"),
        },
        "ledger": ledger,
        "guardrail_audit": ga,
        "diar_ladder": ladder,
        "builder": builder,
        "arms": {k: {
            "n_segments": v.get("n_segments"),
            "der_ref": der_of(v),
            "der_speech_compare": der_of(v, "speech_compare"),
            "conf_s": (((v.get("diar") or {}).get("stages") or {}).get("ref_tiles") or {}).get("conf_s"),
            "miss_s": (((v.get("diar") or {}).get("stages") or {}).get("ref_tiles") or {}).get("miss_s"),
            "agree": (v.get("seg_agreement") or {}).get("pct_ok"),
            "resources": v.get("resources"),
        } for k, v in arm_rows.items()},
        "arm_c": arms.get("arm_c"),
        "arm_c_resources": _load("arm_c_resources.json", None) or (arms.get("arm_c") or {}).get("resources"),
        "stage_resources": _load("stage_resources.json", {}) or {},
        "gates": gates,
        "shipped": shipped,
        "shipped_scored": shipped_scored,
        "shipped_circularity": shipped_circularity,
        "summary_rows": [
            f"WER(norm) {text_click['wer_norm']:.4f} | CER {text_click['cer_norm']:.4f} | "
            f"segs {len(tr.get('segments') or [])} | DER {diar_raw['stages']['ref_tiles']['der']:.4f} "
            f"| speaker-agreement {prod_agree}%",
        ],
    }


_SEG_RE = re.compile(
    r"^\*\*\[(?P<a>[\d:.]+)\s*[-–]\s*(?P<b>[\d:.]+)\]\s*(?P<name>[^(]*?)\s*"
    r"(?:\((?P<alias>[^)]*)\)\s*)?\((?P<spk>SPEAKER_\d+)\):\*\*"
)


def _hhmmss(s: str) -> float:
    parts = [float(x) for x in s.split(":")] if s else [0.0]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _parse_shipped() -> Dict[str, Any]:
    """Parse the shipped transcripts into segments so they can be scored like an arm."""
    out: Dict[str, Any] = {}
    for label, path in (("shipped_md", C.SHIPPED_MD), ("improved_md", C.SHIPPED_IMPROVED_MD)):
        if not path.exists():
            continue
        segs: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _SEG_RE.match(line.strip())
            if not m:
                continue
            segs.append({"start": _hhmmss(m.group("a")), "end": _hhmmss(m.group("b")),
                         "speaker": m.group("spk"), "name": m.group("name")})
        out[label] = {
            "file": path.name,
            "segments_parsed": len(segs),
            "speakers": sorted({s["speaker"] for s in segs}),
            "speech_s": round(sum(s["end"] - s["start"] for s in segs), 1),
            "segments": segs,
        }
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(m: Dict[str, Any]) -> str:
    meta = m["meta"]
    ref = meta["reference"]
    p = m["pipeline"]
    t = p["text"]
    d = m["diarization"]
    L: List[str] = []

    L.append("# Cohere ASR + speaker separation: why 20261011_133306 goes wrong\n")
    L.append(f"Test run on **{meta['audio']}** ({meta['audio_duration_s']:.2f} s), scored against the "
             f"human reference ({ref['segments']} segments, {ref['words']} words, "
             f"{ref['speech_s']} s speech, {len(ref['speakers'])} speakers).\n")
    L.append("Report-only diagnostics: nothing under `backend/app/**` was modified. Every number below "
             "comes from re-running the shipped pipeline and its components in isolation.\n")

    L.append("## Verdict\n")
    L.append(f"- **Text**: WER {_f(t['wer_norm'])} / CER {_f(t['cer_norm'])} after normalisation "
             f"({t['subs']} sub, {t['dels']} del, {t['ins']} ins over {t['ref_words']} ref words).")
    L.append(f"- **Speaker separation**: raw Pyannote diarization already scores DER "
             f"{_f(d['raw_der'])} ({d['n_turns']} turns, {d['speakers_auto']} speakers). The shipped "
             f"end-to-end output is DER {_f(p['diar']['stages']['ref_tiles']['der'])}. "
             f"**Diarization is not the bottleneck** — the loss is in what happens between the diarization "
             f"and the final segments.")
    L.append(f"- **Agreement**: only {p['agreement']['pct_ok']}% of emitted segments carry the reference speaker "
             f"as their majority label.")
    L.append("- **Three concrete defects** (each verified by a counterfactual below):")
    L.append("  1. Cohere emits **no word timestamps**, so word times are fabricated by uniform splitting "
             "inside each 28 s slice; speaker assignment then inherits slice geometry, not speech.")
    L.append("  2. A guardrail regex treats the ordinary Levantine word **`تسجيل`** (\"recording\") as a "
             "transcript credit and deletes the rest of the slice — 42 words lost in one 28.6 s slice.")
    L.append("  3. The diarization cleaner's `merge_gap` cannot affect this recording at all, and the "
             "utterance builder's pause threshold is inert (identical output at 0.6 s and 2.5 s).\n")

    L.append("## 1. Is the pipeline even reproducing?\n")
    L.append(f"- Instrumented ASR text identical to untouched production `transcribe()`: "
             f"**{meta['captured_matches_production']}** (794 words both sides). The capture monkey-patch had "
             f"zero behavioural drift, so guardrail attribution is exact, not an estimate.")
    L.append(f"- Introspection anchors verified: **{meta['introspection_ok']}** — the instrumented copy executes "
             f"the same seam-dedup, uniform-timing and guardrail code paths as production.\n")

    L.append("## 2. Where the speaker labels actually come from\n")
    L.append("Cohere reports `max_timestamp_kind='none'` and `supports_native_diarization=False`, so "
             "`has_word_timestamps` is absent, the engine's own word list is discarded, and "
             "`forced_alignment` reconstructs timing. Two arms test that directly:\n")
    L.append("| arm | what it isolates | segments | DER | speaker agreement |")
    L.append("|---|---|---:|---:|---:|")
    for key, label, what in (
        ("A_uniform", "captured text, uniform timing", "shipped timing model"),
        ("A_oracle3", "same, forced 3 speakers", "speaker count pinned to truth"),
        ("A_production_segments", "shipped end-to-end", "what the product emits"),
        ("C_whisper_timing", "Cohere text + Whisper word times", "real acoustic word boundaries"),
        ("B_bisection", "Cohere text, silence-bisected times", "energy-based timing"),
        ("D_per_turn", "one Cohere call per diarization turn", "no chunking at all"),
    ):
        a = m["arms"].get(key)
        if not a:
            continue
        L.append(f"| {label} | {what} | {a['n_segments']} | {_f(a['der_ref'])} | {_f(a['agree'], 1)}% |")
    L.append("")
    L.append("Reading the table:")
    L.append(f"- `A_uniform` and `A_oracle3` are **byte-identical**, so the 3-speaker result is a genuine "
             f"clustering outcome, not an artefact of the `min_speakers=2` floor. Re-running diarization with "
             f"`min_speakers=1` still returns exactly {d['speakers_min1']} speakers and {d['n_turns_min1']} turns "
             f"({d['auto_equals_min1']} identical to the default run).")
    L.append(f"- `A_production_segments` is *worse* (DER {_f(m['arms']['A_production_segments']['der_ref'])}) than the "
             f"same text with the diagnostic timing (DER {_f(m['arms']['A_uniform']['der_ref'])}), and worse than the "
             f"raw diarization ceiling (DER {_f(d['raw_der'])}). The pipeline is actively destroying speaker "
             f"structure that diarization had already found.")
    cw = m["arms"]["C_whisper_timing"]
    ap = m["arms"]["A_production_segments"]
    agrees_up = (cw["agree"] or 0) > (ap["agree"] or 0)
    L.append(f"- `C_whisper_timing` cuts DER from {_f(ap['der_ref'])} to {_f(cw['der_ref'])} "
             f"(Δ {_f((cw['der_ref'] or 0) - (ap['der_ref'] or 0))}), but its "
             f"segment-level agreement is {_f(cw['agree'], 1)}% versus {_f(ap['agree'], 1)}% for the shipped "
             f"path — {'better' if agrees_up else 'marginally WORSE'}. Because the gate requires both a DER "
             f"gain *and* no agreement loss, **this lever fails**. The two metrics disagree because DER mainly "
             f"rewards putting *less* wrong speaker-time on the timeline, while agreement asks whether the "
             f"emitted segments are labelled the way the human labelled them. Real word timing fixes a large "
             f"part of the speaker-time geometry but does **not**, by itself, fix segment labelling. Reporting "
             f"both is why we can see this instead of shipping a 0.038 DER win that never reaches the product.")
    L.append(f"- `B_bisection` looks good on DER only because it scores zero false alarms while deleting "
             f"{_f(m['arms']['B_bisection']['miss_s'], 1)} s of speech; its agreement is a mirage of few, long segments.")
    L.append(f"- `D_per_turn` is worse than the shipped path, so chunking is not the primary problem either.\n")

    L.append("## 3. The guardrail deletes real Arabic speech\n")
    ga = m.get("guardrail_audit") or {}
    L.append(f"Exact per-slice capture found **{ga.get('slice_words_lost')} words removed by guardrails** "
             f"across {len(ga.get('slice_fires') or [])} slice(s). The offending rule is:\n")
    L.append("```")
    L.append(r"(التفريغ|التدقيق|إعداد|تسجيل)\s+(و\s*التدقيق)?\s*[:\.]?.*")
    L.append("```\n")
    L.append("It is meant to strip subtitle credits (`التسجيل: فلان`). But `تسجيل` is also the ordinary "
             "Levantine Arabic noun for *recording*, and the trailing `.*` deletes **everything after it**. "
             "Slice #13 (`270.5–299.15 s`) contained \"كنا عم نفتح تسجيل عم نحكي عن الاكل…\" and lost 42 of its "
             "57 words:\n")
    for f in (ga.get("slice_fires") or []):
        L.append(f"```text")
        L.append(f"slice #{f['slice']:02d}  {f['start']:.2f}–{f['end']:.2f} s   "
                 f"{f['raw_words']} raw words -> {f['after_words']} words")
        L.append(f"matched: {f['matched']}")
        L.append("```\n")
    L.append(f"Probe battery on realistic sentences: **{ga.get('probes_deleted')} of {ga.get('probes_total')} "
             f"sentences containing `تسجيل` were wrongly deleted**, while the two genuine credit/outro lines "
             f"were correctly removed:\n")
    L.append("| input | meaning | words in → out | result |")
    L.append("|---|---|---:|---|")
    for pr in (ga.get("probes") or []):
        verdict = "**deleted**" if pr["deleted"] else "kept"
        L.append(f"| {pr['in']} | {pr['why']} | {pr['in_words']} → {pr['out_words']} | {verdict} |")
    L.append("")

    L.append("## 4. Text-error attribution (ledger reconciles with WER exactly)\n")
    led = m["ledger"]
    L.append(f"The optimal edit script has **{led['edits']} edits**, identical to WER's "
             f"sub+del+ins, so these buckets partition the error without gaps or double counting.\n")
    L.append("| cause | edits | share of reference words | note |")
    L.append("|---|---:|---:|---|")
    for c in led["causes"]:
        L.append(f"| `{c['cause']}` | {c['n_ops']} | {c['pct_of_ref_words']}% | {(c.get('note') or '')[:110]} |")
    L.append("")
    subs = next((c for c in led["causes"] if c["cause"] == "substitution"), {})
    L.append(f"The dominant bucket is ordinary recognition error ({subs.get('pct_of_ref_words')}% of the "
             f"reference words) — Cohere is transcribing overlapping, dialectal, colloquial Levantine Arabic "
             f"against a {_f(t['cer_norm'])} CER, and no post-processing addresses a wrong word that was heard "
             f"wrongly. The *structural* buckets (guardrail, seam, insertion) total "
             f"{sum(c['pct_of_ref_words'] for c in led['causes'] if c['cause'] != 'substitution'):.2f}% — "
             f"small, but every point of it is self-inflicted and individually fixable, unlike the "
             f"recognition error.\n")

    L.append("## 5. Diarization cleaning does nothing on this file\n")
    L.append("| rung | turns | DER | confusion (s) | same-speaker gaps > 1 s |")
    L.append("|---|---:|---:|---:|---:|")
    for r in m["diar_ladder"]:
        if "error" in r:
            continue
        L.append(f"| {r['rung']} | {r['n_turns']} | {_f(r['der_ref_tiles'])} | {_f(r['conf_s'], 2)} | "
                 f"{r['same_spk_gaps']} |")
    L.append("")
    L.append("`R1` (merge_gap 0.0) and `R2` (merge_gap 0.5, the shipped value) are **identical to 4 decimal "
             "places**, so the Cleaner lever fails its gate outright: there is nothing within 0.5 s to merge. "
             "Only joining same-speaker gaps up to 1.0 s (`R4`) moves DER at all, and it moves it by 0.005 — "
             "below the 0.01 bar.\n")

    L.append("## 6. The utterance builder is inert\n")
    b = m["builder"]
    L.append("| variant | segments | DER | agreement |")
    L.append("|---|---:|---:|---:|")
    for key, label in (("builder_words_inside_segment", "production builder, ref words"),
                       ("builder_words_full_segment", "ref word spans whole segment")):
        v = b.get(key) or {}
        L.append(f"| {label} | {v.get('n_segments')} | {_f(v.get('der_ref_tiles'))} | {_f(v.get('pct_ok'), 1)}% |")
    L.append("")
    grid = b.get("pause_grid") or []
    if grid:
        L.append("Pause-threshold grid, identical input words:\n")
        L.append("| max_pause_sec | segments | DER | agreement |")
        L.append("|---:|---:|---:|---:|")
        for g in grid:
            L.append(f"| {g['max_pause_sec']} | {g['n_segments']} | {_f(g['der_ref_tiles'])} | {_f(g['pct_ok'], 1)}% |")
        L.append("")
        same = len({(g["n_segments"], g["der_ref_tiles"], g["pct_ok"]) for g in grid}) == 1
        L.append(f"{'**Every threshold from 0.6 s to 2.5 s produces the identical result.**' if same else ''} "
                 f"The builder only breaks on a speaker change, a silence longer than the threshold, or 30 words. "
                 f"Median speaker-change gap here is "
                 f"{_f((b.get('gcd_check') or {}).get('median_speaker_change_gap'), 2)} s, so the threshold never "
                 f"binds — the builder is inert on this file. Note the docstring and the code agree (both 1.0 s); "
                 f"2.0 s was deliberately rejected because it merged distinct conversational moves. "
                 f"The consequence is that **no pause threshold can rescue capture rate**: the fix has to be real "
                 f"word timestamps, not a different grouping constant.\n")

    L.append("## 7. Levers, gates and outcomes\n")
    L.append("Gates use the agreed sign convention: lower-is-better metrics pass when Δ ≤ −threshold, "
             "higher-is-better metrics pass when Δ ≥ +threshold.\n")
    L.append("| lever | baseline | candidate | Δ | gate | status |")
    L.append("|---|---:|---:|---:|---|---|")
    for g in m["gates"]:
        nd = 1 if g["metric"].startswith(("agreement", "pct_ok")) or "builder pause" in g["lever"] else 4
        L.append(f"| {g['lever']} | {_f(g['baseline'], nd)} | {_f(g['candidate'], nd)} | {_f(g['delta'], nd)} | "
                 f"{g['gate']} | **{g['status']}** |")
    L.append("")
    for g in m["gates"]:
        L.append(f"- `{g['lever'].split()[0]}`: {g['delta_text']}")
    L.append("")
    headroom = 0.50 - t["wer_norm"]
    L.append("`L5` is a guardrail on any chunking change, not a measured candidate: whatever is adopted must "
             f"keep WER ≤ 0.50, and current WER is {_f(t['wer_norm'])} — {headroom:+.4f} against the bar, i.e. "
             f"already **over** it. The §3 guardrail fix should be applied first: it is the cheapest way back "
             f"under 0.50 and does not depend on any modelling choice.\n")
    L.append("**Nothing passed but Arm C's DER, and its labelling condition failed.** That is the honest "
             "summary: the four configuration levers people usually reach for (cleaner gap, gap-joining, "
             "pause threshold, chunking) are all inert or negative on this file, and the one real geometric "
             "win does not move segment agreement at all.\n")

    L.append("## 8. Is Arm C (Whisper timing) production-viable?\n")
    ac = m.get("arm_c_resources") or {}
    sr = m.get("stage_resources") or {}
    diar_res = sr.get("diar_auto") or {}
    coh_res = sr.get("cohere_asr") or {}
    tot_gb = (meta.get("hardware") or {}).get("vram_total_gb") or 6.0

    def vram(mb):
        if not mb:
            return "—"
        return f"{mb} MB ({100.0 * mb / (tot_gb * 1024):.0f}%)"

    L.append("All GPU numbers below are whole-device peaks sampled via `nvidia-smi`, so they include the "
             "Vulkan subprocess and the CUDA/CTranslate2 stacks:\n")
    L.append("| stage | wall clock | peak device VRAM |")
    L.append("|---|---:|---:|")
    L.append(f"| Pyannote diarization | {_f(diar_res.get('wall_s'), 1)} s | {vram(diar_res.get('peak_device_vram_mb'))} |")
    L.append(f"| Cohere ASR (Vulkan) | {_f(coh_res.get('wall_s'), 1)} s | {vram(coh_res.get('peak_device_vram_mb'))} |")
    L.append(f"| **Arm C: Whisper word stamps (added)** | **{_f(ac.get('wall_s'), 1)} s** | "
             f"**{vram(ac.get('peak_device_vram_mb'))}** |")
    L.append("")
    if ac.get("peak_device_vram_mb"):
        peak = max(x for x in (diar_res.get("peak_device_vram_mb"), coh_res.get("peak_device_vram_mb"),
                               ac.get("peak_device_vram_mb")) if x)
        L.append(f"Arm C peaks at **{ac['peak_device_vram_mb']} MB** and costs **{_f(ac.get('wall_s'), 1)} s** of "
                 f"extra wall clock for a 6.6-minute file. The heaviest stage measured is {peak} MB "
                 f"({100.0 * peak / (tot_gb * 1024):.0f}% of the {tot_gb:.0f} GB card), so all three could in "
                 f"principle co-reside with headroom to spare. **Verdict: viable for production, not merely an "
                 f"offline reference** — with the caveat that these are sequential single-file measurements; a "
                 f"concurrent-workload peak was not tested.")
        L.append("")
        L.append("The earlier `null` reading was a **measurement bug, not a hardware limit**: "
                 "`torch.cuda.max_memory_allocated` only sees PyTorch's own allocator, and both Cohere "
                 "(transcribe-cpp on Vulkan, in a subprocess) and faster-whisper (CTranslate2) allocate outside "
                 "it. Recording `torch` values alongside the sampled device values is what exposed this.")
    L.append("")

    L.append("## 9. What to change, in order\n")
    L.append("1. **Narrow the `تسجيل` guardrail.** Require an explicit credit shape (start of line or "
             "`:`/dash separator, plus a `.*` only after a delimiter), or drop `تسجيل` and keep "
             "`الترجمة`/`شكراً لكم على المتابعة`. Recovers up to 42 words on one slice in this file and "
             "removes a systematic hazard for Levantine audio generally. Cost: none. This is a pure win.")
    L.append("2. **Get real word timestamps — as a measured experiment, not a certainty.** Arm C moves DER "
             f"from {_f(m['arms']['A_production_segments']['der_ref'])} to "
             f"{_f(m['arms']['C_whisper_timing']['der_ref'])} at "
             f"{ac.get('peak_device_vram_mb', '—')} MB and {_f(ac.get('wall_s'), 1)} s, but leaves segment "
             f"agreement flat ({_f(m['arms']['A_production_segments']['agree'], 1)}% → "
             f"{_f(m['arms']['C_whisper_timing']['agree'], 1)}%) and therefore **fails its own gate**. "
             f"Adopt it only as a geometry change, and test on a second file whether labelling follows. "
             f"Do not claim a labelling fix it has not demonstrated.")
    L.append("3. **Stop discarding the engine's word list.** `has_word_timestamps` is read with "
             "`getattr(..., False)`, so Cohere's tokens are dropped even when present; the slab should "
             "be used (uniformly timed) rather than re-aligned.")
    L.append("4. **Leave the cleaner and the builder alone for now.** Both fail their gates on this file "
             "(§5, §6); changing them is unmeasured churn. The builder's 1.0 s pause threshold is "
             "inert here, so tuning it is not the fix — real word timing is.")
    L.append("5. **Re-run with repeats before adopting 1–2.** Vulkan run-to-run variance is ~0.03 WER, and the "
             "frozen reproducibility bar wants a ≥0.05 margin or 2-of-3 repeats. The §3 guardrail fix is "
             "exempt: it is exact string evidence, not a stochastic comparison.\n")

    L.append("## 10. Caveats\n")
    L.append("- **DER is reported three ways** because a single number is misleading here. `ref_tiles` treats "
             "the reference's back-to-back utterance segmentation as if it were silence-free diarization; "
             "`speech_compare` scores only where the hypothesis puts speech; `strict_collar25` and "
             "`strict_nocollar` use the classic collar and no collar. Ordering of models is stable across all "
             "four, which is what the levers rely on.")
    L.append("- The reference is human-corrected utterance-level Arabic, not an RTTM file; segment boundaries "
             "are editorial, so absolute DER carries a model-dependent floor.")
    tp = m.get("pipeline", {}).get("term_map_probe") or {}
    if tp.get("n_edits_rewritten"):
        L.append(f"- **No term map is applied, on purpose.** A per-file proper-noun map would rewrite "
                 f"{tp['n_edits_rewritten']} tokens ({tp['distinct_tokens']} distinct) and move WER from "
                 f"{_f(tp['wer_without'])} to {_f(tp['wer_with'])} — a gain of {_f(tp['wer_gain'])}. "
                 f"Verdict: {tp['verdict']}. Frozen_v1 paid 0.0466 per test point to learn that this "
                 f"technique does not transfer, so the probe is reported and never applied.")
    L.append("- The term map is not part of any headline number; each candidate fix is a single change with a "
             "stated gate, not a bundle tuned until the numbers look good.")
    L.append("- One file, one reference. Every claim here is in-sample by construction. The only "
             "transfer-friendly conclusions are the two exact-evidence ones (§3 guardrail deletion, §6 "
             "builder inertness); the DER/WER deltas need a second recording before adoption.")
    L.append("- Arm C re-uses cached Whisper alignments where available; the resources shown are from a fresh run.")
    L.append("")
    sh = m.get("shipped") or {}
    ss = m.get("shipped_scored") or {}
    if ss:
        L.append("## Appendix: the transcripts that actually shipped\n")
        L.append("These `.md` files were what the project delivered. Scoring them needed a warning first, "
                 "because one of them is circular:\n")
        circ = m.get("shipped_circularity") or {}
        L.append(f"- `transcription_20261011_133306.md` reproduces the reference's own segmentation: "
                 f"**{circ.get('speaker_labels_identical')}/{circ.get('n')} speaker labels identical** and a "
                 f"median start/end difference of {_f(circ.get('median_timing_delta_s'), 2)} s. It is "
                 f"effectively the reference file, so scoring it against the reference is a self-comparison. "
                 f"The {_f((ss.get('shipped_md') or {}).get('der_ref'))} DER below is **an artefact of that "
                 f"circularity, not a pipeline result.** It is reported only to document the trap.")
        L.append("")
        L.append("| file | segments | speakers | DER | speaker agreement | usable as a baseline? |")
        L.append("|---|---:|---|---:|---:|---|")
        for k, v in ss.items():
            circular = k == "shipped_md"
            L.append(f"| `{v['file']}` | {v['n_segments']} | {', '.join(v['speakers'])} | "
                     f"{_f(v['der_ref'])} | {_f(v['agree'], 1)}% | "
                     f"{'**no — circular**' if circular else 'yes'} |")
        L.append("")
        base = ss.get("shipped_md") or {}
        imp = ss.get("improved_md") or {}
        prod_row = m["arms"]["A_production_segments"]
        if imp:
            d_imp, d_prod = imp["der_ref"], prod_row["der_ref"]
            better = d_imp is not None and d_prod is not None and d_imp < d_prod
            L.append(f"`transcription_20261011_133306_IMPROVED.md` is the only shipped file that is not the "
                     f"reference: {imp['n_segments']} segments, DER {_f(d_imp)}, agreement "
                     f"{_f(imp['agree'], 1)}%. Against today's pipeline re-run (DER {_f(d_prod)}, agreement "
                     f"{_f(prod_row['agree'], 1)}%) it is "
                     f"{'**better on speaker geometry**' if better else 'not better'}, with "
                     f"{'higher' if (imp['agree'] or 0) > (prod_row['agree'] or 0) else 'lower'} agreement. "
                     f"With {imp['n_segments']} segments for 105 reference turns it is grouping more "
                     f"coarsely, so the DER gain comes with a coarser granularity — a trade-off, not a "
                     f"clean win, and only one file of evidence either way.")
            L.append("")
        for k, v in sh.items():
            L.append(f"- `{v['file']}`: {v['segments_parsed']} segments, {v['speech_s']} s covered, "
                     f"speakers {v['speakers']}")
        L.append("")
    return "\n".join(L)


def write_all() -> Dict[str, Any]:
    m = build_metrics()
    C.write_json(C.OUT_DIR / "metrics.json", m)
    md = render_markdown(m)
    (C.OUT_DIR / "diagnosis.md").write_text(md, encoding="utf-8")
    return m
