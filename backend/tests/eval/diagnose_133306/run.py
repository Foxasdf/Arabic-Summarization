"""CLI orchestrator for the 133306 Cohere + Pyannote diagnostics.

Stages
------
``verify``  introspection anchors + reference sanity (seconds)
``repro``   decode, diarize (auto + oracle), one captured Cohere pass, production text
``arms``    diar ladder, builder isolation, timing ablations, per-turn, Whisper arm
``score``   text + speaker metrics for every arm onto the reference
``report``  diagnosis.md + metrics.json from cached artifacts only
``all``     everything (default)

Usage (from repo root, inside backend/.venv):
    python backend/tests/eval/diagnose_133306/run.py --stage verify
    python backend/tests/eval/diagnose_133306/run.py --stage repro
    python backend/tests/eval/diagnose_133306/run.py --stage arms
    python backend/tests/eval/diagnose_133306/run.py --stage all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Import the package under ONE name only. Previously this module also put
# backend/tests/eval on sys.path and did `from diagnose_133306 import config`,
# which created a SECOND copy of the package: set_active() rebound the globals in
# one copy while the pipeline modules read the other, so a 300 s recording was
# silently measured with the 397.8 s audio and its cached artifacts.
#
# Anchor every import to the package's own directory so `python -m
# backend.tests.eval.diagnose_133306.run` and a direct script launch resolve to
# the same module objects.
_HERE = Path(__file__).resolve()
_PKG_PARENT = _HERE.parents[1]  # backend/tests/eval
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from diagnose_133306 import config as C  # noqa: E402

sys.path.insert(0, str(C.BACKEND))

C.bootstrap_sys_path()
C.add_torch_dll_dirs()


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# stage: verify
# ---------------------------------------------------------------------------
def stage_verify() -> dict:
    from . import patches

    C.ensure_dirs()
    anchors = patches.verify_introspection()
    ok = all(anchors.values())
    _log(f"introspection anchors: {anchors} -> {'OK' if ok else 'DRIFT'}")

    ref = C.ref_segments()
    audio_dur = None
    try:
        from app.services.audio_processor import get_audio_duration

        audio_dur = get_audio_duration(C.AUDIO_SOURCE)
    except Exception as e:
        _log(f"audio duration probe failed: {e}")

    ref_words = sum(len(s["text"].split()) for s in ref)
    ref_speech = sum(s["end"] - s["start"] for s in ref)
    overlaps = []
    for a, b in zip(sorted(ref, key=lambda x: x["start"]), sorted(ref, key=lambda x: x["start"])[1:]):
        ov = a["end"] - b["start"]
        if ov > 0.01:
            overlaps.append((round(b["start"], 2), round(ov, 2)))

    out = {
        "introspection": anchors,
        "introspection_ok": ok,
        "reference": {
            "n_segments": len(ref),
            "n_words": ref_words,
            "speech_sec": round(ref_speech, 2),
            "coverage_pct": round(100 * ref_speech / audio_dur, 1) if audio_dur else None,
            "speakers": sorted({s["speaker"] for s in ref}),
            "overlaps": overlaps,
            "gaps": sum(
                1 for a, b in zip(sorted(ref, key=lambda x: x["start"]),
                                  sorted(ref, key=lambda x: x["start"])[1:])
                if b["start"] - a["end"] > 0.05
            ),
        },
        "audio_duration": audio_dur,
        "settings": C.settings_snapshot(),
        "hardware": C.hardware_snapshot(),
    }
    C.write_json(C.OUT_DIR / "verify.json", out)
    _log(f"reference: {len(ref)} segs, {ref_words} words, {ref_speech:.1f}s speech")
    return out


# ---------------------------------------------------------------------------
# stage: repro
# ---------------------------------------------------------------------------
def stage_repro() -> dict:
    from . import pipeline_runner as PR

    C.ensure_dirs()
    meta = PR.ensure_canonical_audio()
    _log(f"canonical audio ready ({meta['wall_s']}s)")
    asr_wav = Path(meta["asr_wav"])
    spk_wav = Path(meta["speaker_wav"])

    runs = PR.diarize_arms(asr_wav, spk_wav)
    for label, r in runs.items():
        _log(f"diarization[{label}]: {len(r['turns'])} turns, "
             f"{len({t['speaker'] for t in r['turns']})} speakers ({r['source']})")

    # captured continuous pass (uniform timing) -- the ledger source
    cap_path = C.ARTIFACT_DIR / "captured_asr_uniform.json"
    if cap_path.exists():
        captured = C.read_json(cap_path)
        _log(f"captured ASR reused from cache ({len(captured['words'])} words)")
    else:
        _log("running captured Cohere pass (this takes a few minutes)...")
        t0 = time.time()
        captured = PR.run_captured_asr(asr_wav, timing="uniform")
        C.write_json(cap_path, captured)
        _log(f"captured ASR done in {time.time() - t0:.0f}s: "
             f"{len(captured['words'])} words, {len(captured['slices'])} slices")

    # production text path for parity check
    prod_path = C.ARTIFACT_DIR / "production_text.json"
    if prod_path.exists():
        prod = C.read_json(prod_path)
    else:
        _log("running untouched production transcribe() for parity check...")
        prod = PR.run_production_asr(asr_wav)
        C.write_json(prod_path, prod)

    from .scoring import subsequence_wer

    a = captured["text"].split()
    b = (prod.get("text") or "").split()
    parity = {
        "captured_words": len(a),
        "production_words": len(b),
        "identical": a == b,
        "wer_self": subsequence_wer(captured["text"], prod.get("text") or "")["wer"],
    }
    _log(f"parity: captured={len(a)} words, production={len(b)} words, identical={parity['identical']}")

    tr_path = C.ARTIFACT_DIR / "transcribe_with_speakers.json"
    if tr_path.exists():
        tr = C.read_json(tr_path)
    else:
        _log("running production transcribe_with_speakers (auto diarization)...")
        tr = PR.run_segments_arm(asr_wav, runs["auto"]["turns"], mode="continuous")
        C.write_json(tr_path, tr)
    _log(f"production segments: {len(tr['segments'])}")

    out = {
        "canonical": {k: v for k, v in meta.items() if k != "manifest"},
        "diarization": {k: {"n_turns": len(v["turns"]),
                            "n_speakers": len({t["speaker"] for t in v["turns"]}),
                            "wall_s": v.get("wall_s"), "resources": v.get("resources"),
                            "source": v.get("source")} for k, v in runs.items()},
        "captured": {
            "n_words": len(captured["words"]),
            "n_slices": len(captured["slices"]),
            "resources": captured.get("resources"),
            "guardrail_slices": sum(1 for s in captured["slices"] if s.get("guardrail_changed")),
            "vad_rejected": sum(1 for s in captured["slices"] if s.get("vad_passed") is False),
        },
        "parity": parity,
        "production_segments": len(tr["segments"]),
    }
    C.write_json(C.OUT_DIR / "repro.json", out)
    return out


# ---------------------------------------------------------------------------
# stage: arms
# ---------------------------------------------------------------------------
def stage_arms() -> dict:
    from . import ablations as AB
    from . import pipeline_runner as PR
    from . import text_ledger

    C.ensure_dirs()
    ref = C.ref_segments()
    n_ref_spk = len({s["speaker"] for s in ref})
    turns_auto = C.read_json(C.ARTIFACT_DIR / "turns_auto.json")
    turns_oracle = C.read_json(C.ARTIFACT_DIR / f"turns_oracle{n_ref_spk}.json")
    captured = C.read_json(C.ARTIFACT_DIR / "captured_asr_uniform.json")
    asr_wav = C.ARTIFACT_DIR / "asr.16k.wav"
    audio_dur = float(captured.get("audio_duration") or C.ACTIVE.duration())

    result: dict = {}

    # 1. diarization ladder -- no ASR
    _log("diar ladder...")
    result["diar_ladder"] = AB.diar_ladder(ref, turns_auto, audio_dur)

    # 1b. send-scikit; also oracle ladder
    result["diar_ladder_oracle"] = AB.diar_ladder(ref, turns_oracle, audio_dur)

    # 2. builder isolation (item 3)
    _log("builder isolation...")
    result["builder"] = AB.builder_isolation(ref, turns_auto, audio_dur)

    # 3. timing arms -- uniform (captured) and bisection (needs a Cohere pass)
    from app.services.transcription import transcribe_with_speakers

    def _from_words(words, turns, label):
        from app.services.alignment import align_timestamps

        segs = align_timestamps(words, turns)
        return AB.score_segments_arm(label, ref, segs, audio_dur)

    arms = {}
    words_uniform = captured["words"]
    arms["A_uniform"] = _from_words(words_uniform, turns_auto, "A_uniform")
    arms["A_oracle"] = _from_words(words_uniform, turns_oracle, "A_oracle")

    bis_path = C.ARTIFACT_DIR / "captured_asr_bisection.json"
    if bis_path.exists():
        cap_bis = C.read_json(bis_path)
    else:
        _log("running bisection-timing Cohere pass...")
        cap_bis = PR.run_captured_asr(asr_wav, timing="bisection")
        C.write_json(bis_path, cap_bis)
    arms["B_bisection"] = _from_words(cap_bis["words"], turns_auto, "B_bisection")
    arms["B_bisection"]["resources"] = cap_bis.get("resources")

    # 4. production end-to-end (transcribe_with_speakers) -- the shipped path
    tr = C.read_json(C.ARTIFACT_DIR / "transcribe_with_speakers.json")
    arms["A_production_segments"] = AB.score_segments_arm(
        "A_production_segments", ref, tr["segments"], audio_dur,
        {"resources": tr.get("resources")})

    # 5. Whisper timing arm (C)
    whisper_meta = {"status": "skipped"}
    try:
        _log("Arm C: Whisper word stamps for Cohere text...")
        wc = AB.whisper_timing_words(asr_wav, captured["text"], cache_tag="C")
        whisper_meta = {"status": "ok", "source": wc["source"], "resources": wc["resources"],
                        "n_words": len(wc["words"])}
        arms["C_whisper_timing"] = _from_words(wc["words"], turns_auto, "C_whisper_timing")
        arms["C_whisper_timing"]["resources"] = wc["resources"]
        arms["C_whisper_oracle"] = _from_words(wc["words"], turns_oracle, "C_whisper_oracle")
    except Exception as e:
        _log(f"Arm C failed: {e}")
        whisper_meta = {"status": "error", "error": repr(e)}
    result["arm_c"] = whisper_meta

    # 6. D: per-turn arm
    d_path = C.ARTIFACT_DIR / "segments_per_turn.json"
    if d_path.exists():
        dres = C.read_json(d_path)
    else:
        _log("Arm D: per-turn transcription...")
        dres = PR.run_segments_arm(asr_wav, turns_auto, mode="per_turn")
        C.write_json(d_path, dres)
    arms["D_per_turn"] = AB.score_segments_arm(
        "D_per_turn", ref, dres["segments"], audio_dur, {"resources": dres.get("resources")})

    result["arms"] = arms

    # 7. text ledger on the production/captured text
    _log("text ledger...")
    hyp_tokens = captured["text"].split()
    result["ledger"] = text_ledger.build_ledger(ref, hyp_tokens, captured, audio_wav=C.ARTIFACT_DIR / "speaker.16k.wav",
                                                total=audio_dur)
    C.write_json(C.OUT_DIR / "ledger.json", result["ledger"])

    # 8. guardrail audit (which rules fired, what they cost, probe battery)
    from . import guardrail_audit

    _log("guardrail audit...")
    result["guardrail_audit"] = guardrail_audit.audit(captured, captured["text"])

    C.write_json(C.OUT_DIR / "arms_raw.json", result)
    return {k: (v if not isinstance(v, list) else f"{len(v)} rows") for k, v in result.items()}


# ---------------------------------------------------------------------------
# stage: score + report
# ---------------------------------------------------------------------------
def stage_score_and_report() -> dict:
    from . import report

    C.ensure_dirs()
    metrics = report.build_metrics()
    C.write_json(C.OUT_DIR / "metrics.json", metrics)
    md = report.render_markdown(metrics)
    (C.OUT_DIR / "diagnosis.md").write_text(md, encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# stage: candidates (runtime patches, paired against the same captured audio)
# ---------------------------------------------------------------------------
def stage_candidates() -> dict:
    from . import candidate_eval as CE

    C.ensure_dirs()
    path = C.OUT_DIR / "candidates.json"
    out = C.read_json(path) if path.exists() else {}

    _log("candidate: hardened transcription-credit regex...")
    out["credit_regex"] = CE.evaluate_text_candidate(C.ACTIVE.key, "credit_regex")
    _log(f"  WER {out['credit_regex']['wer_before']} -> {out['credit_regex']['wer_after']} "
         f"({out['credit_regex']['delta']:+.4f}), words {out['credit_regex']['words_delta']:+d}, "
         f"slices changed {len(out['credit_regex']['slices_changed'])}")

    _log("candidate: engine word-list handoff (capability probe)...")
    out["engine_words"] = CE.evaluate_engine_words(C.ACTIVE.key)
    _log(f"  emits {out['engine_words']['n_words']} words "
         f"(text {out['engine_words']['n_text_words']}), valid={out['engine_words']['passes_validation']}")

    _log("candidate: fuzzy seam dedupe (fresh Cohere pass)...")
    try:
        out["fuzzy_seam"] = CE.evaluate_fuzzy_seam(C.ACTIVE.key)
        b, a = out["fuzzy_seam"]["baseline"], out["fuzzy_seam"]["candidate_metrics"]
        _log(f"  DER {b['der']} -> {a['der']}, words {out['fuzzy_seam']['words_before']} -> "
             f"{out['fuzzy_seam']['words_after']}")
    except Exception as e:
        out["fuzzy_seam"] = {"error": repr(e)}
        _log(f"  fuzzy seam failed: {e}")

    _log("candidate: VAD-aware word timing...")
    try:
        out["vad_word_timing"] = CE.evaluate_vad_word_timing(C.ACTIVE.key)
        b, a = out["vad_word_timing"]["baseline"], out["vad_word_timing"]["candidate_metrics"]
        _log(f"  DER {b['der']} -> {a['der']} ({a['der'] - b['der']:+.4f}), "
             f"masks {out['vad_word_timing']['slices_with_mask']}/{out['vad_word_timing']['n_slices']}")
    except Exception as e:
        out["vad_word_timing"] = {"error": repr(e)}
        _log(f"  vad timing failed: {e}")

    _log("candidate: Pyannote boundary snapping...")
    try:
        out["snap_words"] = CE.evaluate_snap_words(C.ACTIVE.key)
        b, a = out["snap_words"]["baseline"], out["snap_words"]["candidate_metrics"]
        _log(f"  DER {b['der']} -> {a['der']} ({a['der'] - b['der']:+.4f}), "
             f"moved {out['snap_words']['words_moved']}/{out['snap_words']['n_words']}")
    except Exception as e:
        out["snap_words"] = {"error": repr(e)}
        _log(f"  snap words failed: {e}")

    C.write_json(path, out)
    return out


# ---------------------------------------------------------------------------
# stage: e2e (candidates through the real pipeline, not cached slices)
# ---------------------------------------------------------------------------
def stage_e2e(force: bool = True) -> dict:
    from . import e2e_candidates as E

    C.ensure_dirs()
    _log("end-to-end candidate runs (baseline + patches, real pipeline)...")
    out = E.run_e2e(C.ACTIVE.key, ["credit_regex", "engine_words"], force=force)
    for label, v in out["runs"].items():
        _log(f"  {label:14s} segs={v['n_segments']:3d} DER={v['der']:.4f} "
             f"WER={v['wer_norm']:.4f} words={v['hyp_words']:3d} agree_w={v['agree_word']:.1f}")
    return out


# ---------------------------------------------------------------------------
# stage: suite (cross-file tables, macro average, joint gates)
# ---------------------------------------------------------------------------
def stage_suite() -> dict:
    from . import suite_report as SR

    keys = list(C.SUITE)
    records = []
    for k in keys:
        C.set_active(k)
        try:
            records.append(SR.file_metrics(k))
        except Exception as e:
            _log(f"suite: {k} skipped ({e})")

    cand: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        p = C.SUITE[k].out_dir / "candidates.json"
        if p.exists():
            cand[k] = C.read_json(p)

    gates = []
    notes = []

    # --- END-TO-END gates (authoritative: real pipeline, not cached text) ---
    e2e: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        p = C.SUITE[k].out_dir / "e2e.json"
        if p.exists():
            e2e[k] = C.read_json(p).get("runs") or {}

    def _e2e_gate(label: str, cnd: str, metric: str, higher: bool,
                  min_gain: Optional[float] = None):
        b = {k: (e2e[k].get("baseline") or {}).get(metric) for k in keys if k in e2e}
        a = {k: (e2e[k].get(cnd) or {}).get(metric) for k in keys if k in e2e}
        if not b or not a or any(v is None for v in b.values()) or any(v is None for v in a.values()):
            return None
        return SR.two_file_gate(label, b, a, metric, higher_is_better=higher, min_gain=min_gain)

    for cnd, lbl in (("credit_regex", "credit regex — end-to-end"),
                     ("engine_words", "engine word list — end-to-end")):
        if all(cnd in (e2e.get(k) or {}) for k in keys):
            g1 = _e2e_gate(f"{lbl}: WER", cnd, "wer_norm", False)
            g2 = _e2e_gate(f"{lbl}: DER", cnd, "der", False, min_gain=0.0001)
            g3 = _e2e_gate(f"{lbl}: word agreement", cnd, "agree_word", True)
            for g in (g1, g2, g3):
                if g:
                    gates.append(g)
            if g1 and g2 and g3:
                notes.append(
                    f"**{lbl}** — WER {g1['status']}; DER "
                    + ", ".join(f"`{pf['file']}` {pf['before']} → {pf['after']} ({pf['delta']:+.4f})"
                                for pf in g2["per_file"])
                    + "; word agreement "
                    + ", ".join(f"`{pf['file']}` {pf['before']:.1f} → {pf['after']:.1f}%"
                                for pf in g3["per_file"])
                    + ". Measured through the real pipeline, so these are the authoritative "
                      "verdicts; the text-replay gates below can only approximate them.")

    # --- Arm C: does Whisper word timing beat production on BOTH files? ---
    if all((r["arms"].get("C_whisper_timing") or {}).get("der") is not None for r in records):
        der_b = {r["key"]: r["production"]["der"] for r in records}
        der_a = {r["key"]: r["arms"]["C_whisper_timing"]["der"] for r in records}
        gates.append(SR.two_file_gate("Arm C — DER", der_b, der_a, "DER (ref_tiles)",
                                      higher_is_better=False))
        w_b = {r["key"]: r["production"]["agree_word"] for r in records}
        w_a = {r["key"]: r["arms"]["C_whisper_timing"]["agree_word"] for r in records}
        gates.append(SR.two_file_gate("Arm C — word agreement", w_b, w_a,
                                      "agreement (word)", higher_is_better=True))

    # --- candidate 1: hardened credit regex -------------------------------
    if all(("credit_regex" in cand.get(k, {})) for k in keys):
        before = {k: cand[k]["credit_regex"]["wer_before"] for k in keys}
        after = {k: cand[k]["credit_regex"]["wer_after"] for k in keys}
        # A recall-only fix: it can only restore text that the old rule wrongly
        # deleted, and it provably cannot touch a slice the old rule left alone.
        # So zero regression is the hard requirement and any gain at all is a win.
        gates.append(SR.two_file_gate("1. hardened credit regex", before, after,
                                      "WER (norm)", higher_is_better=False,
                                      tol=SR.TOL_WER, min_gain=0.0001))
        rec = sum(cand[k]["credit_regex"]["words_delta"] for k in keys)
        notes.append(f"Credit-regex candidate recovered **{rec:+d}** words across both files; "
                     f"slices touched: "
                     + ", ".join(f"{k}={len(cand[k]['credit_regex']['slices_changed'])}" for k in keys)
                     + ". Zero slices changed on the validation file is the required zero-FP result.")

    # --- candidate 2: engine words ---------------------------------------
    if all(("engine_words" in cand.get(k, {})) for k in keys):
        ok = all(cand[k]["engine_words"]["passes_validation"] for k in keys)
        notes.append(f"Engine word-list handoff: every file's captured pass emits a word list that passes "
                     f"production's own count validation "
                     f"({', '.join(str(cand[k]['engine_words']['n_words']) + '/' + str(cand[k]['engine_words']['n_text_words']) for k in keys)}); "
                     f"the list was discarded only because `has_word_timestamps` is undefined "
                     f"on every engine class. Validity: **{'PASS' if ok else 'FAIL'}**.")

    # --- candidate 3: fuzzy seam -----------------------------------------
    fs_ok = all("fuzzy_seam" in cand.get(k, {}) and "error" not in cand[k]["fuzzy_seam"] for k in keys)
    if fs_ok:
        before = {k: cand[k]["fuzzy_seam"]["baseline"]["der"] for k in keys}
        after = {k: cand[k]["fuzzy_seam"]["candidate_metrics"]["der"] for k in keys}
        g = SR.two_file_gate("3. fuzzy seam dedupe", before, after, "DER (ref_tiles)",
                             higher_is_better=False, min_gain=0.0001)
        gates.append(g)
        ab = {k: cand[k]["fuzzy_seam"]["baseline"]["agree_word"] for k in keys}
        aa = {k: cand[k]["fuzzy_seam"]["candidate_metrics"]["agree_word"] for k in keys}
        gates.append(SR.two_file_gate("3b. fuzzy seam — word agreement", ab, aa,
                                      "agreement (word)", higher_is_better=True))
    else:
        notes.append("Fuzzy seam dedupe: could not be evaluated on every file.")

    # --- candidate 4: VAD-aware word timing -------------------------------
    if all(("vad_word_timing" in cand.get(k, {})) for k in keys):
        before = {k: cand[k]["vad_word_timing"]["baseline"]["der"] for k in keys}
        after = {k: cand[k]["vad_word_timing"]["candidate_metrics"]["der"] for k in keys}
        gates.append(SR.two_file_gate("4. VAD-aware word timing", before, after,
                                      "DER (ref_tiles)", higher_is_better=False,
                                      min_gain=0.0001))
        ab = {k: cand[k]["vad_word_timing"]["baseline"]["agree_word"] for k in keys}
        aa = {k: cand[k]["vad_word_timing"]["candidate_metrics"]["agree_word"] for k in keys}
        gates.append(SR.two_file_gate("4b. VAD-aware word timing — word agreement", ab, aa,
                                      "agreement (word)", higher_is_better=True))

    # --- candidate 5: Pyannote boundary snapping --------------------------
    if all(("snap_words" in cand.get(k, {})) for k in keys):
        before = {k: cand[k]["snap_words"]["baseline"]["der"] for k in keys}
        after = {k: cand[k]["snap_words"]["candidate_metrics"]["der"] for k in keys}
        gates.append(SR.two_file_gate("5. Pyannote boundary snapping", before, after,
                                      "DER (ref_tiles)", higher_is_better=False,
                                      min_gain=0.0001))

    # --- design-space frontier (word list x timing) ------------------------
    fr = None
    try:
        from . import combined as CB
        fr = CB.frontier(keys)
        _log(f"frontier: best DER = {fr['best_der_cell']}, best agreement = "
             f"{fr['best_agreement_cell']}, conflict={fr['conflict']}")
    except Exception as e:
        _log(f"frontier unavailable: {e}")

    # --- hole recovery (speech the ASR never emitted) ----------------------
    recov: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        try:
            from . import recovery as R
            from . import recovery_gate as RG

            sweep_path = C.SUITE[k].out_dir / "recovery.json"
            # Prefer the best gated run (zero false positives) over the ungated one.
            gated = C.SUITE[k].out_dir / "recovery_gate_wps_0.3to3.json"
            gate_path = gated if gated.exists() else C.SUITE[k].out_dir / "recovery_gate.json"
            entry: Dict[str, Any] = {}
            if sweep_path.exists():
                entry["sweep"] = R.summarize(C.read_json(sweep_path))
            if gate_path.exists():
                entry["gate"] = C.read_json(gate_path)
            if entry:
                recov[k] = entry
        except Exception as e:
            _log(f"recovery: {k} unavailable ({e})")
    if recov:
        for k, v in recov.items():
            g = v.get("gate") or {}
            if g:
                _log(f"recovery {k}: DER {g['baseline']['der']} -> {g['recovered']['der']} "
                     f"({g['d_der']:+.4f}), WER {g['baseline']['wer_norm']} -> {g['recovered']['wer_norm']}")

    # --- discriminator sweep (reference-free rules vs reference labels) ----
    disc_rows_list: List[str] = []
    try:
        from . import discriminate as DIS

        col = {k: DIS.collect(k) for k in keys}
        sweep = DIS.evaluate_thresholds(col)
        for rule, pf in sweep.items():
            p1, p2 = pf.get("133306") or {}, pf.get("nlp_5min") or {}
            if not p1 or not p2:
                continue

            def _fmt(v, key):
                x = v.get(key)
                if x is None:
                    return "—"
                return f"{x:.2f}" if key in ("precision", "recall") else str(x)

            disc_rows_list.append(
                f"| `{rule}` | {p1['tp']}/{p1['fp']}/{p1['fn']} | {_fmt(p1, 'precision')} | "
                f"{_fmt(p1, 'recall')} | {p2['tp']}/{p2['fp']}/{p2['fn']} | "
                f"{_fmt(p2, 'precision')} | {_fmt(p2, 'recall')} |")
        _log(f"discriminator: {len(sweep)} rules swept")
    except Exception as e:
        _log(f"discriminator unavailable: {e}")

    out = {
        "files": records,
        "macro": SR.macro_average(records),
        "gates": gates,
        "notes": notes,
        "candidates": cand,
        "frontier": fr,
        "recovery": recov,
        "discriminator": disc_rows_list,
    }
    C.set_active(keys[0])
    C.write_json(C.OUT_ROOT / "suite" / "suite.json", out)
    md = SR.render_suite(records, gates, notes, rec_cand=cand, rec_e2e=e2e,
                         frontier=fr, recovery=recov, disc_rows=disc_rows_list)
    p = C.OUT_ROOT / "suite" / "suite.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    _log(f"suite written to {p}")
    for g in gates:
        _log(f"  gate {g['candidate']}: {g['status']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["verify", "repro", "arms", "candidates", "e2e", "score",
                             "report", "suite", "all"])
    ap.add_argument("--file", default=None,
                    help="suite member key (default: all members, primary first)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    _configure_logging(args.verbose)

    keys = [args.file] if args.file else list(C.SUITE)
    t0 = time.time()
    for key in keys:
        C.set_active(key)
        C.ensure_dirs()
        _log(f"=== {key} ({C.AUDIO_SOURCE.name}) ===")
        if args.stage in ("verify", "all"):
            stage_verify()
        if args.stage in ("repro", "all"):
            stage_repro()
        if args.stage in ("arms", "all"):
            stage_arms()
        if args.stage in ("candidates", "all"):
            stage_candidates()
        if args.stage in ("e2e", "all"):
            stage_e2e()
        if args.stage in ("score", "report", "all"):
            m = stage_score_and_report()
            _log(f"metrics written to {C.OUT_DIR / 'metrics.json'}")
            for row in m.get("summary_rows", []):
                _log(row)
    if args.stage in ("suite", "report", "score", "all") or (args.stage == "suite"):
        stage_suite()
    _log(f"stage '{args.stage}' complete for {keys} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
