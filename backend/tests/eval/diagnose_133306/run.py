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

from .config import BACKEND, ROOT  # noqa: E402  (path bootstrap below)

sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "backend" / "tests" / "eval"))

from diagnose_133306 import config as C  # noqa: E402

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
    turns_auto = C.read_json(C.ARTIFACT_DIR / "turns_auto.json")
    turns_oracle = C.read_json(C.ARTIFACT_DIR / "turns_oracle3.json")
    captured = C.read_json(C.ARTIFACT_DIR / "captured_asr_uniform.json")
    asr_wav = C.ARTIFACT_DIR / "asr.16k.wav"
    audio_dur = float(captured.get("audio_duration") or 397.8)

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
    arms["A_oracle3"] = _from_words(words_uniform, turns_oracle, "A_oracle3")

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
        arms["C_whisper_oracle3"] = _from_words(wc["words"], turns_oracle, "C_whisper_oracle3")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["verify", "repro", "arms", "score", "report", "all"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    _configure_logging(args.verbose)
    C.ensure_dirs()

    t0 = time.time()
    if args.stage in ("verify", "all"):
        stage_verify()
    if args.stage in ("repro", "all"):
        stage_repro()
    if args.stage in ("arms", "all"):
        stage_arms()
    if args.stage in ("score", "report", "all"):
        m = stage_score_and_report()
        _log(f"metrics written to {C.OUT_DIR / 'metrics.json'}")
        for row in m.get("summary_rows", []):
            _log(row)
    _log(f"stage '{args.stage}' complete in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
