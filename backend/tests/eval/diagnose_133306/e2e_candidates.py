"""End-to-end candidate verification.

``candidate_eval`` replays candidates against the *cached* per-slice text, which is
fast and exactly paired but only proves the text-level effect. This module re-runs
the **real pipeline** with the candidate patch active, all the way to final
speaker segments, so a candidate is verified where it actually matters:

    guardrail / word-list  ->  transcribe()  ->  align_timestamps()  ->  segments

That distinction caught a real discrepancy before: a text-only replay can look
correct while the segment stage behaves differently. Anything claimed as a win has
to survive this path.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import config as C


def _score(spec: C.FileSpec, segs: List[Dict[str, Any]], hub) -> Dict[str, Any]:
    from . import scoring
    from . import scoring_spk as SS

    ref = spec.ref()
    dur = float(C.read_json(spec.artifact_dir / "captured_asr_uniform.json").get("audio_duration")
                or spec.duration())
    dj = SS.score_diarization(ref, segs, dur)
    ag = SS.segment_agreement_weighted(ref, segs)
    st = dj["stages"]["ref_tiles"]
    txt = scoring.score_text(ref, segs)
    return {
        "n_segments": len(segs),
        "der": st["der"],
        "conf_s": st["conf_s"],
        "miss_s": st["miss_s"],
        "agree": ag["pct_ok"],
        "agree_word": ag["agree_word"],
        "agree_duration": ag["agree_duration"],
        "wer_norm": txt["wer_norm"],
        "wer_dialect_norm": txt["wer_dialect_norm"],
        "hyp_words": txt["hyp_words"],
    }


def run_e2e(key: str, candidates: List[str], force: bool = False) -> Dict[str, Any]:
    """Run the production segment path once per candidate and score it.

    The baseline is the *unpatched* production path on the same machine in the
    same call, so the delta cannot be attributed to drift between sessions.
    """
    from . import candidates as CN
    from . import pipeline_runner as PR

    spec = C.SUITE[key]
    asr_wav = spec.artifact_dir / "asr.16k.wav"
    turns_auto = C.read_json(spec.artifact_dir / "turns_auto.json")

    out_path = spec.out_dir / "e2e.json"
    if out_path.exists() and not force:
        cached = C.read_json(out_path)
        if set(candidates) <= set((cached.get("runs") or {})):
            return cached

    runs: Dict[str, Any] = {}
    if out_path.exists() and not force:
        runs = (C.read_json(out_path).get("runs") or {})

    for label in ["baseline"] + list(candidates):
        t0 = time.perf_counter()
        if label == "baseline":
            segs = PR.run_segments_arm(asr_wav, turns_auto, mode="continuous")["segments"]
            applied = False
        else:
            with CN.apply(label):
                segs = PR.run_segments_arm(asr_wav, turns_auto, mode="continuous")["segments"]
            applied = True
        runs[label] = {
            **_score(spec, segs, None),
            "applied": applied,
            "wall_s": round(time.perf_counter() - t0, 2),
        }

    out = {"key": key, "runs": runs, "candidates": candidates}
    C.write_json(out_path, out)
    return out


def compare(key: str, candidate: str) -> Optional[Dict[str, Any]]:
    """Baseline vs candidate, plus deltas, from the cached e2e run."""
    p = C.SUITE[key].out_dir / "e2e.json"
    if not p.exists():
        return None
    runs = (C.read_json(p) or {}).get("runs") or {}
    b, a = runs.get("baseline"), runs.get(candidate)
    if not b or not a:
        return None
    return {
        "key": key,
        "candidate": candidate,
        "baseline": b,
        "candidate_metrics": a,
        "d_der": round(a["der"] - b["der"], 4),
        "d_wer": round(a["wer_norm"] - b["wer_norm"], 4),
        "d_wer_dialect": round(a["wer_dialect_norm"] - b["wer_dialect_norm"], 4),
        "d_agree_word": round(a["agree_word"] - b["agree_word"], 2),
        "d_agree_duration": round(a["agree_duration"] - b["agree_duration"], 2),
        "d_words": a["hyp_words"] - b["hyp_words"],
    }
