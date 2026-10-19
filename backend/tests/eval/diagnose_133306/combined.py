"""The real 2x2: word list x word timing.

Why this module exists
----------------------
Earlier rounds compared candidates that turned out not to be independent. The
Cohere engine's own word list and Arm C's aligned list contain the **same token
sequence** on both suite files (verified: 794 == 794 and 668 == 668, zero
non-equal opcodes under a sequence alignment). Arm C is produced by aligning onto
that same transcript, so it re-times exactly the words the engine already emits.

That makes the comparison a 2x2 rather than a choice between two fixes:

                       fabricated timing      acoustic timing (Arm C)
    words discarded    production             arm_c
    words preserved    engine_words           combined

On both files the *sequence* is identical, so:
  * production and arm_c differ ONLY in timing
  * engine_words and combined differ ONLY in timing
  * production and engine_words differ only if the text path drops words

The last point is the open question, and it is why this is measured end to end
rather than asserted.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import config as C


def _arm_c_words(spec: C.FileSpec) -> Optional[List[Dict[str, Any]]]:
    p = spec.word_dir / "C_words.json"
    if not p.exists():
        return None
    w = C.read_json(p)
    return w.get("words") if isinstance(w, dict) else w


def _retime(engine_words: List[Dict[str, Any]],
            arm_c_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give engine words Arm C's acoustic timing, preserving order and count.

    Both lists come from the same Cohere transcript, so they normally agree token
    for token. Where they do not, a sequence alignment keeps them in order and any
    engine word without a counterpart is interpolated between its neighbours so
    the result stays monotonic -- never dropping a word, which is the whole point
    of preserving the engine list.
    """
    import difflib

    e = [w.get("word", "") for w in engine_words]
    a = [w.get("word", "") for w in arm_c_words]
    sm = difflib.SequenceMatcher(a=e, b=a, autojunk=False)

    mapped: Dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                mapped[i1 + k] = j1 + k

    out: List[Dict[str, Any]] = []
    for i, w in enumerate(engine_words):
        if i in mapped:
            src = arm_c_words[mapped[i]]
            out.append({"word": w.get("word", ""), "start": src["start"], "end": src["end"]})
        else:
            out.append({"word": w.get("word", ""), "start": None, "end": None})

    last_end = 0.0
    for i, w in enumerate(out):
        if w["start"] is not None:
            last_end = w["end"]
            continue
        nxt = next((out[j]["start"] for j in range(i + 1, len(out))
                    if out[j]["start"] is not None), None)
        lo = last_end
        hi = nxt if nxt is not None else lo + 0.3
        if hi <= lo:
            hi = lo + 0.1
        w["start"] = round(lo, 2)
        w["end"] = round(hi, 2)
        last_end = w["end"]
    return out


def word_sequence_identity(key: str) -> Dict[str, Any]:
    """Do the engine word list and Arm C hold the same tokens in the same order?"""
    import difflib

    spec = C.SUITE[key]
    cap = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
    engine = [w.get("word", "") for w in (cap.get("words") or [])]
    arms = _arm_c_words(spec)
    if arms is None:
        return {"key": key, "available": False}
    ac = [w.get("word", "") for w in arms]
    sm = difflib.SequenceMatcher(a=engine, b=ac, autojunk=False)
    diffs = [{"op": t, "n_engine": i2 - i1, "n_armC": j2 - j1}
             for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal"]
    return {
        "key": key,
        "available": True,
        "n_engine_words": len(engine),
        "n_armC_words": len(ac),
        "identical_sequence": engine == ac,
        "n_diff_regions": len(diffs),
        "diffs": diffs[:10],
    }


def run_matrix(key: str, force: bool = False) -> Dict[str, Any]:
    """Score the 2x2 and report DER *and* agreement for each cell."""
    from . import pipeline_runner as PR
    from app.services.alignment import align_timestamps

    spec = C.SUITE[key]
    turns = C.read_json(spec.artifact_dir / "turns_auto.json")
    asr_wav = spec.artifact_dir / "asr.16k.wav"

    out_path = spec.out_dir / "matrix.json"
    if out_path.exists() and not force:
        return C.read_json(out_path)

    cap = C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
    engine_words = cap.get("words") or []
    arms = _arm_c_words(spec)

    cells: Dict[str, Any] = {}

    # fabricated timing, words as the text path keeps them (shipped behaviour)
    prod = PR.run_segments_arm(asr_wav, turns, mode="continuous")["segments"]
    cells["production"] = _score(spec, prod)
    cells["production"]["text_words"] = sum(len(s.get("text", "").split()) for s in prod)

    # fabricated timing, engine words preserved -- measured end to end because the
    # text path may drop or re-split words, and only a real run can show that.
    from . import candidates as CN
    with CN.candidate_engine_words():
        ew = PR.run_segments_arm(asr_wav, turns, mode="continuous")["segments"]
    cells["engine_words"] = _score(spec, ew)
    cells["engine_words"]["text_words"] = sum(len(s.get("text", "").split()) for s in ew)

    if arms:
        cells["arm_c"] = _score(spec, align_timestamps(arms, turns))
        cells["arm_c"]["text_words"] = sum(len(s.get("text", "").split())
                                           for s in align_timestamps(arms, turns))
        # Acoustic timing applied to the preserved engine words.
        #
        # Because the engine list and the Arm C list are the *same sequence* (checked
        # by word_sequence_identity), re-timing the engine words with Arm C's stamps
        # is by construction identical to arm_c. An earlier version of this cell
        # passed the raw engine words instead, which carry the fabricated timing --
        # so it silently reproduced engine_words and looked like a fourth distinct
        # configuration. It was not. We build the cell correctly and then record
        # whether it coincides with arm_c, rather than reporting a phantom cell.
        retimed = _retime(engine_words, arms)
        cells["combined"] = _score(spec, align_timestamps(retimed, turns))
        cells["combined"]["text_words"] = sum(len(s.get("text", "").split())
                                              for s in align_timestamps(retimed, turns))
        cells["combined"]["equals_arm_c"] = (cells["combined"]["der"] == cells["arm_c"]["der"]
                                             and cells["combined"]["agree_word"]
                                             == cells["arm_c"]["agree_word"])

    out = {
        "key": key,
        "cells": cells,
        "identity": word_sequence_identity(key),
        "engine_word_count": len(engine_words),
    }
    C.write_json(out_path, out)
    return out


def _score(spec: C.FileSpec, segs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from . import scoring
    from . import scoring_spk as SS

    ref = spec.ref()
    dur = float(C.read_json(spec.artifact_dir / "captured_asr_uniform.json")
                .get("audio_duration") or spec.duration())
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


def frontier(keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Summarise which cell is best per metric, across files.

    Because DER and speaker agreement disagree about which timing is better, there
    is no single 'best' configuration. Reporting the Pareto front makes that
    explicit instead of letting one metric silently win.
    """
    keys = keys or list(C.SUITE)
    per_file: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        p = C.SUITE[k].out_dir / "matrix.json"
        if not p.exists():
            run_matrix(k)
        per_file[k] = C.read_json(p)["cells"]

    # macro-average each cell
    cells = ["production", "engine_words", "arm_c", "combined"]
    macro: Dict[str, Any] = {}
    for c in cells:
        vals = {metric: [] for metric in ("der", "agree_word", "agree_duration", "wer_norm")}
        for k in keys:
            cell = per_file.get(k, {}).get(c)
            if not cell:
                continue
            for metric in vals:
                if cell.get(metric) is not None:
                    vals[metric].append(cell[metric])
        macro[c] = {m: (round(sum(v) / len(v), 4) if v else None) for m, v in vals.items()}

    best_der = min((c for c in cells if macro[c]["der"] is not None), key=lambda c: macro[c]["der"])
    best_agree = max((c for c in cells if macro[c]["agree_word"] is not None),
                     key=lambda c: macro[c]["agree_word"])
    return {
        "per_file": per_file,
        "macro": macro,
        "best_der_cell": best_der,
        "best_agreement_cell": best_agree,
        "conflict": best_der != best_agree,
    }
