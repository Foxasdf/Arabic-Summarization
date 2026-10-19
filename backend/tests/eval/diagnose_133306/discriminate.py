"""Deciding whether a word-stream hole is a pause or a dropped utterance.

The previous round established the problem precisely: re-decoding every hole that
contains audio recovers *something* at every window length, DER improves, and WER
gets worse — because a 2-3 s hole is usually the speaker pausing, and re-decoding
a pause returns either nothing or the neighbouring speech.

What separates the two cases is not the audio. Both are loud, Pyannote-bounded and
wordless. The separation has to come from the *decode*: a genuine dropped utterance
produces text that belongs between its neighbours, whereas a pause produces text
that repeats or contradicts them.

So this module scores each hole's decode by how well it fits the word stream around
it, using only signals available at inference time:

  * tail_distinct  — are the decoded words already the ones immediately before the
                     hole? (a pause re-decoded as the preceding utterance)
  * head_distinct  — same, against the words immediately after the hole
  * word_in_vocab  — share of decoded words that occur anywhere else in this same
                     recording (an Arabic word the engine never otherwise produces
                     in this speaker's stream is suspect)

The reference transcript is used **only** to label holes for evaluating the
discriminator, never as an input to it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import config as C

_HOLE_WIN = 16.0


def _norm(w: str) -> str:
    return re.sub(r"[^\w\u0600-\u06FF]", "", w or "")


def _tokens(words: List[Dict[str, Any]]) -> List[str]:
    return [_norm(w.get("word", "")) for w in words if _norm(w.get("word", ""))]


def _overlap_ratio(decoded: List[str], context: List[str]) -> float:
    """Fraction of decoded tokens that already appear in the adjacent context."""
    if not decoded:
        return 0.0
    ctx = set(context)
    return sum(1 for t in decoded if t in ctx) / len(decoded)


def features_for_hole(key: str, hole: Dict[str, Any],
                      decoded: List[str], base_words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reference-free features describing one hole's decode."""
    before = _tokens([w for w in base_words
                      if float(w.get("end", 0)) <= hole["start"] + 0.01])[-12:]
    after = _tokens([w for w in base_words
                     if float(w.get("start", 0)) >= hole["end"] - 0.01])[:12]
    vocab = set(_tokens(base_words))

    return {
        "n_decoded": len(decoded),
        "tail_dup": round(_overlap_ratio(decoded[:6], before), 3),
        "head_dup": round(_overlap_ratio(decoded[-6:], after), 3),
        "in_vocab": round(sum(1 for t in decoded if t in vocab) / len(decoded), 3) if decoded else 0.0,
        "dur": hole["dur"],
        "rms_db": hole.get("rms_db"),
        # Words per second of the hole. Ordinary Arabic speech runs ~2 words/s; a
        # short hole that decodes at 6-8 words/s is not recovering speech *from the
        # hole*, it is pulling in the surrounding utterances, which then duplicate
        # text already in the stream. This separates "recovered a missed line" from
        # "re-decoded the neighbours" without needing a reference.
        "wps": round(len(decoded) / hole["dur"], 2) if hole.get("dur") else 0.0,
    }


def label_hole(hole: Dict[str, Any], ref: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ground-truth label for evaluating the discriminator (NOT an input).

    A hole is a true miss if the reference transcribes speech inside it, and a
    pause if the reference has essentially nothing there.
    """
    inside = [x for x in ref
              if float(x["end"]) > hole["start"] + 0.4
              and float(x["start"]) < hole["end"] - 0.4]
    refw = sum(len((x.get("text") or "").split()) for x in inside)
    return {"ref_words_in_hole": refw, "is_true_miss": refw >= 5}


def collect(key: str) -> Dict[str, Any]:
    """Per-hole features (reference-free) alongside the evaluation label."""
    from . import recovery as R

    spec = C.SUITE[key]
    sweep_path = spec.out_dir / "recovery.json"
    sweep = C.read_json(sweep_path)
    base_words = C.read_json(spec.word_dir / "C_words.json")
    ref = spec.ref()

    rows = []
    for res in sweep["results"]:
        h = res["hole"]
        t = next((x for x in res["trials"] if abs(x["len"] - _HOLE_WIN) < 0.01), None)
        if t is None:
            continue
        decoded = [_norm(w) for w in (t.get("text") or "").split() if _norm(w)]
        rows.append({
            "hole": h,
            "decoded_text": (t.get("text") or "").strip(),
            "features": features_for_hole(key, h, decoded, base_words),
            "label": label_hole(h, ref),
        })
    return {"key": key, "rows": rows}


def evaluate_thresholds(collected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Does any reference-free rule separate true misses from pauses, on BOTH files?

    Sweeps simple rules so the choice is not hand-picked on one file. A rule is
    only usable if it holds on the primary and the validation file alike -- the
    same joint gate every other lever in this suite gets.
    """
    rules = {
        "dur >= 4s": lambda f: f["dur"] >= 4.0,
        "tail_dup < 0.5": lambda f: f["tail_dup"] < 0.5,
        "head_dup < 0.5": lambda f: f["head_dup"] < 0.5,
        "in_vocab >= 0.8": lambda f: f["in_vocab"] >= 0.8,
        "n_decoded >= 5": lambda f: f["n_decoded"] >= 5,
        # rate-based rules: reject decodes that are implausibly dense for the hole
        "wps <= 3": lambda f: f["wps"] <= 3.0,
        "wps <= 2.5": lambda f: f["wps"] <= 2.5,
        "wps in 0.3-3": lambda f: 0.3 <= f["wps"] <= 3.0,
        "wps<=3 and in_vocab>=0.5": lambda f: f["wps"] <= 3.0 and f["in_vocab"] >= 0.5,
        "wps in 0.3-3 and head_dup<0.5": lambda f: 0.3 <= f["wps"] <= 3.0 and f["head_dup"] < 0.5,
    }

    out: Dict[str, Any] = {}
    for name, fn in rules.items():
        per_file = {}
        for key, col in collected.items():
            tp = fp = fn_ = tn = 0
            for r in col["rows"]:
                pred = bool(fn(r["features"]))
                truth = bool(r["label"]["is_true_miss"])
                if pred and truth:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif not pred and truth:
                    fn_ += 1
                else:
                    tn += 1
            per_file[key] = {"tp": tp, "fp": fp, "fn": fn_, "tn": tn,
                             "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
                             "recall": round(tp / (tp + fn_), 3) if (tp + fn_) else None}
        out[name] = per_file
    return out
