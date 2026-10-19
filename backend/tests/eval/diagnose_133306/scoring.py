"""Text scoring: WER/CER (raw + normalised) and per-speaker WER.

NO term map is applied by the primary scorer. ``DEFAULT_TERMS`` below is kept
only as a *documented negative result*: applied to this file it improved WER by
0.0044 by rewriting six tokens, all one dish name (see ``term_map_probe``). That
is precisely the frozen_v1 overfit pattern that scored 0.4557 in-sample and
0.5022 held-out, so it is deliberately excluded from every headline number.

Scoring is otherwise plain: strip punctuation, normalise with the project's
``ArabicTextNormalizer``, and compute Levenshtein WER/CER. The one structural
choice is that the hypothesis is scored as the concatenation of its segments, so
no credit is given or taken for segmentation.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from . import config as C

PUNCT = set("،؛؟«»“”!?:;,.\"'()[]{}…-–—")

# ---------------------------------------------------------------------------
# Term map: retained ONLY as an audit of a rejected idea (never applied)
# ---------------------------------------------------------------------------
DEFAULT_TERMS: Dict[str, Any] = {
    "_meta": {
        "purpose": "Hypothesis-side normalization for tokens where the human-corrected reference "
                   "codifies a durable lexical decision (Levantine food terms, numerals).",
        "rule": "Each entry needs: replacement, reason, and >=1 example with timestamps.",
        "version": 1,
    },
    "types": ["يبرق", "يلنجي", "كبة"],
    "terms": {
        "يبرا": {
            "to": "يبرق",
            "reason": "Levantine quince name (quince dolma); Cohere renders the /q/ as a glottal stop "
                      "(اليا, يبرا). Reference codifies يبرق at 314.0-317.5 and 323.5-325.5.",
            "examples": [["314.00", "317.50", "يبرق، بحب اليبرق، ورق عنب..."],
                         ["323.50", "325.50", "يبرق قصدك اليبرق ولا يلنجي؟"]],
        },
        "اليبرا": {
            "to": "اليبرق",
            "reason": "Definite-form of the same term as يبرا (glottal-stop rendering).",
            "examples": [["314.00", "317.50", "بحب اليبرق"]],
        },
        "يبري": {
            "to": "يبرق",
            "reason": "Phonetic /i/ rendering of the same term.",
            "examples": [["314.00", "317.50", "بحب اليبرق"]],
        },
        "اليلنجي": {
            "to": "يلنجي",
            "reason": "Yalangi (vine leaves) takes the article in dialect; reference writes يلنجي.",
            "examples": [["323.50", "325.50", "يبرق قصدك اليبرق ولا يلنجي؟"]],
        },
        "كبه": {
            "to": "كبة",
            "reason": "Kibbeh: Cohere omits taa marbuta; the normalizer unifies ة->ه in both sides but "
                      "raw CER keeps it, so this entry keeps the raw metric honest.",
            "examples": [["361.50", "364.00", "كبة هيك... كبة مشوية..."]],
        },
    },
}


def load_terms() -> Dict[str, Any]:
    if C.TERM_MAP.exists():
        return C.read_json(C.TERM_MAP)
    C.ensure_dirs()
    C.write_json(C.TERM_MAP, DEFAULT_TERMS)
    return DEFAULT_TERMS


def apply_terms(text: str, terms: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Whole-token, case-insensitive replacement; returns (text, hits)."""
    from app.services.preprocessing.normalizer import ArabicTextNormalizer

    if not text:
        return "", []
    norm = ArabicTextNormalizer()
    mapping = {}
    for src, spec in (terms.get("terms") or {}).items():
        mapping[norm.normalize(src).strip()] = spec["to"]
    if not mapping:
        return text, []

    out, hits = [], []
    for tok in text.split():
        core = tok.strip("".join(PUNCT))
        key = norm.normalize(core).strip()
        if key in mapping:
            target = mapping[key]
            # Preserve leading punctuation prefix if any (rare); keep simple.
            out.append(target)
            hits.append({"from": tok, "to": target})
        else:
            out.append(tok)
    return " ".join(out), hits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wer_cer(ref: str, hyp: str) -> Dict[str, Any]:
    from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer

    return calculate_wer_cer(ref, hyp, ArabicTextNormalizer())


def subsequence_wer(ref: str, hyp: str) -> Dict[str, Any]:
    """Cross-check metric: difflib opcodes on normalised tokens (independent of our DP)."""
    from app.services.preprocessing.normalizer import ArabicTextNormalizer

    norm = ArabicTextNormalizer()
    r = norm.normalize(ref).split()
    h = norm.normalize(hyp).split()
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    sub = dele = ins = eq = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            eq += i2 - i1
        elif tag == "replace":
            sub += min(i2 - i1, j2 - j1)
            dele += max(0, (i2 - i1) - (j2 - j1))
            ins += max(0, (j2 - j1) - (i2 - i1))
        elif tag == "delete":
            dele += i2 - i1
        elif tag == "insert":
            ins += j2 - j1
    edits = sub + dele + ins
    return {"wer": round(edits / max(1, len(r)), 4), "subs": sub, "dels": dele, "ins": ins,
            "ref_words": len(r), "hyp_words": len(h), "matched": eq}


def speaker_attributed_text(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, List[str]] = {}
    for s in segments:
        out.setdefault(s.get("speaker", "?"), []).append(s.get("text", ""))
    return {k: " ".join(v).strip() for k, v in out.items()}


def per_speaker_wer(ref_segments: List[Dict[str, Any]], hyp_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    ref_by = speaker_attributed_text(ref_segments)
    hyp_by = speaker_attributed_text(hyp_segments)
    rows = {}
    for spk, ref_txt in ref_by.items():
        r = wer_cer(ref_txt, hyp_by.get(spk, ""))
        rows[spk] = {"wer_norm": r["norm_wer"], "cer_norm": r["norm_cer"],
                     "ref_words": r["ref_word_count"], "hyp_words": r["hyp_word_count"],
                     "deletions": r["deletions"], "substitutions": r["substitutions"]}
    return rows


def score_text(ref_segments: List[Dict[str, Any]], hyp_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline WER/CER. No term map, no per-file tuning -- see term_map_probe()."""
    ref_text = " ".join(s.get("text", "") for s in ref_segments).strip()
    hyp_text = " ".join(s.get("text", "") for s in hyp_segments).strip()
    base = wer_cer(ref_text, hyp_text)
    return {
        "n_segments_hyp": len(hyp_segments),
        "ref_words": base["ref_word_count"],
        "hyp_words": base["hyp_word_count"],
        "wer_norm": base["norm_wer"],
        "cer_norm": base["norm_cer"],
        "wer_raw": base["raw_wer"],
        "cer_raw": base["raw_cer"],
        "subs": base["substitutions"],
        "dels": base["deletions"],
        "ins": base["insertions"],
        "subseq": subsequence_wer(ref_text, hyp_text),
        "per_speaker": per_speaker_wer(ref_segments, hyp_segments),
        "hyp_text": hyp_text,
    }


def term_map_probe(ref_segments: List[Dict[str, Any]], hyp_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report what a per-file proper-noun map *would* buy, to keep it honest.

    Frozen_v1 learned this lesson expensively: a term map tuned on the reference
    gained 0.0466 WER in-sample and lost 0.0466 out-of-sample. So the map is
    measured and reported, never applied. A gain concentrated in one or two
    distinct tokens is a red flag, not a result.
    """
    ref_text = " ".join(s.get("text", "") for s in ref_segments).strip()
    hyp_text = " ".join(s.get("text", "") for s in hyp_segments).strip()
    base = wer_cer(ref_text, hyp_text)
    mapped, hits = apply_terms(hyp_text, load_terms())
    if not hits:
        return {"applied": [], "distinct_tokens": 0, "wer_gain": 0.0,
                "note": "no map entry matched the hypothesis"}
    m = wer_cer(ref_text, mapped)
    gain = round(base["norm_wer"] - m["norm_wer"], 4)
    distinct = len({h["from"] for h in hits})
    return {
        "applied": hits,
        "n_edits_rewritten": len(hits),
        "distinct_tokens": distinct,
        "wer_without": base["norm_wer"],
        "wer_with": m["norm_wer"],
        "wer_gain": gain,
        "verdict": ("REJECT: gain is concentrated in too few tokens to transfer"
                    if distinct <= 3 or gain < 0.01 else "would need held-out confirmation"),
        "note": "For reference: frozen_v1 in-sample 0.4557 vs held-out 0.5022 with this same technique.",
    }
