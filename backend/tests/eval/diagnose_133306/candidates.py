"""Candidate fixes, applied at runtime for gate evaluation only.

Each candidate is a self-contained, reviewable change to a *production* code path,
expressed as a context manager that patches the module attribute on entry and
restores the exact original on exit. Nothing here edits ``backend/app/**``; a
candidate that passes the 2-file gate is meant to be promoted by hand, with this
file serving as the exact diff.

Design rules, to stay generalisable:
  * no per-file constants, no term maps, no proper-noun lists;
  * each candidate is a single mechanical change to one rule;
  * each carries the evidence that motivated it, in a docstring.
"""
from __future__ import annotations

import contextlib
import re
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Candidate 1: transcription-credit guardrail
# ---------------------------------------------------------------------------
# Production rule (backend/app/models/asr/guardrails.py:23):
#     r"(التفريغ|التدقيق|إعداد|تسجيل)\s+(و\s*التدقيق)?\s*[:\.]?.*"
#
# Defect: it matches the *bare noun* anywhere in the sentence, and the trailing
# `.*` then deletes everything after it. `تسجيل` is the ordinary Levantine word
# for "recording", so mid-sentence speech such as "كنا عم نفتح تسجيل عم نحكي
# عن الاكل" is wiped to the empty string. Measured on 133306: 42 of 57 words in
# one 28 s slice; 7 of 8 probe sentences wiped; 2 true credits correctly caught.
#
# Fix: a credit line is a *label*, so require label shape --
#   (a) one of the credit words (with optional "والتدقيق" tail) followed by an
#       explicit delimiter ( ':' , '-' , '–' , '|' ), or
#   (b) the word at the very start of the segment with nothing after it.
# The `.*` tail is retained after the delimiter because a real credit trails its
# names ("الترجمة من قبل أحمد" is matched by its own entry, unchanged).
CREDIT_PATTERN_OLD = r"(التفريغ|التدقيق|إعداد|تسجيل)\s+(و\s*التدقيق)?\s*[:\.]?.*"
# Optional definite article, then the credit noun, then a required delimiter.
# Note the alternation: the definite article is spelled out per noun rather than
# captured as `ال?`, because `ال?` before `إعداد` lets the regex swallow the
# hamza as a fake article and then fail to match `عداد`.
CREDIT_CORE = (
    r"(?:التفريغ|التدقيق|الإعداد|الإخراج|الترجمة|تفريغ|تدقيق|إعداد|تسجيل|التسجيل)"
    r"(?:\s*و\s*(?:ال)?(?:تدقيق|تفريغ|إعداد))?"
)
CREDIT_PATTERN_NEW = (
    r"(?:(?<![\w\u0600-\u06FF])(?:" + CREDIT_CORE + r")\s*[:：\-–—|].*)"
    r"|(?:^\s*(?:" + CREDIT_CORE + r")\s*$)"
)


def harden_credit_pattern(patterns: List[str]) -> List[str]:
    """Replace the defective credit rule in a pattern list, preserving order."""
    out: List[str] = []
    for p in patterns:
        if p == CREDIT_PATTERN_OLD:
            out.append(CREDIT_PATTERN_NEW)
        else:
            out.append(p)
    return out


@contextlib.contextmanager
def candidate_credit_regex(enabled: bool = True):
    """Swap the credit rule in guardrails.HALLUCINATION_PATTERNS for the gate run."""
    import app.models.asr.guardrails as G

    if not enabled:
        yield False
        return
    original = list(G.HALLUCINATION_PATTERNS)
    patched = harden_credit_pattern(original)
    G.HALLUCINATION_PATTERNS[:] = patched
    try:
        yield True
    finally:
        G.HALLUCINATION_PATTERNS[:] = original


# ---------------------------------------------------------------------------
# Candidate 2: use the engine's own word list
# ---------------------------------------------------------------------------
# Production code (backend/app/services/transcription.py:133-134):
#     has_native_words = getattr(engine, "has_word_timestamps", False)
#     words = asr_res.get("words") or [] if has_native_words else []
#
# `has_word_timestamps` is defined on *no* engine class (verified: absent from
# BaseASREngine and CohereASREngine), so getattr returns False for every engine
# and the emitted word list is unconditionally discarded -- then thrown away
# again because forced_alignment re-derives timing from the same text.
#
# The fix is a capability declaration, not a behaviour switch: engines that do
# emit words advertise it, others keep the forced-alignment fallback. The
# existing validation (word count must match text within 5%) still guards
# against a bad list.
def install_word_capability() -> bool:
    """Declare that BaseASREngine can emit a usable word list."""
    from app.models.asr.base import BaseASREngine

    if "has_word_timestamps" in vars(BaseASREngine):
        return False
    BaseASREngine.has_word_timestamps = True
    return True


def remove_word_capability(installed: bool) -> None:
    if not installed:
        return
    from app.models.asr.base import BaseASREngine

    if "has_word_timestamps" in vars(BaseASREngine):
        delattr(BaseASREngine, "has_word_timestamps")


@contextlib.contextmanager
def candidate_engine_words(enabled: bool = True):
    installed = install_word_capability() if enabled else False
    try:
        yield installed
    finally:
        remove_word_capability(installed)


# ---------------------------------------------------------------------------
# Candidate 3: dialect-orthography normalisation (SCORING ONLY)
# ---------------------------------------------------------------------------
# Important: this is a *diagnostic metric*, never applied to guardrails or to
# the emitted transcript. It exists only to answer "how much of the residual WER
# is spelling convention rather than misrecognition?".
#
# Every rule below is a documentary Levantine<->MSA orthography pair, applied
# symmetrically to both sides so it cannot favour the hypothesis. It is not
# tuned per file: the rule list is fixed and identical for every recording.
# Rules must be *convergent*: both spellings of a pair have to land on the same
# canonical form. A one-way rewrite is not enough and silently does nothing.
# For example mapping ذ->د alone gives هدا for هذا while هاد stays هاد, so the
# pair still disagrees. Sound shifts therefore standardise on one character.
LEVANTINE_RULES: List[tuple[str, str]] = [
    # Whole-word / prefix rules come FIRST: once a character-level rule has
    # rewritten the token, a later ^anchored rule would no longer match the form
    # it was written for. (This ordering bug is why هذا did not converge on
    # هاد: ذ->د turned it into هدا before ^هذا$ was ever tried.)
    # Both spellings are listed so the rule converges whichever appears.
    (r"^(?:هذا|هدا)$", "هاد"),
    (r"^(?:هذه|هده)$", "هادي"),
    (r"^يلي$", "اللي"),
    (r"^انو$", "انه"),
    (r"^انو", "انه"),
    # هالقمة (dialect demonstrative + article) vs اللقمه (MSA)
    (r"^هال", "ال"),
    (r"^هب", "ب"),
    # 3rd-person masculine suffix -o/-u (و) vs MSA ه -- converge both on ه
    (r"و$", "ه"),
    # final -ih/-eh (ة) vs ه -- converge both on ه
    (r"ة$", "ه"),
    # interdental realisation, converged on the dialect symbol
    #   كتير/كثير -> كتير ; تلاتة/ثلاثه -> تلاته
    ("ث", "ت"),
    #   هذا/هاد/هدا -> هاد ; ذهب/دهب -> دهب
    ("ذ", "د"),
    # alef maqsura / ya spelling
    ("ى$", "ي"),
    # hamza seats are frequently dropped in dialect spelling
    ("أ", "ا"),
    ("إ", "ا"),
    ("آ", "ا"),
]


def dialect_norm_token(tok: str) -> str:
    """Map one normalised token onto a dialect-insensitive spelling."""
    for pat, rep in LEVANTINE_RULES:
        tok = re.sub(pat, rep, tok)
    return tok


def dialect_norm_text(text: str) -> str:
    return " ".join(dialect_norm_token(t) for t in text.split())


# ---------------------------------------------------------------------------
# Candidate 4: fuzzy seam dedupe
# ---------------------------------------------------------------------------
# `BaseASREngine._find_word_overlap` compares raw strings, so a seam where the
# two slices spell the same word differently ("اللقمه" vs "اللقمة", or an
# elided alef) leaves a duplicate. The candidate compares normalised forms and
# allows a small character-distance slack.
def _prod_norm_word(w: str) -> str:
    """Replicate ``BaseASREngine._find_word_overlap``'s own normalisation exactly.

    This must match production or the fuzzy comparison operates in a different
    space than the exact one: normalising differently made every candidate pair
    look unequal, so longer-but-wrong spans silently matched nothing and the
    fuzzier matcher removed *fewer* duplicates than the exact one.
    """
    w = re.sub(r"[،,.\?!؟;:\"\'\-\(\)\[\]]", "", w).strip()
    w = re.sub(r"[إأآٱ]", "ا", w)
    w = re.sub(r"ة", "ه", w)
    w = re.sub(r"ى", "ي", w)
    return w


def _close(a: str, b: str) -> bool:
    """Match production's own fuzzy rule, not a stricter one.

    Production accepts ``a == b or a.startswith(b) or b.startswith(a)``. Any
    candidate that is *stricter* than production will dedupe less than
    production, which is how an earlier version of this function appeared to
    duplicate text. We match the same rule and additionally allow a single
    inner substitution (the case production misses).
    """
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def find_word_overlap_fuzzy(tail: List[str], head: List[str], max_k: int = 12) -> int:
    """Exact pass, then production's own k-1-of-k tolerant pass, one word wider.

    Production already runs a tolerant pass over k >= 3 accepting k-1 matching
    words (a==b or one is a prefix of the other). The gap this candidate targets
    is the *single* word pair that fails both tests (e.g. قصص vs صوص), which
    production rejects even though every neighbouring word matches. So we allow
    up to two deviating pairs at k >= 4 -- strictly more tolerant than
    production, never less, so this can only remove duplicates that production
    leaves behind and can never duplicate text.
    """
    tn = [_prod_norm_word(w) for w in tail]
    hn = [_prod_norm_word(w) for w in head]
    limit = min(len(tn), len(hn), max_k)

    # 1. exact normalised match, longest first (production rule 1)
    for k in range(limit, 0, -1):
        if tn[-k:] == hn[:k]:
            return k
    # 2. production rule 2: at most one deviating pair
    for k in range(limit, 2, -1):
        if sum(1 for a, b in zip(tn[-k:], hn[:k]) if not _close(a, b)) <= 1:
            return k
    # 3. widening: at most two deviating pairs, but only for longer overlaps
    for k in range(limit, 3, -1):
        if sum(1 for a, b in zip(tn[-k:], hn[:k]) if not _close(a, b)) <= 2:
            return k
    return 0


@contextlib.contextmanager
def candidate_fuzzy_seam(enabled: bool = True):
    """Patch BaseASREngine._find_word_overlap to tolerate spelling-only drift.

    The production method is a ``@staticmethod``, so it is replaced with another
    staticmethod -- wrapping it like an instance method would shift the arguments.
    """
    from app.models.asr.base import BaseASREngine

    if not enabled or not hasattr(BaseASREngine, "_find_word_overlap"):
        yield False
        return
    original = BaseASREngine.__dict__.get("_find_word_overlap")

    @staticmethod
    def patched(tail_words, head_words):
        try:
            return find_word_overlap_fuzzy(list(tail_words), list(head_words))
        except Exception:
            return BaseASREngine.__dict__["_find_word_overlap"].__func__(tail_words, head_words)

    BaseASREngine._find_word_overlap = patched
    try:
        yield True
    finally:
        if original is not None:
            BaseASREngine._find_word_overlap = original
        elif "_find_word_overlap" in BaseASREngine.__dict__:
            delattr(BaseASREngine, "_find_word_overlap")


@contextlib.contextmanager
def apply(cand: str, enabled: bool = True):
    """Convenience dispatcher so the runner can name a candidate by string."""
    table: Dict[str, Callable[..., Any]] = {
        "credit_regex": candidate_credit_regex,
        "engine_words": candidate_engine_words,
        "fuzzy_seam": candidate_fuzzy_seam,
    }
    if cand not in table:
        raise KeyError(f"unknown candidate {cand!r}; have {sorted(table)}")
    with table[cand](enabled) as applied:
        yield applied
