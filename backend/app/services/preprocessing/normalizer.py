"""
Arabic text normalization and WER/CER metric calculation for ASR evaluation (Production v3.1).
Supports both raw (strict) and normalized evaluation modes.
"""

import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional

NORMALIZER_VERSION = "1.0.0"

# Tashkeel (diacritics) Unicode range
TASHKEEL_REGEX = re.compile(r"[\u064B-\u065F\u0670]")
# Tatweel (kashida)
TATWEEL_REGEX = re.compile(r"\u0640")
# Punctuation & symbols
PUNCTUATION_REGEX = re.compile(r"[،؛؟«»“”!?:;,\.\(\)\[\]\{\}\'\"\\/\-_+=*&^%$#@~`|\<\>]")
# Whitespace
WHITESPACE_REGEX = re.compile(r"\s+")


class ArabicTextNormalizer:
    """
    Standard Arabic text normalizer with configurable stages.
    """
    def __init__(self, version: str = NORMALIZER_VERSION):
        self.version = version

    def normalize(
        self,
        text: str,
        remove_tashkeel: bool = True,
        remove_tatweel: bool = True,
        remove_punct: bool = True,
        unify_alef: bool = True,
        unify_yaa: bool = True,
        unify_taa_marbuta: bool = True,
    ) -> str:
        if not text:
            return ""

        t = text
        if remove_tashkeel:
            t = TASHKEEL_REGEX.sub("", t)

        if remove_tatweel:
            t = TATWEEL_REGEX.sub("", t)

        if remove_punct:
            t = PUNCTUATION_REGEX.sub(" ", t)

        if unify_alef:
            # Unify أ, إ, آ, ٱ -> ا
            t = re.sub(r"[أإآٱ]", "ا", t)

        if unify_yaa:
            # Unify ى -> ي
            t = re.sub(r"ى", "ي", t)

        if unify_taa_marbuta:
            # Unify ة -> ه
            t = re.sub(r"ة", "ه", t)

        t = WHITESPACE_REGEX.sub(" ", t).strip()
        return t

    def normalize_soft(self, text: str) -> str:
        """
        Soft normalization: removes tashkeel, tatweel, and punctuation,
        WITHOUT unifying ta-marbuta or alef, to preserve orthographic distinctions.
        """
        return self.normalize(
            text,
            remove_tashkeel=True,
            remove_tatweel=True,
            remove_punct=True,
            unify_alef=False,
            unify_yaa=False,
            unify_taa_marbuta=False,
        )


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> Tuple[int, int, int, int]:
    """
    Standard Levenshtein distance dynamic programming.
    Returns (distance, substitutions, deletions, insertions).
    """
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                )

    # Backtrack to count operations
    i, j = n, m
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            inss += 1
            j -= 1
        else:
            break

    return dp[n][m], subs, dels, inss


def calculate_wer_cer(
    reference: str,
    hypothesis: str,
    normalizer: Optional[ArabicTextNormalizer] = None,
) -> Dict[str, Any]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER)
    reporting both RAW and NORMALIZED values.
    """
    normalizer = normalizer or ArabicTextNormalizer()

    # Raw computation (clean whitespace/punct only so whitespace doesn't distort words)
    raw_ref = normalizer.normalize_soft(reference)
    raw_hyp = normalizer.normalize_soft(hypothesis)

    ref_words_raw = raw_ref.split()
    hyp_words_raw = raw_hyp.split()
    w_dist_raw, w_sub_raw, w_del_raw, w_ins_raw = levenshtein_distance(ref_words_raw, hyp_words_raw)
    raw_wer = (w_dist_raw / len(ref_words_raw)) if ref_words_raw else (1.0 if hyp_words_raw else 0.0)

    ref_chars_raw = list(raw_ref.replace(" ", ""))
    hyp_chars_raw = list(raw_hyp.replace(" ", ""))
    c_dist_raw, _, _, _ = levenshtein_distance(ref_chars_raw, hyp_chars_raw)
    raw_cer = (c_dist_raw / len(ref_chars_raw)) if ref_chars_raw else (1.0 if hyp_chars_raw else 0.0)

    # Fully normalized computation
    norm_ref = normalizer.normalize(reference)
    norm_hyp = normalizer.normalize(hypothesis)

    ref_words_norm = norm_ref.split()
    hyp_words_norm = norm_hyp.split()
    w_dist_norm, w_sub_norm, w_del_norm, w_ins_norm = levenshtein_distance(ref_words_norm, hyp_words_norm)
    norm_wer = (w_dist_norm / len(ref_words_norm)) if ref_words_norm else (1.0 if hyp_words_norm else 0.0)

    ref_chars_norm = list(norm_ref.replace(" ", ""))
    hyp_chars_norm = list(norm_hyp.replace(" ", ""))
    c_dist_norm, _, _, _ = levenshtein_distance(ref_chars_norm, hyp_chars_norm)
    norm_cer = (c_dist_norm / len(ref_chars_norm)) if ref_chars_norm else (1.0 if hyp_chars_norm else 0.0)

    return {
        "raw_wer": round(raw_wer, 4),
        "raw_cer": round(raw_cer, 4),
        "norm_wer": round(norm_wer, 4),
        "norm_cer": round(norm_cer, 4),
        "ref_word_count": len(ref_words_norm),
        "hyp_word_count": len(hyp_words_norm),
        "substitutions": w_sub_norm,
        "deletions": w_del_norm,
        "insertions": w_ins_norm,
    }
