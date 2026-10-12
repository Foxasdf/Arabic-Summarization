"""
Evaluation Metrics for Arabic ASR and Diarization.
Implements:
1. Standardized Arabic text normalizer (tashkeel stripping, alef/teh/alef-maqsura unification, dialectal variants).
2. Word Error Rate (WER) via dynamic programming Levenshtein distance.
3. Character Error Rate (CER).
4. Concatenated Minimum-Permutation Word Error Rate (cpWER) for speaker-attributed ASR.
"""
import re
import itertools
from typing import List, Dict, Any, Tuple


def normalize_arabic_text(text: str) -> str:
    """
    Standard Arabic text normalization for ASR benchmarking:
    1. Strip all Arabic diacritics / tashkeel (fat-hah, dammah, kasrah, sukun, shaddah, tanween).
    2. Unify alef forms: إ, أ, آ, ٱ -> ا
    3. Unify teh marbuta and heh at word end: ة -> ه
    4. Unify alef maqsura and yaa: ى -> ي
    5. Strip all Latin and Arabic punctuation marks and extra whitespace.
    """
    if not text:
        return ""

    # Strip tashkeel
    tashkeel_regex = re.compile(r"[\u064B-\u0652\u0670\u0640]")
    text = tashkeel_regex.sub("", text)

    # Normalize alef variants
    text = re.sub(r"[إأآٱ]", "ا", text)

    # Normalize teh marbuta
    text = re.sub(r"ة", "ه", text)

    # Normalize alef maqsura
    text = re.sub(r"ى", "ي", text)

    # Strip punctuation
    text = re.sub(r"[،,.\?!؟;:\"\'\-\(\)\[\]«»_/\\]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance between two sequences of tokens.
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
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
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


def compute_wer(ref: str, hyp: str) -> float:
    """
    Compute Word Error Rate (WER) between normalized reference and hypothesis.
    WER = (S + D + I) / N_ref
    """
    ref_norm = normalize_arabic_text(ref)
    hyp_norm = normalize_arabic_text(hyp)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist, _, _, _ = levenshtein_distance(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)


def compute_cer(ref: str, hyp: str) -> float:
    """
    Compute Character Error Rate (CER) between normalized reference and hypothesis.
    """
    ref_norm = normalize_arabic_text(ref).replace(" ", "")
    hyp_norm = normalize_arabic_text(hyp).replace(" ", "")

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    dist, _, _, _ = levenshtein_distance(list(ref_norm), list(hyp_norm))
    return round(dist / len(ref_norm), 4)


def compute_cpwer(
    ref_segments: List[Dict[str, Any]],
    hyp_segments: List[Dict[str, Any]]
) -> float:
    """
    Compute Concatenated Minimum-Permutation Word Error Rate (cpWER).
    Concatenates all utterances for each speaker in reference and hypothesis,
    evaluates all permutations of speaker mappings, and selects the minimum WER.
    """
    # 1. Collect reference transcripts per speaker
    ref_by_spk: Dict[str, List[str]] = {}
    for seg in ref_segments:
        spk = seg.get("speaker", "SPEAKER_00")
        txt = normalize_arabic_text(seg.get("text", ""))
        if txt:
            ref_by_spk.setdefault(spk, []).append(txt)

    # 2. Collect hypothesis transcripts per speaker
    hyp_by_spk: Dict[str, List[str]] = {}
    for seg in hyp_segments:
        spk = seg.get("speaker", "SPEAKER_00")
        txt = normalize_arabic_text(seg.get("text", ""))
        if txt:
            hyp_by_spk.setdefault(spk, []).append(txt)

    ref_speakers = list(ref_by_spk.keys())
    hyp_speakers = list(hyp_by_spk.keys())

    if not ref_speakers:
        return 0.0 if not hyp_speakers else 1.0

    ref_total_words = sum(len(" ".join(texts).split()) for texts in ref_by_spk.values())
    if ref_total_words == 0:
        return 0.0

    # Ensure symmetric speaker lists for permutation
    max_k = max(len(ref_speakers), len(hyp_speakers))
    all_refs = ref_speakers + [f"DUMMY_REF_{i}" for i in range(max_k - len(ref_speakers))]
    all_hyps = hyp_speakers + [f"DUMMY_HYP_{i}" for i in range(max_k - len(hyp_speakers))]

    best_error_count = float("inf")

    # If small number of speakers (<= 6), search all permutations
    for perm in itertools.permutations(all_hyps, len(all_refs)):
        cur_errors = 0
        for r_spk, h_spk in zip(all_refs, perm):
            r_text = " ".join(ref_by_spk.get(r_spk, []))
            h_text = " ".join(hyp_by_spk.get(h_spk, []))
            r_words = r_text.split()
            h_words = h_text.split()
            dist, _, _, _ = levenshtein_distance(r_words, h_words)
            cur_errors += dist
        if cur_errors < best_error_count:
            best_error_count = cur_errors

    return round(best_error_count / ref_total_words, 4)
