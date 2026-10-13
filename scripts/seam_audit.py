"""Seam-dedup audit: count + inspect overlap-dedup removals during Cohere transcribe.
Part A (observe): run scratch_45s with log capture of 'Deduplicated Nk' events.
Part B (exact-only variant): monkeypatch _find_word_overlap to skip fuzzy branch,
re-run, score vs rebuilt gold. Adopt iff >=0.02 norm_WER gain.
"""
import io
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.models.asr import get_asr_engine  # noqa: E402
from app.models.asr.base import BaseASREngine  # noqa: E402
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer  # noqa: E402
from tests.eval.gold_dataset import GOLD_TEST_SET  # noqa: E402

OUT = ROOT / "eval_results" / "frozen_v1" / "seam_audit.json"

class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []
    def emit(self, record):
        msg = record.getMessage()
        if "Deduplicated" in msg and "boundary words" in msg:
            self.events.append(msg)

def run_once():
    cap = Capture()
    logging.getLogger("app.models.asr.base").addHandler(cap)
    try:
        eng = get_asr_engine("cohere")
        eng.load()
        res = eng.transcribe(ROOT / "scratch_45s.wav", on_progress=None)
        try:
            eng.unload()
        except Exception:
            pass
    finally:
        logging.getLogger("app.models.asr.base").removeHandler(cap)
    return res, cap.events

def main():
    ref = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_text"]
    norm = ArabicTextNormalizer()
    # A: current (exact+fuzzy)
    res_a, events_a = run_once()
    m_a = calculate_wer_cer(ref, res_a.get("text", ""), norm)
    removed_a = sum(int(e.split("Deduplicated ")[1].split(" boundary")[0]) for e in events_a)
    print(f"A current: WER={m_a['norm_wer']:.4f} words={m_a['hyp_word_count']} events={len(events_a)} removed~{removed_a}", flush=True)
    for e in events_a[:10]:
        print(f"  {e}", flush=True)
    # B: exact-only (drop fuzzy branch)
    orig = BaseASREngine._find_word_overlap
    @staticmethod
    def exact_only(tail_words, head_words):
        import re
        def _norm_w(w):
            w = re.sub(r"[،,.\?!؟;:\"\'\-\(\)\[\]]", "", w).strip()
            w = re.sub(r"[إأآٱ]", "ا", w)
            w = re.sub(r"ة", "ه", w)
            w = re.sub(r"ى", "ي", w)
            return w
        tail_norm = [_norm_w(w) for w in tail_words]
        head_norm = [_norm_w(w) for w in head_words]
        max_k = min(len(tail_norm), len(head_norm))
        for k in range(max_k, 0, -1):
            if tail_norm[-k:] == head_norm[:k]:
                return k
        return 0
    BaseASREngine._find_word_overlap = exact_only
    try:
        res_b, events_b = run_once()
    finally:
        BaseASREngine._find_word_overlap = orig
    m_b = calculate_wer_cer(ref, res_b.get("text", ""), norm)
    removed_b = sum(int(e.split("Deduplicated ")[1].split(" boundary")[0]) for e in events_b)
    print(f"B exact-only: WER={m_b['norm_wer']:.4f} words={m_b['hyp_word_count']} events={len(events_b)} removed~{removed_b}", flush=True)
    gain = m_a["norm_wer"] - m_b["norm_wer"]
    decision = "ADOPT exact-only" if gain >= 0.02 else "KEEP current (no >=0.02 gain)"
    print(decision, flush=True)
    OUT.write_text(json.dumps({"A": {"wer": m_a["norm_wer"], "words": m_a["hyp_word_count"], "events": events_a},
                               "B": {"wer": m_b["norm_wer"], "words": m_b["hyp_word_count"], "events": events_b},
                               "gain": round(gain, 4), "decision": decision},
                              ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
