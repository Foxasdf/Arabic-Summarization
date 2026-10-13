"""VAD-gate sweep (follow-up to M1 17.6% deletions).
Dev: scratch_45s.wav vs rebuilt gold. Gate: -48/-42/-36 dBFS.
Scores norm WER + word count (deletion proxy).
Rule: adopt iff best beats current (-42) by >=0.02 norm_WER; else keep.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.models.asr import get_asr_engine  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer  # noqa: E402
from tests.eval.gold_dataset import GOLD_TEST_SET  # noqa: E402

GRID = [-48.0, -42.0, -36.0]
OUT = ROOT / "eval_results" / "frozen_v1" / "vad_sweep.json"

def main():
    ref = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_text"]
    norm = ArabicTextNormalizer()
    rows = []
    for thr in GRID:
        settings.VAD_MIN_RMS_DB = float(thr)
        eng = get_asr_engine("cohere")
        eng.load()
        res = eng.transcribe(ROOT / "scratch_45s.wav", on_progress=None)
        try:
            eng.unload()
        except Exception:
            pass
        m = calculate_wer_cer(ref, res.get("text", ""), norm)
        rows.append({"vad_db": thr, "norm_wer": m["norm_wer"], "norm_cer": m["norm_cer"],
                     "words": m["hyp_word_count"], "ref_words": m["ref_word_count"],
                     "dels": m["deletions"], "inss": m["insertions"]})
        print(f"vad={thr} WER={m['norm_wer']:.4f} CER={m['norm_cer']:.4f} words={m['hyp_word_count']} dels={m['deletions']}", flush=True)
    settings.VAD_MIN_RMS_DB = -42.0
    OUT.write_text(json.dumps({"dev": "scratch_45s", "rows": rows,
                               "rule": "adopt iff best beats -42 by >=0.02 norm_WER"},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} (setting restored to -42)")

if __name__ == "__main__":
    main()
