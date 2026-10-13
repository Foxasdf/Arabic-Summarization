"""Chunking sweep (frozen_v1 step 3, pre-registered grid only).
Dev file: scratch_45s.wav vs backend/tests/eval/gold_dataset.py ref (contaminated note applies).
Grid: (18,2.0) (20,2.0) (25,2.5) (28,3.0). No open-ended search.
Run: .venv/Scripts/python.exe ../scripts/chunk_sweep.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.models.asr import get_asr_engine  # noqa: E402
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer  # noqa: E402
from tests.eval.gold_dataset import GOLD_TEST_SET  # noqa: E402

GRID = [(18.0, 2.0), (20.0, 2.0), (25.0, 2.5), (28.0, 3.0)]
AUDIO = ROOT / "scratch_45s.wav"
OUT = ROOT / "eval_results" / "frozen_v1" / "chunk_sweep.json"

def main():
    ref = GOLD_TEST_SET["scratch_45s" if "scratch_45s" in GOLD_TEST_SET else "sample_meeting_45s"]
    ref_text = ref["ground_truth_text"]
    norm = ArabicTextNormalizer()
    rows = []
    for chunk, ov in GRID:
        eng = get_asr_engine("cohere")
        eng.chunk_len = float(chunk)
        eng.overlap = float(ov)
        t0 = time.time()
        eng.load()
        load_s = time.time() - t0
        t1 = time.time()
        res = eng.transcribe(Path(AUDIO), on_progress=None)
        dt = time.time() - t1
        try:
            eng.unload()
        except Exception:
            pass
        hyp = res.get("text", "")
        m = calculate_wer_cer(ref_text, hyp, norm)
        dur = 45.0
        rows.append({"chunk": chunk, "overlap": ov, "load_s": round(load_s, 1),
                     "trans_s": round(dt, 1), "rtf": round(dt / dur, 3),
                     "norm_wer": m["norm_wer"], "norm_cer": m["norm_cer"],
                     "words": m["hyp_word_count"], "ref_words": m["ref_word_count"]})
        print(f"chunk={chunk} ov={ov} WER={m['norm_wer']:.4f} CER={m['norm_cer']:.4f} words={m['hyp_word_count']} t={dt:.1f}s", flush=True)
    OUT.write_text(json.dumps({"grid": rows, "dev": "scratch_45s",
                               "rule": "Change default only if best beats current (28,3.0) by >=0.02 norm_WER with n=1 file; else keep to avoid churn."},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
