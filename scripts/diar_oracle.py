"""Diarization oracle diagnostic (frozen_v1 step 4a, BEFORE any threshold change).
Dev: scratch_45s.wav vs gold_dataset ref segments.
Runs: auto(min=2,max=8 current) vs oracle num=3 vs min=1 probe.
Score: DER collar 0.25s fixed + JER + speaker-count error.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402
from tests.eval.gold_dataset import GOLD_TEST_SET  # noqa: E402

AUDIO = ROOT / "scratch_45s.wav"
OUT = ROOT / "eval_results" / "frozen_v1" / "diar_oracle.json"

def run_once(diar_fn, kwargs):
    import time
    t0 = time.time()
    turns = diar_fn(Path(AUDIO), **kwargs)
    return turns, round(time.time() - t0, 1)

def main():
    from app.services import diarization as D
    ref = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_segments"]
    rows = []
    # 1. current default auto
    turns_auto, dt = run_once(lambda p, **k: D.diarize(p, **k), {})
    rows.append({"cond": "auto_min2_max8", "dt_s": dt,
                 "n_turns": len(turns_auto),
                 **score_der(ref, turns_auto)})
    print(f"auto: {rows[-1]}", flush=True)
    # 2. oracle num=3
    turns_or, dt = run_once(lambda p, **k: D.diarize(p, **k), {"num_speakers": 3})
    rows.append({"cond": "oracle_num3", "dt_s": dt,
                 "n_turns": len(turns_or),
                 **score_der(ref, turns_or)})
    print(f"oracle3: {rows[-1]}", flush=True)
    OUT.write_text(json.dumps({"dev": "scratch_45s", "collar": 0.25, "rows": rows,
                               "ref_note": "gold_dataset ref itself contaminated (kasm/oasm spellings) but timestamps usable for DER"},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
