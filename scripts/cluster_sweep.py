"""Clustering-threshold sweep (follow-up to M1 oracle gap).
Dev: scratch_45s.wav vs rebuilt gold segments. Thresholds: 0.60/0.65/0.7046/0.75/0.80.
Temporarily patches models config.yaml, invalidates pipeline cache per value,
restores file afterwards. Scores DER collar 0.25s + JER + count error.
Rule: adopt iff best beats current (0.7046) by >=0.02 DER with no JER regression >0.02.
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

CFG = ROOT / "models" / "pyannote" / "speaker-diarization-3.1" / "config.yaml"
AUDIO = ROOT / "scratch_45s.wav"
OUT = ROOT / "eval_results" / "frozen_v1" / "cluster_sweep.json"
GRID = [0.60, 0.65, 0.7045654963945799, 0.75, 0.80]

def main():
    import yaml
    from app.services import diarization as D
    ref = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_segments"]
    orig = CFG.read_text(encoding="utf-8")
    rows = []
    try:
        for th in GRID:
            cfg = yaml.safe_load(orig)
            cfg["params"]["clustering"]["threshold"] = float(th)
            CFG.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
            D.invalidate_pyannote_cache()
            turns = D.diarize(AUDIO)
            s = score_der(ref, turns)
            rows.append({"threshold": round(th, 4), "n_turns": len(turns), **s})
            print(f"thr={th:.4f} DER={s['der']:.4f} JER={s['jer']:.4f} nspk={s['n_hyp_spk']} turns={len(turns)}", flush=True)
    finally:
        CFG.write_text(orig, encoding="utf-8")
        D.invalidate_pyannote_cache()
    OUT.write_text(json.dumps({"dev": "scratch_45s", "collar": 0.25, "rows": rows,
                               "rule": "adopt iff best beats 0.7046 by >=0.02 DER with JER regression <=0.02"},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} (config.yaml restored)")

if __name__ == "__main__":
    main()
