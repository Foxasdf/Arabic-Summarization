"""Isolation check: does the cleaning change regress scratch_45s?
Runs pyannote ONCE, caches raw turns, then compares OLD vs NEW cleaning
on identical raw input -> isolates the cleaning change from pyannote run variance.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402

CACHE = ROOT / "eval_results" / "frozen_v1" / "_raw_turns_45s.json"
OUT = ROOT / "eval_results" / "frozen_v1" / "cleaning_isolation_45s.json"


def rename(turns):
    m, out = {}, []
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in m:
            m[t["speaker"]] = f"SPEAKER_{len(m):02d}"
        out.append({"speaker": m[t["speaker"]], "start": round(float(t["start"]), 2),
                    "end": round(float(t["end"]), 2)})
    return out


def main():
    from app.services.alignment import clean_diarization_turns
    from tests.eval.gold_dataset import GOLD_TEST_SET

    if CACHE.exists():
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        from app.services.diarization import _diarize_pyannote_local
        raw = _diarize_pyannote_local(ROOT / "scratch_45s.wav", None)
        CACHE.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    ref = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_segments"]

    raw_spk = sorted({t["speaker"] for t in raw})
    print(f"raw pyannote speakers: {len(raw_spk)}")

    s_raw = score_der(ref, rename(raw))
    print(f"raw (no cleaning) : DER={s_raw['der']:.4f} JER={s_raw['jer']:.4f} nspk={s_raw['n_hyp_spk']}")

    old = rename(clean_diarization_turns([dict(t) for t in raw], min_dur=0.25, merge_gap=0.5, punch_min=0.75))
    s_old = score_der(ref, old)
    print(f"OLD cleaning      : DER={s_old['der']:.4f} JER={s_old['jer']:.4f} nspk={s_old['n_hyp_spk']} turns={len(old)}")

    new = rename(clean_diarization_turns([dict(t) for t in raw], min_dur=0.15, merge_gap=0.0, punch_min=0.75))
    s_new = score_der(ref, new)
    print(f"NEW cleaning      : DER={s_new['der']:.4f} JER={s_new['jer']:.4f} nspk={s_new['n_hyp_spk']} turns={len(new)}")

    delta = s_new["der"] - s_old["der"]
    verdict = "REGRESSION" if delta > 0.02 else ("IMPROVEMENT" if delta < -0.02 else "neutral")
    print(f"\ncleaning delta: {delta:+.4f} -> {verdict}")
    OUT.write_text(json.dumps({"raw": s_raw, "old": s_old, "new": s_new,
                               "delta": round(delta, 4), "verdict": verdict},
                              ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()