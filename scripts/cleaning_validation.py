"""Validate new cleaning defaults on BOTH 133306 and dev scratch_45s (no regression allowed).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402

OUT = ROOT / "eval_results" / "frozen_v1" / "cleaning_validation.json"


def rename(turns):
    m, out = {}, []
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in m:
            m[t["speaker"]] = f"SPEAKER_{len(m):02d}"
        out.append({"speaker": m[t["speaker"]], "start": round(float(t["start"]), 2),
                    "end": round(float(t["end"]), 2)})
    return out


def score(ref, turns):
    return score_der(ref, rename(turns))


def main():
    from app.services.alignment import clean_diarization_turns
    from app.services.diarization import _normalize_turns
    from tests.eval.gold_dataset import GOLD_TEST_SET

    res = {}

    # --- 133306 (user-sanctioned reference for this track) ---
    ref133 = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    raw = json.loads((ROOT / "eval_results" / "frozen_v1" / "_raw_turns.json").read_text(encoding="utf-8"))
    new = _normalize_turns([dict(t) for t in raw])
    old = clean_diarization_turns([dict(t) for t in raw], min_dur=0.25, merge_gap=0.5, punch_min=0.75)
    s_new = score(ref133, new)
    s_old = score(ref133, rename(old))
    s_raw = score(ref133, rename(raw))
    res["133306"] = {"raw": s_raw, "old_defaults": s_old, "new_defaults": s_new,
                     "der_gain": round(s_old["der"] - s_new["der"], 4)}
    print(f"133306 raw      : DER={s_raw['der']:.4f} JER={s_raw['jer']:.4f} nspk={s_raw['n_hyp_spk']} turns={s_raw and len(raw)}")
    print(f"133306 old      : DER={s_old['der']:.4f} JER={s_old['jer']:.4f} nspk={s_old['n_hyp_spk']}")
    print(f"133306 NEW      : DER={s_new['der']:.4f} JER={s_new['jer']:.4f} nspk={s_new['n_hyp_spk']} turns={len(new)}")
    print(f"133306 DER gain : {s_old['der'] - s_new['der']:+.4f}")

    # --- dev scratch_45s (must not regress) ---
    from app.services import diarization as D
    ref45 = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_segments"]
    t45 = D.diarize(ROOT / "scratch_45s.wav")
    s45 = score(ref45, t45)
    res["scratch_45s_new"] = s45
    print(f"\nscratch_45s NEW : DER={s45['der']:.4f} JER={s45['jer']:.4f} nspk={s45['n_hyp_spk']} turns={len(t45)}")
    print(f"scratch_45s OLD : DER=0.0615 JER=0.0781 nspk=3  (baseline from cluster_sweep with 0.7046)")
    print(f"scratch_45s regress? {'YES' if s45['der'] > 0.0615 + 0.02 else 'no'}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()