"""Isolate the cleaning-stage regression on 133306.
Raw pyannote turns -> DER 0.3085 (101 turns)
After _normalize_turns -> DER 0.3496 (81 turns)
So cleaning LOSES 20 turns and degrades DER. Sweep cleaning variants.
Reference: transcription_20261011_133306.json. Collar 0.25s.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402

SPK_WAV = ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "speaker.16k.wav"
CACHE = ROOT / "eval_results" / "frozen_v1" / "_raw_turns.json"
OUT = ROOT / "eval_results" / "frozen_v1" / "cleaning_sweep.json"


def get_raw():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    import numpy as np
    import torch
    from app.services.diarization import _diarize_pyannote_local
    turns = _diarize_pyannote_local(SPK_WAV, None)
    CACHE.write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
    return turns


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
    from app.services.diarization import _normalize_turns
    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    raw = get_raw()
    rows = []

    # 0. raw, no cleaning at all (just rename)
    r = rename(raw)
    s = score_der(ref, r)
    rows.append({"exp": "raw_no_cleaning", "n_turns": len(r), **s})
    print(f"raw_no_cleaning: DER={s['der']:.4f} JER={s['jer']:.4f} nspk={s['n_hyp_spk']} turns={len(r)}", flush=True)

    # 1. current production path
    c = _normalize_turns([dict(t) for t in raw])
    s = score_der(ref, c)
    rows.append({"exp": "current_normalize_turns", "n_turns": len(c), **s})
    print(f"current_normalize_turns: DER={s['der']:.4f} JER={s['jer']:.4f} nspk={s['n_hyp_spk']} turns={len(c)}", flush=True)

    # 2. sweep min_dur / merge_gap / punch_min
    for min_dur in (0.0, 0.15, 0.25):
        for merge_gap in (0.0, 0.3, 0.5):
            for punch in (0.0, 0.5, 0.75):
                cc = rename(clean_diarization_turns([dict(t) for t in raw], min_dur=min_dur,
                                                   merge_gap=merge_gap, punch_min=punch))
                s = score_der(ref, cc)
                rows.append({"exp": f"clean_md{min_dur}_mg{merge_gap}_pm{punch}", "n_turns": len(cc), **s})

    rows.sort(key=lambda x: x["der"])
    print("\nTop 12 by DER:")
    for r_ in rows[:12]:
        print(f"  {r_['exp']:38s} DER={r_['der']:.4f} JER={r_['jer']:.4f} nspk={r_['n_hyp_spk']} turns={r_['n_turns']}")

    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()