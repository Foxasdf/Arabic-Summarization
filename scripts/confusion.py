"""Confusion structure + time-drift check on 133306 raw turns.
Q1: Which voices get confused with which (confusion matrix, duration-weighted)?
Q2: Is the confusion time-localized (drift/identity swap) or uniform (embedding noise)?
Q3: How much of the error sits in short turns (<=3s) vs long turns?
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SPK_WAV = ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "turns_auto.json"
RAW = ROOT / "eval_results" / "frozen_v1" / "_raw_turns.json"
OUT = ROOT / "eval_results" / "frozen_v1" / "confusion.json"


def rename(turns):
    m, out = {}, []
    for t in sorted(turns, key=lambda x: x["start"]):
        k = t["speaker"]
        if k not in m:
            m[k] = f"SPEAKER_{len(m):02d}"
        out.append({"speaker": m[k], "start": round(float(t["start"]), 2), "end": round(float(t["end"]), 2)})
    return out


def main():
    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    hyp = rename(json.loads(RAW.read_text(encoding="utf-8")))

    # optimal 1-1 mapping ref->hyp by total overlap (pure DER mapping)
    spks = sorted(set(t["speaker"] for t in hyp))
    best, bestmap = -1, None
    import itertools
    for perm in itertools.permutations(spks, 3):
        tot = 0.0
        for r in ref:
            ov = sum(max(0.0, min(r["end"], h["end"]) - max(r["start"], h["start"]))
                     for h in hyp if h["speaker"] == perm[int(r["speaker"][-2])])
            tot += ov
        if tot > best:
            best, bestmap = tot, perm
    print("best ref->hyp mapping:", {f"SPEAKER_0{i}": bestmap[i] for i in range(3)})

    pts = sorted(set([p for t in ref for p in (t["start"], t["end"])] +
                     [p for t in hyp for p in (t["start"], t["end"])]))
    conf = defaultdict(float)
    dur_bucket = defaultdict(float)
    time_bins = defaultdict(float)
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 0.02:
            continue
        mid = (a + b) / 2
        rs = [r["speaker"] for r in ref if r["start"] <= mid <= r["end"]]
        hs = [h["speaker"] for h in hyp if h["start"] <= mid <= h["end"]]
        if not rs or not hs:
            continue
        r0 = rs[0]
        h0 = bestmap[int(r0[-2])]
        ok = h0 in hs
        if not ok:
            conf[(r0, tuple(sorted(hs)))] += b - a
            dur_bucket["<=3s" if (b - a) <= 3 else ">3s"] += b - a
            time_bins[f"{int(mid//60):02d}m"] += b - a

    print("\nconfusions (ref -> hyp present):")
    for (r, hs), sec in sorted(conf.items(), key=lambda x: -x[1]):
        print(f"  {r} -> {hs}: {sec:.1f}s")

    tot_err = sum(conf.values())
    print(f"\ntotal confused: {tot_err:.1f}s")
    for k, v in dur_bucket.items():
        print(f"  {k}: {v:.1f}s ({100*v/tot_err:.0f}%)")
    print("\nconfusion by minute:")
    for k in sorted(time_bins):
        print(f"  {k}: {time_bins[k]:.1f}s")

    OUT.write_text(json.dumps({
        "mapping": {f"SPEAKER_0{i}": bestmap[i] for i in range(3)},
        "confusions": {f"{r}->{hs}": round(v, 1) for (r, hs), v in sorted(conf.items(), key=lambda x: -x[1])},
        "by_duration": {k: round(v, 1) for k, v in dur_bucket.items()},
        "by_minute": {k: round(v, 1) for k, v in sorted(time_bins.items())},
        "total_confused_s": round(tot_err, 1)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()