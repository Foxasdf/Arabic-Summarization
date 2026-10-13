import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
hyp = json.loads((ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "turns_auto.json").read_text(encoding="utf-8"))

def s(x):
    return f"{int(x // 60):02d}:{x % 60:04.1f}"

# side-by-side, aligned on time, sorted by start
ref_sorted = sorted(ref, key=lambda r: r["start"])
hyp_sorted = sorted(hyp, key=lambda h: h["start"])

rows = []
i = j = 0
while i < len(ref_sorted) or j < len(hyp_sorted):
    r = ref_sorted[i] if i < len(ref_sorted) else None
    h = hyp_sorted[j] if j < len(hyp_sorted) else None
    if r and h and r["start"] < h["start"] - 0.6:
        i += 1
        continue
    if h and r and h["start"] < r["start"] - 0.6:
        j += 1
        continue
    if r and h:
        rows.append((r, h))
        i += 1
        j += 1
    elif r:
        i += 1
    elif h:
        j += 1

print(f"{'TIME':>12} | {'REF spk dur':>13} | {'REF text':<34} | {'HYP spk dur':>13}")
print("-" * 110)
for r, h in rows:
    rt = f"{s(r['start'])}-{s(r['end'])}"
    rk = f"{r['speaker'][-2:]} {r['end']-r['start']:.1f}s"
    ht = f"{s(h['start'])}-{s(h['end'])}"
    hk = f"{h['speaker'][-2:]} {h['end']-h['start']:.1f}s"
    flag = "" if r["speaker"] == h["speaker"] else "   <-- SPK MISMATCH"
    print(f"{rt:>12} | {rk:>13} | {r['text'][:34]:<34} | {ht:>12} {hk:>13}{flag}")