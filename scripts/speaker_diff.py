"""Correct speaker-attribution diff: M1 auto turns vs transcription_20261011_133306.json.
1. Distinct-speaker overlap fraction (post-resolution).
2. Raw-vs-cleaned turn structure (did my merging/cleaning hurt?).
3. Region-level match/mismatch diff: where and how much misattribution.
4. Per-ref-speaker IoU heat (who absorbs whose turns).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M1 = ROOT / "eval_results" / "frozen_v1" / "m1_M1old"
ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))
ref_segments = ref["segments"]
hyp_turns = json.loads((M1 / "turns_auto.json").read_text(encoding="utf-8"))

def intervals(segments):
    return [(float(s["start"]), float(s["end"]), s["speaker"]) for s in segments]

def speech_total(iv):
    return sum(max(0.0, e - s) for s, e, _ in iv)

def distinct_overlap_frac(iv):
    # union of speech vs sum-of-per-speaker speech, where per-speaker speech is unioned within speaker
    per_spk = {}
    for s, e, spk in iv:
        per_spk.setdefault(spk, []).append((s, e))
    def union_len(xs):
        xs = sorted(xs); tot = 0.0; cs, ce = None, None
        for s, e in xs:
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                tot += ce - cs; cs, ce = s, e
        if cs is not None:
            tot += ce - cs
        return tot
    union_all = union_len([(s, e) for s, e, _ in iv])
    per_spk_sum = sum(union_len(v) for v in per_spk.values())
    if union_all <= 0:
        return 0.0, 0.0, 0.0
    dup = per_spk_sum - union_all
    return dup, union_all, dup / union_all

def regions(iv):
    pts = sorted(set([p for s, e, _ in iv for p in (s, e)]))
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 0.02:
            continue
        mid = (a + b) / 2
        spks = sorted(set(spk for s, e, spk in iv if s <= mid <= e))
        out.append((a, b, spks))
    return out

def iou(a_spks, b_spks):
    A, B = set(a_spks), set(b_spks)
    return len(A & B) / max(1, len(A | B))

def main():
    hyp_iv = intervals(hyp_turns)
    ref_iv = intervals(ref_segments)
    dup, union_all, frac = distinct_overlap_frac(hyp_iv)
    dup_r, union_r, frac_r = distinct_overlap_frac(ref_iv)
    print(f"HYP distinct-speaker dup: {dup:.1f}s of union {union_all:.1f}s = {frac*100:.1f}%")
    print(f"REF distinct-speaker dup: {dup_r:.1f}s of union {union_r:.1f}s = {frac_r*100:.1f}%")

    print(f"\nHyp turns: {len(hyp_turns)}  |  ref segments: {len(ref_segments)}")
    hc = Counter(t["speaker"] for t in hyp_turns)
    print("Hyp turn counts:", dict(hc))

    rr = regions(ref_iv); hr = regions(hyp_iv)
    match, mismatch, ref_only, hyp_only, both_silent = 0.0, 0.0, 0.0, 0.0, 0.0
    mm_by_pair = Counter(); mm_durs = []
    for a, b, rs in rr:
        for c, d, hs in hr:
            lo, hi = max(a, c), min(b, d)
            if hi - lo < 0.02:
                continue
            if not rs and not hs:
                both_silent += hi - lo
            elif not rs:
                hyp_only += hi - lo
            elif not hs:
                ref_only += hi - lo
            elif iou(rs, hs) == 1.0:
                match += hi - lo
            else:
                mismatch += hi - lo
                mm_by_pair[(tuple(rs), tuple(hs))] += hi - lo
                mm_durs.append((hi - lo, a, b, rs, hs))
    tot = match + mismatch + ref_only + hyp_only + both_silent
    print(f"\nRegion breakdown (total {tot:.1f}s):")
    print(f"  exact speaker-set match: {match:.1f}s ({100*match/tot:.1f}%)")
    print(f"  speaker-set MISMATCH   : {mismatch:.1f}s ({100*mismatch/tot:.1f}%)")
    print(f"  ref speech, hyp silent : {ref_only:.1f}s ({100*ref_only/tot:.1f}%)")
    print(f"  hyp speech, ref silent : {hyp_only:.1f}s ({100*hyp_only/tot:.1f}%)")
    print(f"  both silent            : {both_silent:.1f}s")
    print("\nTop mismatch ref->hyp speaker pairs (seconds):")
    for (rs, hs), sec in mm_by_pair.most_common(8):
        print(f"  {rs} -> {hs}: {sec:.1f}s")
    print("\nLongest mismatched regions:")
    for d, a, b, rs, hs in sorted(mm_durs, reverse=True)[:8]:
        print(f"  {a:6.1f}-{b:6.1f} ({d:4.1f}s) ref={rs} hyp={hs}")

    out = {"distinct_overlap_frac_hyp": round(frac, 3), "distinct_overlap_frac_ref": round(frac_r, 3),
           "region_match_s": round(match, 1), "region_mismatch_s": round(mismatch, 1),
           "ref_speech_hyp_silent_s": round(ref_only, 1), "hyp_speech_ref_silent_s": round(hyp_only, 1),
           "top_mismatch_pairs": {str(k): round(v, 1) for k, v in mm_by_pair.most_common(8)}}
    (ROOT / "eval_results" / "frozen_v1" / "speaker_diff.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote speaker_diff.json")

if __name__ == "__main__":
    main()