"""DER scorer, collar 0.25s fixed. Frame-based (10ms), optimal speaker mapping by permutation.
Usage: score_der(ref_rttm, hyp_turns, collar=0.25)
"""
import itertools
from pathlib import Path

FRAME = 0.01

def load_rttm(path):
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        segs.append({"start": float(p[3]), "end": float(p[3]) + float(p[4]), "speaker": p[7]})
    return segs

def turns_to_frames(turns, end):
    n = max(1, int(round(end / FRAME)))
    lab = [None] * n
    for t in turns:
        s = max(0, int(round(float(t["start"]) / FRAME)))
        e = min(n, int(round(float(t["end"]) / FRAME)))
        for i in range(s, e):
            lab[i] = t["speaker"]
    return lab

def score_der(ref_turns, hyp_turns, collar=0.25, end=None):
    if end is None:
        end = max([float(t["end"]) for t in ref_turns + hyp_turns] or [0])
    ref_f = turns_to_frames(ref_turns, end)
    hyp_f = turns_to_frames(hyp_turns, end)
    n = len(ref_f)
    # collar mask: drop frames within collar of any ref boundary
    bounds = sorted(set([float(t["start"]) for t in ref_turns] + [float(t["end"]) for t in ref_turns]))
    excl = [False] * n
    for b in bounds:
        s = max(0, int(round((b - collar) / FRAME)))
        e = min(n, int(round((b + collar) / FRAME)))
        for i in range(s, e):
            excl[i] = True
    ref_spk = sorted(set(t["speaker"] for t in ref_turns))
    hyp_spk = sorted(set(t["speaker"] for t in hyp_turns))
    k = max(len(ref_spk), len(hyp_spk))
    ref_ext = ref_spk + [f"__R{i}" for i in range(k - len(ref_spk))]
    hyp_ext = hyp_spk + [f"__H{i}" for i in range(k - len(hyp_spk))]
    best = None
    for perm in itertools.permutations(hyp_ext, len(ref_ext)):
        mapping = dict(zip(ref_ext, perm))
        miss = fa = conf = tot = 0
        for i in range(n):
            if excl[i]:
                continue
            r, h = ref_f[i], hyp_f[i]
            if r is None and h is None:
                continue
            if r is None:
                fa += 1
                continue
            tot += 1
            if h is None:
                miss += 1
            elif mapping.get(r) != h:
                conf += 1
        der = (miss + fa + conf) / max(1, tot)
        if best is None or der < best[0]:
            best = (der, miss * FRAME, fa * FRAME, conf * FRAME, tot * FRAME)
    der, miss_s, fa_s, conf_s, tot_s = best
    jer_note = "JER computed as 1 - mean per-ref-speaker IoU (collar applied)"
    # JER
    jers = []
    for r in ref_spk:
        # find hyp speaker with max overlap under mapping-free IoU
        best_iou = 0.0
        for h in hyp_spk:
            inter = union = 0
            for i in range(n):
                if excl[i]:
                    continue
                ri = (ref_f[i] == r)
                hi = (hyp_f[i] == h)
                if ri and hi:
                    inter += 1
                if ri or hi:
                    union += 1
            iou = inter / max(1, union)
            best_iou = max(best_iou, iou)
        jers.append(1.0 - best_iou)
    jer = sum(jers) / max(1, len(jers))
    return {"der": round(der, 4), "miss_s": round(miss_s, 2), "fa_s": round(fa_s, 2),
            "conf_s": round(conf_s, 2), "scored_speech_s": round(tot_s, 2),
            "jer": round(jer, 4), "n_ref_spk": len(ref_spk), "n_hyp_spk": len(hyp_spk),
            "spk_count_err": abs(len(ref_spk) - len(hyp_spk)), "collar": collar, "jer_note": jer_note}
