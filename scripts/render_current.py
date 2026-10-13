"""Deliverable: current transcript vs reference, side by side, with speaker agreement marks.
Reads eval_results/frozen_v1/m1/ (latest run)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from der_scorer import score_der  # noqa: E402

M1 = ROOT / "eval_results" / "frozen_v1" / "m1"
OUT_TXT = ROOT / "eval_results" / "frozen_v1" / "CURRENT_transcript.md"


def ts(x):
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


def main():
    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    hyp = json.loads((M1 / "segments_hyp.json").read_text(encoding="utf-8"))
    turns = json.loads((M1 / "turns_auto.json").read_text(encoding="utf-8"))
    s = score_der(ref, turns)

    lines = ["# Current transcript — 133306 (20261011_133306.m4a)", "",
             f"- DER {s['der']} (was 0.3496) | JER {s['jer']} (was 0.5046) | {s['n_hyp_spk']} speakers",
             f"- {len(hyp)} segments vs {len(ref)} reference",
             "- Verbatim: names/dishes are NOT auto-corrected", "",
             "Legend: segments marked OK had the correct speaker for most of their duration.", ""]
    agree = 0
    for seg in hyp:
        st, en, spk = seg["start"], seg["end"], seg["speaker"]
        mid = (st + en) / 2
        rs = sorted({r["speaker"] for r in ref if r["start"] <= mid <= r["end"]})
        ok = spk in rs
        agree += 1 if ok else 0
        mark = "" if ok else "  **WRONG SPEAKER**"
        lines.append(f"**[{ts(st)} - {ts(en)}] {spk}{mark}**")
        lines.append(seg["text"])
        lines.append("")
    lines.insert(7, f"- Speaker-correct segments: {agree}/{len(hyp)} ({100*agree/len(hyp):.0f}%)")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_TXT.name}: {agree}/{len(hyp)} segments speaker-correct")

    # Also emit a clean reference-only view of the worst stretches
    sys.path.insert(0, str(ROOT / "scripts"))
    ref_txt = ROOT / "eval_results" / "frozen_v1" / "REFERENCE_transcript.md"
    rl = ["# Reference transcript (ground truth)", ""]
    for seg in ref:
        rl.append(f"**[{ts(seg['start'])} - {ts(seg['end'])}] {seg['speaker']}:**")
        rl.append(seg["text"])
        rl.append("")
    ref_txt.write_text("\n".join(rl), encoding="utf-8")
    print(f"wrote {ref_txt.name}")


if __name__ == "__main__":
    main()