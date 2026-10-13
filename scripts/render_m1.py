import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
segs = json.loads((ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "segments_hyp.json").read_text(encoding="utf-8"))

def ts(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"

lines = ["# Honest transcript — M1 run (matches current code)", "",
         f"- Segments: {len(segs)} (ref 105) | WER norm 0.5022 | DER 0.3496 | 3 speakers",
         "- Verbatim output: names/dishes NOT auto-corrected", ""]
for s in segs:
    lines.append(f"**[{ts(s['start'])} - {ts(s['end'])}] {s['speaker']}:**")
    lines.append(s["text"])
    lines.append("")
(ROOT / "eval_results" / "frozen_v1" / "M1_honest_transcript.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote M1_honest_transcript.md with", len(segs), "segments")
