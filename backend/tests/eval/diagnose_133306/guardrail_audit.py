"""Guardrail audit: which rules actually fire on this recording, and what they cost.

Deliberately narrow: it evaluates the *production* HALLUCINATION_PATTERNS list
against (a) the captured per-slice raw strings and (b) the joined hypothesis
text, plus a probe battery that demonstrates the failure mode on realistic
Levantine sentences. It never writes to backend/app/**.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from . import config as C
from . import patches

PROBES = [
    ("هي صار خمس دقائق تسجيل", "recording (end-of-session remark)"),
    ("كنا فاتحين تسجيل عم نحكي عن الاكل", "recording (topic statement)"),
    ("التسجيل انقطع", "the recording cut off"),
    ("تسجيل الصوت", "audio recording"),
    ("كانت معي تسجيل للاجتماع كلمة كلمة", "I had a recording of the meeting"),
    ("شو رأيك نعمل تسجيل تاني بعدين", "let's record again later"),
    ("الترجمة من قبل أحمد", "subtitle credit (true positive)"),
    ("شكراً لكم على المتابعة", "youtube outro (true positive)"),
]


def _compile():
    from app.models.asr.guardrails import HALLUCINATION_PATTERNS

    return [(p, re.compile(p, re.IGNORECASE)) for p in HALLUCINATION_PATTERNS]


def audit(captured: Dict[str, Any], joined_text: str) -> Dict[str, Any]:
    from app.models.asr.guardrails import clean_and_guard_asr_text

    pats = _compile()
    fires: List[Dict[str, Any]] = []
    for s in captured.get("slices", []):
        raw = s.get("raw") or ""
        if not raw:
            continue
        for p, rx in pats:
            m = rx.search(raw)
            if not m:
                continue
            removed = re.sub(p, "", raw, flags=re.IGNORECASE).strip()
            fires.append({
                "slice": s["idx"],
                "start": round(s["start"], 2),
                "end": round(s["start"] + s["dur"], 2),
                "pattern": p,
                "matched": m.group(0)[:120],
                "raw_words": len(raw.split()),
                "after_words": len(removed.split()),
                "words_lost": len(raw.split()) - len(removed.split()),
                "wiped_entirely": len(removed.split()) <= 2,
            })
            break

    joined_hits = [p for p, rx in pats if rx.search(joined_text)]

    probes = []
    for text, why in PROBES:
        out = clean_and_guard_asr_text(text, dur_sec=10.0)
        probes.append({
            "in": text,
            "why": why,
            "out": out,
            "in_words": len(text.split()),
            "out_words": len(out.split()),
            "deleted": len(text.split()) - len(out.split()),
        })

    total_lost = sum(f["words_lost"] for f in fires)
    return {
        "slice_fires": fires,
        "slice_words_lost": total_lost,
        "joined_text_pattern_hits": joined_hits,
        "probes": probes,
        "probes_deleted": sum(1 for p in probes if p["deleted"] > 0),
        "probes_total": len(probes),
    }


def run(joined_text: str) -> Dict[str, Any]:
    captured = C.read_json(C.ARTIFACT_DIR / "captured_asr_uniform.json")
    out = audit(captured, joined_text)
    C.write_json(C.OUT_DIR / "guardrail_audit.json", out)
    return out
