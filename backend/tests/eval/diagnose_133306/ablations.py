"""Ablation harness: diarization ladder, builder isolation, timing arms, per-turn.

Each arm returns scored results so ``report.py`` can rank levers. Gates are
evaluated with a single direction (WER lower is better, agreement higher is
better) and the signed delta is printed next to the gate so the inequality can
be read off directly.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as C
from . import pipeline_runner as PR
from . import scoring
from . import scoring_spk as SS


# ---------------------------------------------------------------------------
# Diarization ladder (no ASR needed)
# ---------------------------------------------------------------------------
def diar_ladder(ref_segments: List[Dict[str, Any]], raw_turns: List[Dict[str, Any]],
                audio_duration: float) -> List[Dict[str, Any]]:
    """Score progressively-cleaned diarization so each stage's cost is visible.

    ``raw`` here means "Pyannote output as the pipeline receives it, after
    overlap resolution + renaming but before the conversational cleaner". The
    scorer also resolves overlaps to one label, so it would otherwise hide
    Pyannote's 37 overlapping turns (439 s of speaker-time in a 398 s file).
    """
    import app.services.diarization as D
    from app.services.alignment import clean_diarization_turns

    def stamped(turns):
        m, out = {}, []
        for t in sorted(turns, key=lambda x: (float(x["start"]), float(x["end"]))):
            spk = str(t["speaker"])
            if spk not in m:
                m[spk] = f"SPEAKER_{len(m):02d}"
            out.append({"speaker": m[spk], "start": round(float(t["start"]), 2), "end": round(float(t["end"]), 2)})
        return out

    def cleaned(merge_gap: float, min_dur: float = 0.15):
        return clean_diarization_turns(list(raw_turns), min_dur=min_dur, merge_gap=merge_gap)

    rows = []
    rungs = [
        ("R0 raw-as-received", "Pyannote turns resolved for overlap + renamed, no conversational cleaner",
         lambda: _normalize_only(raw_turns)),
        ("R1 clean(0.0,0.15)", "clean_diarization_turns(merge_gap=0.0)",
         lambda: stamped(cleaned(0.0))),
        ("R2 clean(0.5,0.15)", "clean_diarization_turns(merge_gap=0.5)  <- PRODUCTION",
         lambda: stamped(cleaned(0.5))),
        ("R3 clean(0.5,0.25)", "legacy min_dur=0.25 with merge_gap=0.5",
         lambda: stamped(cleaned(0.5, 0.25))),
        ("R4 clean(0.0,0.15)@1.0s", "merge_gap=0.0 then join same-speaker gaps <= 1.0 s",
         lambda: stamped(_join(cleaned(0.0), 1.0))),
        ("R5 clean(0.5,0.15)@1.0s", "merge_gap=0.5 then join same-speaker gaps <= 1.0 s",
         lambda: stamped(_join(cleaned(0.5), 1.0))),
    ]
    for name, desc, fn in rungs:
        try:
            turns = fn()
        except Exception as e:  # pragma: no cover
            rows.append({"rung": name, "desc": desc, "error": repr(e)})
            continue
        sc = SS.score_diarization(ref_segments, turns, audio_duration)
        ts = sorted(turns, key=lambda t: float(t["start"]))
        gaps = [round(float(b["start"]) - float(a["end"]), 2) for a, b in zip(ts, ts[1:])]
        same_gaps = [g for a, b, g in zip(ts, ts[1:], gaps) if a["speaker"] == b["speaker"] and g > 0.01]
        diff_gaps = [g for a, b, g in zip(ts, ts[1:], gaps) if a["speaker"] != b["speaker"] and g > 0.01]
        rows.append({
            "rung": name,
            "desc": desc,
            "n_turns": len(turns),
            "n_speakers": len({t["speaker"] for t in turns}),
            "der_ref_tiles": sc["stages"]["ref_tiles"]["der"],
            "der_speech_compare": sc["stages"]["speech_compare"]["der"],
            "der_strict_collar": sc["stages"]["strict_collar25"]["der"],
            "conf_s": sc["stages"]["ref_tiles"]["conf_s"],
            "miss_s": sc["stages"]["ref_tiles"]["miss_s"],
            "fa_s": sc["stages"]["ref_tiles"]["fa_s"],
            "jer": sc["jer"],
            "same_spk_gaps": len(same_gaps),
            "same_spk_gap_max": max(same_gaps) if same_gaps else 0.0,
            "same_spk_gap_median": sorted(same_gaps)[len(same_gaps) // 2] if same_gaps else 0.0,
            "diff_spk_gaps": len(diff_gaps),
        })
    return rows


def _normalize_only(raw_turns) -> List[Dict[str, Any]]:
    """Overlap-resolve + rename without the conversational cleaner."""
    m, out = {}, []
    for t in sorted(raw_turns, key=lambda x: (float(x["start"]), float(x["end"]))):
        spk = str(t["speaker"])
        if spk not in m:
            m[spk] = f"SPEAKER_{len(m):02d}"
        out.append({"speaker": m[spk], "start": round(float(t["start"]), 2), "end": round(float(t["end"]), 2)})
    return SS.resolve_midpoints(out)


def _join(turns, max_gap: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in sorted(turns, key=lambda x: float(x["start"])):
        if out and out[-1]["speaker"] == t["speaker"] and 0.0 <= (float(t["start"]) - out[-1]["end"]) <= max_gap:
            out[-1]["end"] = max(out[-1]["end"], float(t["end"]))
        else:
            out.append(dict(t))
    return out


# ---------------------------------------------------------------------------
# Builder isolation (item 3): ref text + ref timings through the production builder
# ---------------------------------------------------------------------------
def builder_isolation(ref_segments: List[Dict[str, Any]], turns: List[Dict[str, Any]],
                      audio_duration: float) -> Dict[str, Any]:
    """Item 3: how much segmentation loss comes from the utterance builder alone?

    All variants are fed reference words on reference timing, so ASR error is
    removed and only the builder's grouping/timing decisions remain.
    """
    from app.services import alignment as A

    turns_clean = A.clean_diarization_turns(turns)

    # (a) neutral upper bound: segments *are* the reference segments
    oracle = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"]}
              for s in ref_segments]

    def words_for_span(full_segment: bool) -> List[Dict[str, Any]]:
        words = []
        for s in ref_segments:
            toks = (s.get("text") or "").split()
            if not toks:
                continue
            step = (s["end"] - s["start"]) / len(toks)
            for j, tok in enumerate(toks):
                if full_segment:
                    words.append({"word": tok, "start": s["start"], "end": s["end"]})
                else:
                    words.append({"word": tok, "start": round(s["start"] + j * step, 2),
                                  "end": round(s["start"] + (j + 1) * step, 2)})
        return words

    def run_builder(words, max_pause: float):
        """Exact production sequence with a configurable pause threshold."""
        assigned = A.assign_words_to_speakers([dict(w) for w in words], turns_clean)
        smoothed = A.smooth_speaker_assignments(assigned, turns_clean)
        utt = A.build_utterances_from_words(smoothed, max_pause_sec=max_pause)
        return A.merge_consecutive_speaker_segments(utt)

    def score(segs):
        dj = SS.score_diarization(ref_segments, segs, audio_duration)
        ag = SS.segment_agreement(ref_segments, segs)
        ts = sorted(segs, key=lambda x: float(x["start"]))
        diff_gaps = [round(float(b["start"]) - float(a["end"]), 2)
                     for a, b in zip(ts, ts[1:]) if a["speaker"] != b["speaker"]]
        return {
            "n_segments": len(segs),
            "der_ref_tiles": dj["stages"]["ref_tiles"]["der"],
            "der_speech_compare": dj["stages"]["speech_compare"]["der"],
            "der_strict_collar": dj["stages"]["strict_collar25"]["der"],
            "conf_s": dj["stages"]["ref_tiles"]["conf_s"],
            "miss_s": dj["stages"]["ref_tiles"]["miss_s"],
            "fa_s": dj["stages"]["ref_tiles"]["fa_s"],
            "pct_ok": ag["pct_ok"],
            "segments_checked": ag["segments_checked"],
            "speaker_change_gaps": len(diff_gaps),
            "speaker_change_gap_median": (sorted(diff_gaps)[len(diff_gaps) // 2] if diff_gaps else None),
            "segments": segs,
        }

    out: Dict[str, Any] = {
        "oracle_ref_text": {
            "description": "reference text on reference timeline (upper bound; no ASR, no builder)",
            "n_segments": len(oracle),
            "pct_ok": SS.segment_agreement(ref_segments, oracle)["pct_ok"],
        },
        "builder_words_inside_segment": {
            "description": "reference words spread inside their reference segment, then production "
                           "builder with max_pause_sec=1.0",
            **{k: v for k, v in score(run_builder(words_for_span(False), 1.0)).items() if k != "segments"},
        },
        "builder_words_full_segment": {
            "description": "each reference word spans its whole reference segment (perfect word seals), "
                           "then production builder",
            **{k: v for k, v in score(run_builder(words_for_span(True), 1.0)).items() if k != "segments"},
        },
    }

    # (e) pause-threshold grid on the same words -- the builder knob itself
    words = words_for_span(False)
    grid = []
    for pause in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5):
        r = score(run_builder(words, pause))
        grid.append({"max_pause_sec": pause, **{k: v for k, v in r.items() if k != "segments"}})
    out["pause_grid"] = grid

    # (f) diagnostic: what pause threshold would reproduce the reference count?
    out["gcd_check"] = _gcd_check(turns_clean, ref_segments)
    return out


def _gcd_check(turns: List[Dict[str, Any]], ref_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How close is the first speaker-change gap to an exact multiple of the current threshold?"""
    ts = sorted(turns, key=lambda t: float(t["start"]))
    diff_gaps = sorted(round(float(b["start"]) - float(a["end"]), 2)
                       for a, b in zip(ts, ts[1:]) if a["speaker"] != b["speaker"] and float(b["start"]) > float(a["end"]))
    return {
        "note": "Production build_utterances_from_words uses max_pause_sec=1.0 while its docstring "
                "claims 2.0 s. Small thresholds make the builder a no-op here because it breaks only "
                "on speaker change or a silence longer than the threshold.",
        "speaker_change_gaps_observed": len(diff_gaps),
        "median_speaker_change_gap": (diff_gaps[len(diff_gaps) // 2] if diff_gaps else None),
        "min_speaker_change_gap": (diff_gaps[0] if diff_gaps else None),
        "max_speaker_change_gap": (diff_gaps[-1] if diff_gaps else None),
        "ref_segments": len(ref_segments),
    }


# ---------------------------------------------------------------------------
# Whisper timing arm (C)
# ---------------------------------------------------------------------------
def whisper_timing_words(asr_wav: Path, hyp_text: str, cache_tag: str = "C") -> Dict[str, Any]:
    """Whisper word stamps for the Cohere *text* (no text substitution)."""
    from app.services.forced_alignment import align_words_to_audio

    path = C.WORD_DIR / f"{cache_tag}_words.json"
    meta = C.WORD_DIR / f"{cache_tag}_meta.json"
    if path.exists() and meta.exists():
        return {"words": C.read_json(path), "resources": C.read_json(meta), "source": "cache"}
    with PR.ResourceMeter("armC_whisper") as m:
        words = align_words_to_audio(asr_wav, hyp_text)
    res = m.report()
    C.write_json(path, words)
    C.write_json(meta, res)
    return {"words": words, "resources": res, "source": "fresh"}


# ---------------------------------------------------------------------------
# Arm evaluation helpers
# ---------------------------------------------------------------------------
def score_segments_arm(name: str, ref_segments: List[Dict[str, Any]], segs: List[Dict[str, Any]],
                       audio_duration: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    diar = SS.score_diarization(ref_segments, segs, audio_duration)
    agree = SS.segment_agreement_weighted(ref_segments, segs)
    return {
        "arm": name,
        "n_segments": len(segs),
        "diar": diar,
        "seg_agreement": agree,
        "agree": agree["pct_ok"],
        "agree_word": agree["agree_word"],
        "agree_duration": agree["agree_duration"],
        **(extra or {}),
    }
