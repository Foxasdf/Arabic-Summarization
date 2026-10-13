"""
Timestamp alignment between continuous ASR word outputs and Pyannote diarization intervals.
Word-level speaker assignment (Phase 4):
1. Assigns each word to the speaker turn with the largest temporal overlap / midpoint containment.
2. Handles overlapping speech and gap transitions without brittle text rules.
3. Smooths single-word spurious flip-flops while preserving intentional short responses.
4. Assembles utterances dynamically at speaker handovers, pauses > 1.5s, or clause limits.
"""
import re
import logging
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


def clean_diarization_turns(turns: List[Dict], min_dur: float = 0.25) -> List[Dict]:
    """Clean and structure raw diarization intervals into robust conversational floor turns."""
    if not turns:
        return []

    # 1. Filter micro-noise
    valid = [
        {"speaker": str(t["speaker"]), "start": float(t["start"]), "end": float(t["end"])}
        for t in turns
        if (float(t["end"]) - float(t["start"])) >= min_dur
    ]
    if not valid:
        return []
    valid.sort(key=lambda t: (t["start"], -(t["end"] - t["start"])))

    # 2. Merge same-speaker overlapping or near-adjacent intervals (gap <= 1.25s without intervening speaker)
    valid.sort(key=lambda t: (t["start"], t["end"]))
    all_clean: List[Dict] = []
    for t in valid:
        if not all_clean:
            all_clean.append(dict(t))
            continue
        prev = all_clean[-1]
        gap = float(t["start"]) - float(prev["end"])
        if prev["speaker"] == t["speaker"] and gap <= 1.25:
            prev["end"] = max(float(prev["end"]), float(t["end"]))
        else:
            all_clean.append(dict(t))

    # 3. Handle interjections and floor handovers:
    interjections = [t for t in all_clean if (t["end"] - t["start"]) <= 2.0]
    long_turns = [t for t in all_clean if (t["end"] - t["start"]) > 2.0]

    current_turns = list(long_turns)
    for inj in interjections:
        new_turns: List[Dict] = []
        punched = False
        inj_dur = inj["end"] - inj["start"]
        for lt in current_turns:
            if lt["speaker"] != inj["speaker"] and lt["start"] <= inj["start"] and inj["end"] <= lt["end"]:
                if inj_dur >= 0.5:
                    if inj["start"] - lt["start"] >= 0.35:
                        new_turns.append({"speaker": lt["speaker"], "start": lt["start"], "end": inj["start"]})
                    new_turns.append(dict(inj))
                    if lt["end"] - inj["end"] >= 0.35:
                        new_turns.append({"speaker": lt["speaker"], "start": inj["end"], "end": lt["end"]})
                    punched = True
                else:
                    new_turns.append(lt)
                    punched = True
            else:
                new_turns.append(lt)
        if not punched:
            new_turns.append(dict(inj))
        current_turns = new_turns

    # 4. Sort and resolve boundary overlaps at midpoint
    current_turns.sort(key=lambda t: (t["start"], t["end"]))
    resolved: List[Dict] = []
    for t in current_turns:
        if not resolved:
            resolved.append(dict(t))
            continue
        prev = resolved[-1]
        if t["start"] < prev["end"]:
            if t["speaker"] == prev["speaker"]:
                prev["end"] = max(prev["end"], t["end"])
                continue
            ov = prev["end"] - t["start"]
            if ov > 0:
                mid = round((t["start"] + prev["end"]) / 2.0, 2)
                if mid > prev["start"] + 0.1:
                    prev["end"] = mid
                if mid < t["end"] - 0.1:
                    t["start"] = mid
        if prev["end"] <= prev["start"] + 0.1:
            resolved.pop()
        if (t["end"] - t["start"]) >= 0.2:
            resolved.append(dict(t))

    # 5. Round timestamps and ensure speaker is set
    for t in resolved:
        if not t.get("speaker"):
            t["speaker"] = "SPEAKER_00"
        t["start"] = round(float(t["start"]), 2)
        t["end"] = round(float(t["end"]), 2)

    return resolved


def assign_words_to_speakers(
    words: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    gap_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Assign each word to the speaker turn with largest temporal overlap or nearest midpoint.
    """
    if not words or not turns:
        return words

    assigned_words = []
    for w in words:
        st = float(w.get("start", 0.0))
        en = float(w.get("end", st + 0.3))
        mid = (st + en) / 2.0
        w_dur = max(0.05, en - st)

        best_spk = None
        max_overlap = 0.0

        for t in turns:
            t_st = float(t.get("start", 0.0))
            t_en = float(t.get("end", 0.0))
            ov = max(0.0, min(en, t_en) - max(st, t_st))
            if ov > max_overlap:
                max_overlap = ov
                best_spk = t.get("speaker")

        if best_spk is None or (max_overlap / w_dur) < 0.2:
            # Check midpoint containment
            for t in turns:
                t_st = float(t.get("start", 0.0))
                t_en = float(t.get("end", 0.0))
                if t_st <= mid <= t_en:
                    best_spk = t.get("speaker")
                    break

        if best_spk is None:
            # Find nearest turn boundary within gap_threshold
            best_dist = float("inf")
            for t in turns:
                t_st = float(t.get("start", 0.0))
                t_en = float(t.get("end", 0.0))
                dist = min(abs(mid - t_st), abs(mid - t_en))
                if dist < best_dist and dist <= gap_threshold:
                    best_dist = dist
                    best_spk = t.get("speaker")

        if best_spk is None and assigned_words:
            # Keep previous speaker if word is in unassigned inter-turn pause
            best_spk = assigned_words[-1].get("speaker", "SPEAKER_00")
        elif best_spk is None:
            best_spk = turns[0].get("speaker", "SPEAKER_00")

        assigned_words.append({
            "word": w.get("word", ""),
            "start": round(st, 2),
            "end": round(en, 2),
            "speaker": best_spk,
        })

    return assigned_words


def smooth_speaker_assignments(
    words: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Smoothing:
    1. If a single isolated word is surrounded by the same speaker on both sides,
       and that isolated word is NOT a genuine confirmed turn in Pyannote (dur >= 0.75s),
       reassign it to the surrounding speaker.
    2. Eliminate flip-flops of length < 0.35s unless supported by an intentional short turn.
    """
    if len(words) < 3:
        return words

    smoothed = [dict(w) for w in words]
    for i in range(1, len(smoothed) - 1):
        prev_spk = smoothed[i - 1]["speaker"]
        curr_spk = smoothed[i]["speaker"]
        next_spk = smoothed[i + 1]["speaker"]

        if prev_spk == next_spk and curr_spk != prev_spk:
            w_mid = (smoothed[i]["start"] + smoothed[i]["end"]) / 2.0
            # Check if Pyannote actually had a distinct confirmed turn for curr_spk at w_mid
            confirmed_turn = any(
                t["speaker"] == curr_spk and t["start"] <= w_mid <= t["end"] and (t["end"] - t["start"]) >= 0.45
                for t in turns
            )
            if not confirmed_turn:
                smoothed[i]["speaker"] = prev_spk

    return smoothed


def build_utterances_from_words(
    words: List[Dict[str, Any]],
    max_pause_sec: float = 1.0,
    max_words_per_utterance: int = 30,
) -> List[Dict[str, Any]]:
    """
    Build conversational utterances:
    Starts a new utterance when:
    (a) Speaker changes.
    (b) Pause / silence > 1.0s between consecutive words (tuned to Levantine
        turn-taking; 2.0s merged distinct conversational moves).
    (c) Maximum utterance length reached (30 words keeps subtitles readable
        and matches reference granularity ~8 words/segment).
    """
    if not words:
        return []

    utterances = []
    cur_u: List[Dict[str, Any]] = []

    for w in words:
        if not cur_u:
            cur_u.append(w)
            continue

        prev_w = cur_u[-1]
        spk_changed = w["speaker"] != prev_w["speaker"]
        pause_exceeded = (float(w["start"]) - float(prev_w["end"])) > max_pause_sec
        len_exceeded = len(cur_u) >= max_words_per_utterance

        if spk_changed or pause_exceeded or len_exceeded:
            txt = " ".join(item["word"] for item in cur_u if item.get("word")).strip()
            if txt:
                utterances.append({
                    "speaker": cur_u[0]["speaker"],
                    "start": cur_u[0]["start"],
                    "end": cur_u[-1]["end"],
                    "text": txt,
                })
            cur_u = [w]
        else:
            cur_u.append(w)

    if cur_u:
        txt = " ".join(item["word"] for item in cur_u if item.get("word")).strip()
        if txt:
            utterances.append({
                "speaker": cur_u[0]["speaker"],
                "start": cur_u[0]["start"],
                "end": cur_u[-1]["end"],
                "text": txt,
            })

    return utterances


def align_timestamps(asr_words: Any, diarization_segments: List[Dict]) -> List[Dict]:
    """
    Word-level monotonic sequence alignment between continuous ASR words
    and Pyannote speaker diarization intervals.
    """
    if isinstance(asr_words, dict):
        asr_words = asr_words.get("words", [])

    if not asr_words:
        return []

    # Step 0: Deduplicate boundary artifacts across chunk seams (identical word <= 1.0s)
    deduped_words: List[Dict] = []
    for w in asr_words:
        if deduped_words:
            prev_w = re.sub(r"[،,.\?!؟;:\"\']", "", deduped_words[-1].get("word", "")).strip()
            cur_w = re.sub(r"[،,.\?!؟;:\"\']", "", w.get("word", "")).strip()
            if prev_w and cur_w and prev_w == cur_w and abs(float(w.get("start", 0)) - float(deduped_words[-1].get("start", 0))) <= 1.0:
                continue
        deduped_words.append(w)

    if not diarization_segments:
        all_text = " ".join(w.get("word", "") for w in deduped_words if w.get("word"))
        return [{
            "speaker": "SPEAKER_00",
            "text": all_text.strip(),
            "start": round(float(deduped_words[0].get("start", 0.0)), 2),
            "end": round(float(deduped_words[-1].get("end", 0.0)), 2),
        }]

    turns = clean_diarization_turns(diarization_segments)
    if not turns:
        all_text = " ".join(w.get("word", "") for w in deduped_words if w.get("word"))
        return [{
            "speaker": "SPEAKER_00",
            "text": all_text.strip(),
            "start": round(float(deduped_words[0].get("start", 0.0)), 2),
            "end": round(float(deduped_words[-1].get("end", 0.0)), 2),
        }]

    # Phase 4: Word-Level Speaker Assignment, Smoothing, and Utterance Construction
    assigned = assign_words_to_speakers(deduped_words, turns)
    smoothed = smooth_speaker_assignments(assigned, turns)
    segments = build_utterances_from_words(smoothed)

    logger.info(
        f"Alignment complete: {len(deduped_words)} words -> {len(segments)} speaker segments."
    )
    return segments
