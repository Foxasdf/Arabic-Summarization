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

import numpy as np

logger = logging.getLogger(__name__)


def clean_diarization_turns(
    turns: List[Dict],
    min_dur: float = 0.15,
    merge_gap: float = 0.5,
    punch_min: float = 0.75,
) -> List[Dict]:
    """Clean raw diarization intervals into conversational speaker turns.
    Preserves all authentic speech intervals without destructive hole-punching or truncating.

    1. Filters micro-noise clicks (< min_dur).
    2. Merges same-speaker contiguous or near-adjacent intervals (gap <= merge_gap).
    3. Preserves concurrent speech tracks so each speaker's voice is retained.
    """
    if not turns:
        return []

    # 1. Filter micro-noise clicks (< min_dur)
    valid = [
        {"speaker": str(t["speaker"]), "start": round(float(t["start"]), 2), "end": round(float(t["end"]), 2)}
        for t in turns
        if (float(t["end"]) - float(t["start"])) >= min_dur
    ]
    if not valid:
        return []

    # 2. Chronological same-speaker merging: ONLY merge strictly consecutive turns of same speaker
    valid.sort(key=lambda t: (t["start"], t["end"]))
    merged_turns: List[Dict] = []
    for t in valid:
        if not merged_turns:
            merged_turns.append(dict(t))
            continue
        prev = merged_turns[-1]
        if t["speaker"] == prev["speaker"] and 0.0 <= (t["start"] - prev["end"]) <= merge_gap:
            prev["end"] = max(prev["end"], t["end"])
        else:
            merged_turns.append(dict(t))

    # 3. Ensure speaker name and valid rounded timestamps
    for t in merged_turns:
        if not t.get("speaker"):
            t["speaker"] = "SPEAKER_00"
        t["start"] = round(float(t["start"]), 2)
        t["end"] = round(float(t["end"]), 2)

    return merged_turns


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
        best_dur = float("inf")

        for t in turns:
            t_st = float(t.get("start", 0.0))
            t_en = float(t.get("end", 0.0))
            t_dur = max(0.01, t_en - t_st)
            ov = max(0.0, min(en, t_en) - max(st, t_st))
            if ov > max_overlap + 1e-4:
                max_overlap = ov
                best_dur = t_dur
                best_spk = t.get("speaker")
            elif ov > 0 and abs(ov - max_overlap) <= 1e-4 and t_dur < best_dur:
                best_dur = t_dur
                best_spk = t.get("speaker")

        if best_spk is None or (max_overlap / w_dur) < 0.2:
            # Check midpoint containment, preferring shorter/more specific turn
            best_dur = float("inf")
            for t in turns:
                t_st = float(t.get("start", 0.0))
                t_en = float(t.get("end", 0.0))
                t_dur = max(0.01, t_en - t_st)
                if t_st <= mid <= t_en and t_dur < best_dur:
                    best_dur = t_dur
                    best_spk = t.get("speaker")

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
            w_st = smoothed[i]["start"]
            w_en = smoothed[i]["end"]
            w_mid = (w_st + w_en) / 2.0
            # Check if Pyannote actually had a distinct confirmed turn for curr_spk at or overlapping this word
            confirmed_turn = any(
                t["speaker"] == curr_spk
                and (
                    t["start"] <= w_mid <= t["end"]
                    or max(0.0, min(w_en, t["end"]) - max(w_st, t["start"])) >= 0.1
                )
                and (t["end"] - t["start"]) >= 0.20
                for t in turns
            )
            if not confirmed_turn:
                smoothed[i]["speaker"] = prev_spk

    return smoothed


def _supporting_turn_stats(
    turns: List[Dict[str, Any]],
    turn_embeddings: List[Optional[np.ndarray]],
    start: float,
    end: float,
    speaker: str,
) -> tuple:
    """Return total supporting-turn duration and mean embedding for a word run."""
    total_duration = 0.0
    vectors: List[np.ndarray] = []
    for turn, emb in zip(turns, turn_embeddings):
        if emb is None or turn.get("speaker") != speaker:
            continue
        overlap = max(
            0.0,
            min(end, float(turn.get("end", 0.0)))
            - max(start, float(turn.get("start", 0.0))),
        )
        if overlap < 0.1:
            continue
        total_duration += float(turn.get("end", 0.0)) - float(turn.get("start", 0.0))
        vectors.append(emb)

    if not vectors:
        return total_duration, None
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    return total_duration, (mean / norm if norm > 0 else None)


def smooth_short_runs_with_embeddings(
    words: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    turn_embeddings: List[Optional[np.ndarray]],
    speaker_centroids: Dict[str, np.ndarray],
    max_run_sec: float = 0.35,
    max_run_words: int = 3,
    max_support_sec: float = 1.0,
    margin: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Absorb short word runs that are acoustically closer to the surrounding
    speaker than to their current label.

    A run is only considered when:
    1. It is a very short fragment (<= max_run_sec) or at most max_run_words.
    2. Both neighbouring runs belong to the same speaker.
    3. The supporting turn for the current label is <= max_support_sec, so a
       genuine longer interjection is never absorbed just because it is short.
    4. The surrounding speaker's centroid is better by at least ``margin``.

    This targets Pyannote's short overlapping turns stealing a few words from
    a continuous utterance without flattening intentional backchannels.
    """
    if len(words) < 3 or not turns or not turn_embeddings or not speaker_centroids:
        return words
    if len(turns) != len(turn_embeddings):
        logger.warning(
            "Speaker embedding context does not match turns; skipping short-run smoothing."
        )
        return words

    smoothed = [dict(w) for w in words]
    runs: List[Dict[str, Any]] = []
    for i, w in enumerate(smoothed):
        if runs and runs[-1]["spk"] == w.get("speaker"):
            runs[-1]["end"] = i
        else:
            runs.append({"spk": w.get("speaker"), "start": i, "end": i})

    changed = 0
    for run in runs:
        i0, i1 = run["start"], run["end"]
        if i0 == 0 or i1 == len(smoothed) - 1:
            continue

        current = run["spk"]
        prev_spk = smoothed[i0 - 1].get("speaker")
        next_spk = smoothed[i1 + 1].get("speaker")
        if prev_spk != next_spk or current == prev_spk:
            continue

        start = float(smoothed[i0].get("start", 0.0))
        end = float(smoothed[i1].get("end", start))
        duration = end - start
        n_words = i1 - i0 + 1
        if duration > max_run_sec and n_words > max_run_words:
            continue

        support_duration, support_emb = _supporting_turn_stats(
            turns, turn_embeddings, start, end, current
        )
        if support_emb is None or support_duration > max_support_sec:
            continue

        current_centroid = speaker_centroids.get(current)
        neighbour_centroid = speaker_centroids.get(prev_spk)
        if current_centroid is None or neighbour_centroid is None:
            continue

        current_sim = float(np.dot(support_emb, current_centroid))
        neighbour_sim = float(np.dot(support_emb, neighbour_centroid))
        if neighbour_sim - current_sim < margin:
            continue

        for i in range(i0, i1 + 1):
            smoothed[i]["speaker"] = prev_spk
        changed += 1

    if changed:
        logger.info(
            f"Short-run smoothing reassigned {changed} word run(s) using speaker embeddings."
        )
    return smoothed


def merge_consecutive_speaker_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments belonging to the same speaker into a single segment.
    Ensures that a single speaker never appears in consecutive back-to-back fields.
    If the gap between consecutive segments > 2.0s, separates with an Arabic comma '، '
    (unless the previous text already ends with punctuation) for enhanced readability.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    punctuation = (".", "،", "؟", "!", "؛", ",", "?", ":", "…")

    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        spk = seg.get("speaker") or "SPEAKER_00"
        start = round(float(seg.get("start", 0.0)), 2)
        end = round(float(seg.get("end", 0.0)), 2)

        if merged and merged[-1].get("speaker") == spk:
            prev_end = float(merged[-1].get("end", 0.0))
            gap = start - prev_end
            merged[-1]["end"] = max(prev_end, end)
            prev_txt = merged[-1]["text"].rstrip()
            if gap > 2.0 and not prev_txt.endswith(punctuation):
                merged[-1]["text"] = f"{prev_txt}، {txt}"
            else:
                merged[-1]["text"] = f"{prev_txt} {txt}"
        else:
            merged.append({
                "speaker": spk,
                "start": start,
                "end": end,
                "text": txt,
            })

    return merged


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
    Finally consolidates any consecutive segments of the same speaker so no
    speaker appears in two consecutive fields.
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

    return merge_consecutive_speaker_segments(utterances)


def align_timestamps(
    asr_words: Any,
    diarization_segments: List[Dict],
    speaker_context: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Word-level monotonic sequence alignment between continuous ASR words
    and Pyannote speaker diarization intervals.

    ``speaker_context`` may contain per-turn embeddings and per-speaker
    centroids from the cached Pyannote embedding model. When present, short
    mislabelled word runs are absorbed into the surrounding speaker.
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
    if speaker_context:
        embeddings = speaker_context.get("embeddings")
        centroids = speaker_context.get("centroids")
        if embeddings is not None and centroids:
            smoothed = smooth_short_runs_with_embeddings(
                smoothed,
                diarization_segments,
                embeddings,
                centroids,
            )
    raw_segments = build_utterances_from_words(smoothed)
    segments = merge_consecutive_speaker_segments(raw_segments)

    logger.info(
        f"Alignment complete: {len(deduped_words)} words -> {len(segments)} speaker segments."
    )
    return segments
