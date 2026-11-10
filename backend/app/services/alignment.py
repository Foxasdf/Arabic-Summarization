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
    min_dur: Optional[float] = None,
    merge_gap: Optional[float] = None,
    punch_min: float = 0.75,
    absorb_enclosed_micro: bool = False,
    *,
    max_same_speaker_gap: Optional[float] = None,
    min_duration: Optional[float] = None,
    isolated_min_duration: Optional[float] = None,
    drop_blip_duration: Optional[float] = None,
) -> List[Dict]:
    """Clean raw diarization intervals into conversational speaker turns.

    1. Drops invalid timestamps and isolated micro-blips (< min_dur, default 0.20s).
    2. Merges same-speaker overlapping or near-adjacent intervals (gap <= merge_gap, default 0.4s),
       even when another speaker had a brief enclosed backchannel inside the earlier turn.
    3. Absorbs micro-segments (< 0.50s) when they are near-adjacent to a same-speaker turn
       or (when absorb_enclosed_micro=True) enclosed inside another speaker's dominant turn.
    4. Keeps isolated short turns in [0.25s, 0.50s] (e.g. "نعم", "لا", "طيب", "صحيح") surrounded
       by silence so +/-200ms boundary padding expands them to ~0.65s-0.90s for Cohere ASR.
    """
    if not turns:
        return []

    from ..config import settings

    if min_dur is None:
        min_dur = (
            float(drop_blip_duration)
            if drop_blip_duration is not None
            else float(getattr(settings, "TURN_DROP_BLIP_DURATION_SEC", 0.20))
        )
    if merge_gap is None:
        merge_gap = (
            float(max_same_speaker_gap)
            if max_same_speaker_gap is not None
            else float(getattr(settings, "TURN_MERGE_SAME_SPEAKER_GAP", 0.4))
        )
    micro_thresh = (
        float(min_duration)
        if min_duration is not None
        else float(getattr(settings, "TURN_MIN_DURATION_SEC", 0.5))
    )
    isolated_min_dur = (
        float(isolated_min_duration)
        if isolated_min_duration is not None
        else float(getattr(settings, "TURN_ISOLATED_MIN_DURATION_SEC", 0.25))
    )

    # 1. Sanitize and drop invalid or ultra-short noise blips (< min_dur)
    raw_valid: List[Dict] = []
    for t in turns:
        st = max(0.0, float(t.get("start", 0.0)))
        en = float(t.get("end", 0.0))
        if en - st < min_dur:
            continue
        spk = str(t.get("speaker") or "SPEAKER_00")
        raw_valid.append({"speaker": spk, "start": round(st, 2), "end": round(en, 2)})

    if not raw_valid:
        return []

    raw_valid.sort(key=lambda t: (t["start"], t["end"]))

    # 2. Merge same-speaker turns that overlap (gap < 0) or have gap <= merge_gap
    # when no other speaker is active inside the gap [prev_same["end"], cur["start"]].
    # For micro-segments (< micro_thresh = 0.5s), allow slightly wider same-speaker absorption (up to 2 * merge_gap).
    merged_turns: List[Dict] = []
    for t in raw_valid:
        cur = dict(t)
        cur_dur = cur["end"] - cur["start"]
        merged = False

        # Search backwards for the most recent turn of the same speaker
        for idx in range(len(merged_turns) - 1, -1, -1):
            prev = merged_turns[idx]
            if prev["speaker"] != cur["speaker"]:
                continue

            gap = cur["start"] - prev["end"]
            effective_gap_limit = merge_gap * 1.75 if (cur_dur < micro_thresh or (prev["end"] - prev["start"]) < micro_thresh) else merge_gap

            if gap <= effective_gap_limit:
                # Check if another speaker has a non-enclosed turn occupying the gap [prev["end"], cur["start"]]
                intervening = False
                if gap > 0.0:
                    for other in merged_turns[idx + 1:]:
                        if other["speaker"] != cur["speaker"]:
                            # Intervening if it overlaps the gap and wasn't completely finished before prev["end"]
                            ov = min(cur["start"], other["end"]) - max(prev["end"], other["start"])
                            if ov > 0.05 or (other["start"] >= prev["end"] - 0.05 and other["end"] <= cur["start"] + 0.05):
                                intervening = True
                                break
                if not intervening:
                    prev["end"] = round(max(prev["end"], cur["end"]), 2)
                    merged = True
            break

        if not merged:
            merged_turns.append(cur)

    # 3. Handle remaining micro-segments (< 0.50s):
    # - If absorb_enclosed_micro=True and a <0.50s turn is enclosed inside a dominant turn of another speaker, drop/absorb it.
    # - If a turn is < isolated_min_dur (0.25s) and min_dur >= 0.20s, drop it unless caller explicitly requested a smaller min_dur (e.g. unit test min_dur=0.15).
    effective_isolated_min = min(isolated_min_dur, min_dur) if min_dur < 0.20 else isolated_min_dur
    filtered_turns: List[Dict] = []
    for i, t in enumerate(merged_turns):
        dur = t["end"] - t["start"]
        if dur < micro_thresh:
            if absorb_enclosed_micro:
                is_enclosed = any(
                    i != j
                    and other["speaker"] != t["speaker"]
                    and other["start"] - 0.05 <= t["start"]
                    and other["end"] + 0.05 >= t["end"]
                    and (other["end"] - other["start"]) >= max(1.0, dur * 2.0)
                    for j, other in enumerate(merged_turns)
                )
                if is_enclosed:
                    continue
            if dur < effective_isolated_min:
                continue
        filtered_turns.append(t)

    if not filtered_turns and merged_turns:
        filtered_turns = [merged_turns[0]]

    # 4. Final chronological pass to merge any newly adjacent same-speaker turns
    filtered_turns.sort(key=lambda t: (t["start"], t["end"]))
    final_turns: List[Dict] = []
    for t in filtered_turns:
        if final_turns and final_turns[-1]["speaker"] == t["speaker"] and (t["start"] - final_turns[-1]["end"]) <= merge_gap:
            final_turns[-1]["end"] = round(max(final_turns[-1]["end"], t["end"]), 2)
        else:
            final_turns.append({
                "speaker": t.get("speaker") or "SPEAKER_00",
                "start": round(float(t["start"]), 2),
                "end": round(float(t["end"]), 2),
            })

    return final_turns


def resolve_turn_overlaps(
    turns: List[Dict],
    backchannel_max: float = 0.45,
    min_piece: float = 0.25,
    dup_overlap_ratio: float = 0.70,
    max_passes: int = 2000,
) -> List[Dict]:
    """Make diarization turns non-overlapping for single-channel ASR slicing.

    1. Near-duplicate simultaneous turns (overlap >= 70% of the shorter turn and
       comparable lengths) -> keep only the longer one.
    2. Turn B enclosed inside a longer turn A of another speaker:
       - B < backchannel_max (0.45s): brief backchannel blip -> drop B.
       - otherwise carve A into [A.start, B.start] and [B.end, A.end]
         (pieces shorter than min_piece are discarded).
    3. Partial handover overlap (prev.end > cur.start, different speakers):
       split at the overlap midpoint so the shared audio is transcribed once.
    Returns turns sorted chronologically with no cross-speaker overlap.
    """
    work = [
        {"speaker": str(t.get("speaker") or "SPEAKER_00"), "start": float(t["start"]), "end": float(t["end"])}
        for t in turns
        if float(t["end"]) - float(t["start"]) > 0
    ]

    def _dur(t: Dict) -> float:
        return t["end"] - t["start"]

    for _ in range(max_passes):
        work.sort(key=lambda t: (t["start"], t["end"]))
        changed = False
        n = len(work)
        for i in range(n):
            a = work[i]
            for j in range(i + 1, n):
                b = work[j]
                if b["start"] >= a["end"]:
                    break
                if a["speaker"] == b["speaker"]:
                    a["end"] = max(a["end"], b["end"])
                    work.remove(b)
                    changed = True
                    break
                ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
                if ov <= 0.0:
                    continue
                long_t, short_t = (a, b) if _dur(a) >= _dur(b) else (b, a)
                ld, sd = _dur(long_t), _dur(short_t)

                # 1. Near-identical simultaneous turns -> keep the longer one
                if ov >= dup_overlap_ratio * sd and ld <= max(2.0 * sd, sd + 1.0):
                    work.remove(short_t)
                    changed = True
                    break

                # 2. Enclosed turn (>= 90% of the short turn inside the long turn)
                if ov >= 0.9 * sd:
                    if sd < backchannel_max:
                        work.remove(short_t)
                    else:
                        work.remove(long_t)
                        left = {"speaker": long_t["speaker"], "start": long_t["start"], "end": min(long_t["end"], short_t["start"])}
                        right = {"speaker": long_t["speaker"], "start": max(long_t["start"], short_t["end"]), "end": long_t["end"]}
                        for piece in (left, right):
                            if _dur(piece) >= min_piece:
                                work.append(piece)
                    changed = True
                    break

                # 3. Partial handover overlap -> preserve short turns (< 1.5s) when paired with
                # a longer turn (>= 1.5s) by trimming the longer turn; only use midpoint splitting
                # when both turns are >= 1.5s (or both < 1.5s).
                first, second = (a, b) if a["start"] <= b["start"] else (b, a)
                f_dur, s_dur = _dur(first), _dur(second)
                if f_dur < 1.5 and s_dur >= 1.5:
                    second["start"] = first["end"]
                elif s_dur < 1.5 and f_dur >= 1.5:
                    first["end"] = second["start"]
                else:
                    mid = 0.5 * (first["end"] + second["start"])
                    first["end"] = mid
                    second["start"] = mid
                for t in (first, second):
                    if _dur(t) < min_piece and t in work:
                        work.remove(t)
                changed = True
                break
            if changed:
                break
        if not changed:
            break

    work.sort(key=lambda t: (t["start"], t["end"]))
    return [
        {"speaker": t["speaker"], "start": round(t["start"], 2), "end": round(t["end"], 2)}
        for t in work
    ]


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
