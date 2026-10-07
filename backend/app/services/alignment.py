"""
Timestamp alignment between ASR word outputs and diarization intervals.
Syntactically clause-aware alignment: avoids cutting phrases mid-sentence,
prevents micro-turn fragmentation, and aligns natural dialogue turns.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def align_timestamps(asr_words: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
    """
    Align continuous ASR word-level timestamps with speaker diarization segments
    at the clause/sentence boundary level.
    
    Args:
        asr_words: [{"word": "مرحبا", "start": 0.0, "end": 0.5}, ...]
        diarization_segments: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 10.2}, ...]
        
    Returns:
        List of grouped coherent speaker segments:
        [{"speaker": "SPEAKER_00", "text": "مرحبا بكم في هذا الاجتماع", "start": 0.0, "end": 10.2}, ...]
    """
    if not asr_words:
        return []

    if not diarization_segments:
        all_text = " ".join(w.get("word", "") for w in asr_words if w.get("word"))
        return [{
            "speaker": "SPEAKER_00",
            "text": all_text.strip(),
            "start": round(float(asr_words[0].get("start", 0.0)), 2),
            "end": round(float(asr_words[-1].get("end", 0.0)), 2),
        }]

    # Step 1: Smooth and clean diarization segments
    sorted_turns = sorted(
        (t for t in diarization_segments if float(t.get("end", 0)) - float(t.get("start", 0)) > 0.1),
        key=lambda t: float(t.get("start", 0))
    )

    smoothed_turns: List[Dict] = []
    for t in sorted_turns:
        item = {
            "speaker": str(t["speaker"]),
            "start": float(t["start"]),
            "end": float(t["end"])
        }
        if smoothed_turns and smoothed_turns[-1]["speaker"] == item["speaker"] and item["start"] - smoothed_turns[-1]["end"] <= 1.2:
            smoothed_turns[-1]["end"] = max(smoothed_turns[-1]["end"], item["end"])
        else:
            smoothed_turns.append(item)

    # Filter out micro-turns (< 1.2s) sandwiched between the same speaker
    i = 0
    cleaned_turns: List[Dict] = []
    while i < len(smoothed_turns):
        cur = smoothed_turns[i]
        dur = cur["end"] - cur["start"]
        if dur < 1.2:
            prev_spk = cleaned_turns[-1]["speaker"] if cleaned_turns else None
            next_spk = smoothed_turns[i + 1]["speaker"] if i + 1 < len(smoothed_turns) else None
            if prev_spk and next_spk and prev_spk == next_spk:
                cleaned_turns[-1]["end"] = max(cleaned_turns[-1]["end"], cur["end"])
                i += 1
                continue
        cleaned_turns.append(cur)
        i += 1

    final_turns: List[Dict] = []
    for t in cleaned_turns:
        if final_turns and final_turns[-1]["speaker"] == t["speaker"] and t["start"] - final_turns[-1]["end"] <= 1.2:
            final_turns[-1]["end"] = max(final_turns[-1]["end"], t["end"])
        else:
            final_turns.append(t)

    # Step 2: Group words into grammatically coherent clauses / sentences
    clauses: List[List[Dict]] = []
    cur_clause_words: List[Dict] = []
    terminators = {'.', '!', '؟', '?', ';', '؛'}
    question_starters = {'وش', 'كيف', 'ليه', 'لماذا', 'هل', 'متى', 'أين', 'وين', 'ماذا'}

    for idx, w in enumerate(asr_words):
        word_text = (w.get("word") or "").strip()
        if not word_text:
            continue

        clean_w = re.sub(r"[،,.\?!؟;:\"\']", "", word_text)

        # If a question word appears and previous clause is substantial, split to start question cleanly
        if len(cur_clause_words) >= 4 and clean_w in question_starters:
            clauses.append(cur_clause_words)
            cur_clause_words = []

        cur_clause_words.append(w)
        has_terminator = any(word_text.endswith(term) for term in terminators)
        has_comma = word_text.endswith('،') or word_text.endswith(',')
        long_enough = len(cur_clause_words) >= 6

        # Check silence gap to next word (indicates natural speaker transition or acoustic pause)
        has_gap = False
        if idx + 1 < len(asr_words):
            next_start = float(asr_words[idx + 1].get("start", 0))
            cur_end = float(w.get("end", 0))
            if next_start - cur_end >= 0.8:
                has_gap = True

        if has_terminator or (has_comma and long_enough) or has_gap or len(cur_clause_words) >= 18:
            clauses.append(cur_clause_words)
            cur_clause_words = []

    if cur_clause_words:
        clauses.append(cur_clause_words)

    if not clauses:
        return []

    # Step 3: Assign each clause to the dominant speaker based on temporal overlap
    tagged_clauses: List[Dict] = []
    for c_words in clauses:
        c_start = float(c_words[0].get("start", 0.0))
        c_end = float(c_words[-1].get("end", c_start + 0.5))
        c_text = " ".join(w.get("word", "") for w in c_words).strip()
        if not c_text:
            continue

        speaker_overlaps: Dict[str, float] = {}
        for turn in final_turns:
            overlap = max(0.0, min(c_end, turn["end"]) - max(c_start, turn["start"]))
            if overlap > 0.05:
                spk = turn["speaker"]
                speaker_overlaps[spk] = speaker_overlaps.get(spk, 0.0) + overlap

        if speaker_overlaps:
            best_spk = max(speaker_overlaps.items(), key=lambda x: x[1])[0]
        else:
            # Nearest turn by midpoint distance
            c_mid = (c_start + c_end) / 2.0
            best_spk = min(
                final_turns,
                key=lambda t: min(abs(c_mid - t["start"]), abs(c_mid - t["end"]))
            )["speaker"]

        tagged_clauses.append({
            "speaker": best_spk,
            "text": c_text,
            "start": round(c_start, 2),
            "end": round(c_end, 2),
        })

    # Step 4: Merge adjacent clauses of the same speaker
    merged_segments: List[Dict] = []
    for tc in tagged_clauses:
        if merged_segments and merged_segments[-1]["speaker"] == tc["speaker"]:
            merged_segments[-1]["text"] += " " + tc["text"]
            merged_segments[-1]["end"] = max(merged_segments[-1]["end"], tc["end"])
        else:
            merged_segments.append(dict(tc))

    # Step 5: Post-processing to absorb single-clause micro flip-flops
    final_segments: List[Dict] = []
    i = 0
    while i < len(merged_segments):
        cur = merged_segments[i]
        dur = cur["end"] - cur["start"]
        word_count = len(cur["text"].split())

        # If a short 1-3 word blip (< 1.8s) is sandwiched between the same speaker
        if (
            (word_count <= 3 or dur < 1.8)
            and i > 0
            and i + 1 < len(merged_segments)
            and final_segments
            and final_segments[-1]["speaker"] == merged_segments[i + 1]["speaker"]
        ):
            final_segments[-1]["text"] += " " + cur["text"]
            final_segments[-1]["end"] = max(final_segments[-1]["end"], cur["end"])
            i += 1
            continue

        if final_segments and final_segments[-1]["speaker"] == cur["speaker"]:
            final_segments[-1]["text"] += " " + cur["text"]
            final_segments[-1]["end"] = max(final_segments[-1]["end"], cur["end"])
        else:
            final_segments.append(cur)
        i += 1

    logger.info(f"Clause alignment: {len(asr_words)} words across {len(clauses)} clauses -> {len(final_segments)} smooth speaker turns.")
    return final_segments
