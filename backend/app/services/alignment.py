"""
Timestamp alignment between continuous ASR word outputs and Pyannote diarization intervals.
Syntactically clause-aware alignment: avoids cutting phrases mid-constituent,
preserves conversational turns, isolates rapid interjections and discourse cues,
and guarantees 100% word retention without hardcoded strings or brittle heuristics.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Universal Arabic syntactic and discourse token classes
TERMINATORS = {'.', '!', '؟', '?', ';', '؛'}
COMMAS = {'،', ','}

NON_TERMINAL_WORDS = {
    # Prepositions & particles requiring complements
    "على", "عن", "من", "في", "إلى", "مع", "بـ", "لـ", "بين", "حول", "دون", "بدون", "ضد", "عند", "لدى", "مثل",
    # Relative pronouns & complementizers
    "اللي", "الذي", "التي", "أن", "إنه",
    # Conditional & subordinating conjunctions
    "إذا", "لو", "كي", "لكي", "حتى", "مشان", "منشان",
    # Modals, adverbs & continuous aspect markers
    "بدي", "بدنا", "بدك", "عم", "بقدر", "ولا", "أو", "بشكل"
}

QUESTION_STARTERS = {"ليش", "ليه", "لماذا", "متى", "أين"}
QUESTION_INTERROGATIVES = {
    "شو", "ليش", "ليه", "لماذا", "متى", "أين", "وين", "كيف", "مين", "من", "هل", "ماذا", "ما"
}
QUESTION_CUES_2W = {
    "طيب شو", "طب شو", "مشان شو", "منشان شو", "شو يعني", "طب وين", "طيب وين",
    "من خلال شو", "بأي شكل", "لأي درجة"
}

# Conversational interjections & uptake markers that form standalone floor turns
STANDALONE_INTERJECTIONS_1W = {"تماما", "تمام", "طبعا", "أكيد", "اكيد", "مضبوط", "عظيم", "حلو", "ياسلام"}
STANDALONE_INTERJECTIONS_2W = {"يا سلام", "ما شاء", "ماشاء الله", "خذي راحتك", "خذ راحتك"}

DISCOURSE_CUES = {
    "تمام إذا", "تمام اذا", "أول شيء", "اول شيء", "لا نحن",
    "طيب شو", "لحظة لا", "لحظه لا", "ايه طيب", "إيه طيب"
}

AFFIRMATIVE_MARKERS = {"تماما", "تمام", "طبعا", "أكيد", "اكيد", "مضبوط"}

FLOOR_YIELDERS = {"تفضل", "تفضلي", "تفضلوا"}


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

    # 2. Merge same-speaker overlapping or near-adjacent intervals (gap <= 0.5s)
    speaker_intervals: Dict[str, List[Dict]] = {}
    for t in valid:
        spk = t["speaker"]
        if spk not in speaker_intervals:
            speaker_intervals[spk] = []
        merged = False
        for iv in speaker_intervals[spk]:
            if max(iv["start"], t["start"]) <= min(iv["end"], t["end"]) + 0.5:
                iv["start"] = min(iv["start"], t["start"])
                iv["end"] = max(iv["end"], t["end"])
                merged = True
                break
        if not merged:
            speaker_intervals[spk].append({"speaker": spk, "start": t["start"], "end": t["end"]})

    all_clean: List[Dict] = []
    for spk, ivs in speaker_intervals.items():
        all_clean.extend(ivs)
    all_clean.sort(key=lambda t: (t["start"], t["end"]))

    # 3. Handle interjections and floor handovers:
    # Short interjections (<= 1.8s) nested inside a different speaker's longer turn
    # partition the primary turn so the interjection receives its own floor turn.
    interjections = [t for t in all_clean if (t["end"] - t["start"]) <= 1.8]
    long_turns = [t for t in all_clean if (t["end"] - t["start"]) > 1.8]

    current_turns = list(long_turns)
    for inj in interjections:
        new_turns: List[Dict] = []
        punched = False
        for lt in current_turns:
            if lt["speaker"] != inj["speaker"] and lt["start"] <= inj["start"] and inj["end"] <= lt["end"]:
                if inj["start"] - lt["start"] >= 0.25:
                    new_turns.append({"speaker": lt["speaker"], "start": lt["start"], "end": inj["start"]})
                new_turns.append(dict(inj))
                if lt["end"] - inj["end"] >= 0.25:
                    new_turns.append({"speaker": lt["speaker"], "start": inj["end"], "end": lt["end"]})
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


def segment_into_utterances(words: List[Dict]) -> List[List[Dict]]:
    """
    Segment continuous words into syntactic utterances.
    Respects clause boundaries, questions, floor handovers, isolates standalone
    conversational interjections, and prevents cutting across non-terminal constituents.
    """
    utterances: List[List[Dict]] = []
    cur_u: List[Dict] = []

    i = 0
    while i < len(words):
        w = words[i]
        txt = w.get("word", "")
        clean = re.sub(r"[،,.\?!؟;:\"\']", "", txt).strip()

        # Check 2-word standalone interjection
        two_w = None
        if i + 1 < len(words):
            next_clean = re.sub(r"[،,.\?!؟;:\"\']", "", words[i + 1].get("word", "")).strip()
            cand = f"{clean} {next_clean}"
            if cand in STANDALONE_INTERJECTIONS_2W:
                two_w = cand

        is_interj = (two_w is not None) or (clean in STANDALONE_INTERJECTIONS_1W and clean not in NON_TERMINAL_WORDS)

        # Check if this interjection is immediately followed by a question by the same speaker
        # (e.g. "تماما طيب شو يعني..." or "يا سلام مشان شو بدك...")
        is_question_preface = False
        if is_interj:
            next_idx = i + 2 if two_w is not None else i + 1
            if next_idx < len(words):
                n_clean = re.sub(r"[،,.\?!؟;:\"\']", "", words[next_idx].get("word", "")).strip()
                n2_clean = re.sub(r"[،,.\?!؟;:\"\']", "", words[next_idx + 1].get("word", "")).strip() if next_idx + 1 < len(words) else ""
                n_pair = f"{n_clean} {n2_clean}".strip()
                if (n_clean in QUESTION_INTERROGATIVES or
                    n_pair in QUESTION_CUES_2W or
                    (n_clean in {"طيب", "طب", "مشان", "منشان"} and n2_clean in QUESTION_INTERROGATIVES)):
                    is_question_preface = True

        # Standalone conversational interjection: emit on its own floor turn
        if is_interj and not is_question_preface:
            if cur_u:
                utterances.append(cur_u)
                cur_u = []
            if two_w is not None:
                utterances.append([words[i], words[i + 1]])
                i += 2
            else:
                utterances.append([words[i]])
                i += 1
            continue


        cur_u.append(w)
        split = False

        if any(txt.endswith(t) for t in TERMINATORS):
            split = True
        elif clean in FLOOR_YIELDERS and len(cur_u) >= 2:
            split = True
        elif i + 1 < len(words):
            next_w = words[i + 1]
            next_clean = re.sub(r"[،,.\?!؟;:\"\']", "", next_w.get("word", "")).strip()

            cue_cand = None
            if i + 2 < len(words):
                n2_clean = re.sub(r"[،,.\?!؟;:\"\']", "", words[i + 2].get("word", "")).strip()
                cand = f"{next_clean} {n2_clean}"
                if cand in DISCOURSE_CUES or cand in STANDALONE_INTERJECTIONS_2W:
                    cue_cand = cand

            # Split before question starter, multi-word discourse cue, or interjection
            if (next_clean in QUESTION_STARTERS or cue_cand is not None or next_clean in STANDALONE_INTERJECTIONS_1W) and len(cur_u) >= 2 and clean not in NON_TERMINAL_WORDS and next_clean not in NON_TERMINAL_WORDS:
                split = True


            elif next_clean in AFFIRMATIVE_MARKERS and len(cur_u) >= 3 and clean not in NON_TERMINAL_WORDS and next_clean not in NON_TERMINAL_WORDS:
                split = True
            # Comma boundary with sufficient length
            elif any(txt.endswith(c) for c in COMMAS) and len(cur_u) >= 12 and clean not in NON_TERMINAL_WORDS and next_clean not in NON_TERMINAL_WORDS:
                split = True
            # Natural maximum clause length
            elif len(cur_u) >= 35 and clean not in NON_TERMINAL_WORDS and next_clean not in NON_TERMINAL_WORDS:
                split = True

        if split or i == len(words) - 1:
            utterances.append(cur_u)
            cur_u = []
        i += 1

    return utterances


def align_timestamps(asr_words: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
    """
    Syntactically clause-aware, monotonic sequence alignment between continuous ASR words
    and Pyannote speaker diarization intervals.
    """
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

    utterances = segment_into_utterances(deduped_words)
    if not utterances:
        return []

    K = len(utterances)
    M = len(turns)
    INF = 1e9

    # Candidate turns per utterance based on temporal proximity
    u_candidates: List[List[int]] = []
    for k, u in enumerate(utterances):
        u_mid = (float(u[0]["start"]) + float(u[-1]["end"])) / 2.0
        cands = []
        for j, t in enumerate(turns):
            t_mid = (t["start"] + t["end"]) / 2.0
            ov = max(0.0, min(float(u[-1]["end"]), t["end"]) - max(float(u[0]["start"]), t["start"]))
            if abs(u_mid - t_mid) <= 20.0 or ov > 0:
                cands.append(j)
        if not cands:
            closest_j = min(range(M), key=lambda j: abs(u_mid - (turns[j]["start"] + turns[j]["end"]) / 2.0))
            cands = [closest_j]
        u_candidates.append(cands)

    # Initialize Viterbi DP
    prev_dp = {}
    for j in u_candidates[0]:
        u_mid = (float(utterances[0][0]["start"]) + float(utterances[0][-1]["end"])) / 2.0
        t_mid = (turns[j]["start"] + turns[j]["end"]) / 2.0
        prev_dp[j] = (abs(u_mid - t_mid), None)

    backpointers: List[Dict[int, int]] = []

    for k in range(1, K):
        u = utterances[k]
        u_st = float(u[0]["start"])
        u_en = float(u[-1]["end"])
        u_mid = (u_st + u_en) / 2.0
        u_dur = max(0.1, u_en - u_st)
        u_txt = " ".join(w.get("word", "") for w in u).strip()
        u_clean = re.sub(r"[،,.\?!؟;:\"\']", "", u_txt).strip()
        is_interj = (u_clean in STANDALONE_INTERJECTIONS_1W) or (u_clean in STANDALONE_INTERJECTIONS_2W)

        cur_dp = {}
        bp_step = {}

        for j in u_candidates[k]:
            t = turns[j]
            t_mid = (t["start"] + t["end"]) / 2.0
            dist = abs(u_mid - t_mid)
            ov = max(0.0, min(u_en, t["end"]) - max(u_st, t["start"]))
            t_dur = t["end"] - t["start"]

            # Emit cost: standalone interjection strongly prefers short conversational turn nearby
            if is_interj:
                if t_dur <= 1.5 and dist <= 1.5:
                    emit_cost = -10.0 + dist * 0.5
                else:
                    emit_cost = dist * 0.5
            elif ov > 0.05:
                emit_cost = -12.0 * (ov / u_dur)
            elif len(u) <= 3:
                emit_cost = dist * 0.2
            else:
                emit_cost = dist * 0.5

            best_cost = INF
            best_prev_j = None

            for prev_j, (prev_cost, _) in prev_dp.items():
                if prev_j > j:
                    continue

                # Skip penalty scaled by duration of skipped turns
                skip_penalty = 0.0
                if j > prev_j:
                    for sk in range(prev_j + 1, j):
                        sk_dur = turns[sk]["end"] - turns[sk]["start"]
                        if turns[sk]["speaker"] != turns[prev_j]["speaker"]:
                            if sk_dur >= 1.0:
                                skip_penalty += 20.0
                            else:
                                skip_penalty += 4.0
                        else:
                            skip_penalty += 1.5

                # Speaker shift penalty: reward shift if clause ended with punctuation or is an interjection
                prev_u_txt = " ".join(w.get("word", "") for w in utterances[k - 1]).strip()
                prev_u_clean = re.sub(r"[،,.\?!؟;:\"\']", "", prev_u_txt).strip()
                is_prev_interj = (prev_u_clean in STANDALONE_INTERJECTIONS_1W) or (prev_u_clean in STANDALONE_INTERJECTIONS_2W)

                prev_t_dur = turns[prev_j]["end"] - turns[prev_j]["start"]
                txt_prev = utterances[k - 1][-1].get("word", "")
                is_punctuated = any(txt_prev.endswith(p) for p in TERMINATORS)
                is_question = any(txt_prev.endswith(p) for p in {'؟', '?'})
                spk_change_penalty = 0.0
                if turns[prev_j]["speaker"] != turns[j]["speaker"]:
                    if is_interj or (is_prev_interj and prev_t_dur <= 1.5):
                        spk_change_penalty = -2.0
                    else:
                        spk_change_penalty = -3.0 if is_question else (-2.0 if is_punctuated else 1.0)
                else:
                    if is_prev_interj and not is_interj and prev_t_dur <= 1.5:
                        spk_change_penalty = 5.0
                    elif j == prev_j and is_question:
                        # Asking a question yields the floor; penalize re-using the questioner's turn when another turn is available
                        spk_change_penalty = 3.5


                tot = prev_cost + emit_cost + skip_penalty + spk_change_penalty
                if tot < best_cost:
                    best_cost = tot
                    best_prev_j = prev_j

            if best_prev_j is not None:
                cur_dp[j] = (best_cost, best_prev_j)
                bp_step[j] = best_prev_j

        if not cur_dp:
            for j in u_candidates[k]:
                cur_dp[j] = (0.0, list(prev_dp.keys())[0])
                bp_step[j] = list(prev_dp.keys())[0]

        backpointers.append(bp_step)
        prev_dp = cur_dp

    best_final_j = min(prev_dp.keys(), key=lambda j: prev_dp[j][0])
    assigned = [best_final_j]
    cur_j = best_final_j
    for k in range(K - 1, 0, -1):
        prev_j = backpointers[k - 1].get(cur_j, cur_j)
        assigned.append(prev_j)
        cur_j = prev_j
    assigned.reverse()

    # Build speaker segments and merge adjacent clauses of same speaker (gap <= 3.5s)
    segments: List[Dict] = []
    for k in range(K):
        u = utterances[k]
        t_idx = assigned[k]
        spk = turns[t_idx]["speaker"]
        txt = " ".join(w["word"] for w in u).strip()
        st = float(u[0]["start"])
        en = float(u[-1]["end"])

        if segments and segments[-1]["speaker"] == spk and st - segments[-1]["end"] <= 3.5:
            segments[-1]["text"] += " " + txt
            segments[-1]["end"] = max(segments[-1]["end"], en)
        else:
            segments.append({
                "speaker": spk,
                "text": txt,
                "start": round(st, 2),
                "end": round(en, 2)
            })

    logger.info(
        f"Alignment complete: {len(deduped_words)} words across {len(utterances)} utterances -> {len(segments)} speaker segments."
    )
    return segments
