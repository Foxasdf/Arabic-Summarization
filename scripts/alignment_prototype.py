"""Offline alignment prototype (frozen_v1 step 2).
Isolates TIMING from WER: target words = reference words (perfect hypothesis),
so any speaker-assignment error is pure timing error.
Compares Whisper-anchored DP vs uniform-split baseline on 3 boundary-crossing windows.
Run: .venv/Scripts/python.exe ../scripts/alignment_prototype.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer  # noqa: E402
from app.services import forced_alignment as FA  # noqa: E402
from app.services import alignment as AL  # noqa: E402
from app.services.audio_processor import get_ffmpeg_binary  # noqa: E402

AUDIO = ROOT / "20261011_133306.m4a"
REF_JSON = ROOT / "transcription_20261011_133306.json"
OUT = ROOT / "eval_results" / "frozen_v1" / "alignment_prototype.json"

WINDOWS = [(10.0, 25.0), (70.0, 90.0), (300.0, 320.0)]

def ref_words_in_window(segments, ws, we):
    words = []  # (word, true_speaker)
    for s in segments:
        st, en, spk = float(s["start"]), float(s["end"]), s["speaker"]
        if en <= ws or st >= we:
            continue
        for w in s["text"].split():
            words.append((w, spk))
    return words

def extract_slice(ws, we, dst):
    ff = get_ffmpeg_binary()
    dur = we - ws
    cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{ws:.2f}", "-t", f"{dur:.2f}",
           "-i", str(AUDIO), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0 and dst.exists()

def whisper_ref_words(wav_path):
    import numpy as np
    import faster_whisper
    from scipy.io import wavfile
    from app.config import settings
    wdir = settings.MODELS_DIR / getattr(settings, "WHISPER_DIALECTAL_DIRNAME",
                                         "whisper-large-v3-turbo-arabic-dialectal-v2-ct2")
    sr, raw = wavfile.read(str(wav_path))
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    if raw.dtype == np.int16:
        audio = (raw.astype(np.float32) / 32768.0).copy()
    elif raw.dtype == np.int32:
        audio = (raw.astype(np.float32) / 2147483648.0).copy()
    else:
        audio = raw.astype(np.float32).copy()
    model = faster_whisper.WhisperModel(str(wdir), device="cuda", compute_type="float16")
    try:
        segs, _ = model.transcribe(audio, language="ar", word_timestamps=True,
                                   beam_size=5, temperature=0.0,
                                   condition_on_previous_text=False)
        return [w for s in segs for w in (s.words or [])]
    finally:
        del model

def speaker_at(mid, turns):
    for t in turns:
        if float(t["start"]) <= mid <= float(t["end"]):
            return t["speaker"]
    return None

def accuracy(assigned, turns):
    # assigned: [{word,start,end,speaker}], truth from turns at midpoint
    ok, tot, near_ok, near_tot = 0, 0, 0, 0
    for w in assigned:
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        truth = speaker_at(mid, turns)
        if truth is None:
            continue
        tot += 1
        if w["speaker"] == truth:
            ok += 1
        # near-boundary: within 1.0s of any turn edge
        d_edge = min([abs(mid - float(t["start"])) for t in turns] + [abs(mid - float(t["end"])) for t in turns])
        if d_edge <= 1.0:
            near_tot += 1
            if w["speaker"] == truth:
                near_ok += 1
    return (ok / tot if tot else 0.0), (near_ok / near_tot if near_tot else 0.0), tot, near_tot

def main():
    d = json.loads(REF_JSON.read_text(encoding="utf-8"))
    segments = d["segments"]
    normalizer = ArabicTextNormalizer()
    results = []
    with tempfile.TemporaryDirectory() as td:
        for ws, we in WINDOWS:
            pairs = ref_words_in_window(segments, ws, we)
            target = [w for w, _ in pairs]
            turns = [s for s in segments if float(s["end"]) > ws and float(s["start"]) < we]
            wav = Path(td) / f"win_{int(ws)}.wav"
            ok_ext = extract_slice(ws, we, wav)
            if not ok_ext:
                results.append({"window": [ws, we], "error": "ffmpeg extract failed"})
                continue
            ref_w = whisper_ref_words(wav)
            # DP align target->whisper ref, shift to absolute time
            pairs_ts = FA._align_tokens_dp(target, ref_w)
            # _align_tokens_dp returns times relative to slice; shift by ws
            dp_words = [{"word": target[i], "start": round(ws + st, 2), "end": round(ws + en, 2)}
                        for i, (st, en) in enumerate(pairs_ts)]
            # plausibility (same gates as forced_alignment.py)
            durs = [w["end"] - w["start"] for w in dp_words]
            avg_d = sum(durs) / max(1, len(durs))
            max_d = max(durs) if durs else 0.0
            implausible = (avg_d > 0.9 or max_d > 2.5)
            # match rate: normalized exact matches between target and whisper ref head
            tn = [FA._normalize_token(w) for w in target]
            rn = [FA._normalize_token(getattr(w, "word", "")) for w in ref_w]
            exact = sum(1 for a, b in zip(tn, rn) if a and a == b)
            # speaker assignment both paths
            dp_assigned = AL.assign_words_to_speakers(dp_words, turns)
            dp_acc, dp_near, tot, near_tot = accuracy(dp_assigned, turns)
            # uniform baseline across window span
            n = len(target)
            span = we - ws
            uni = [{"word": target[i], "start": round(ws + i * span / max(1, n), 2),
                    "end": round(ws + (i + 1) * span / max(1, n), 2)} for i in range(n)]
            uni_assigned = AL.assign_words_to_speakers(uni, turns)
            u_acc, u_near, _, _ = accuracy(uni_assigned, turns)
            # boundary MAE: for each true turn edge in window, error of nearest word edge
            import numpy as np
            bounds = sorted(set([float(t["start"]) for t in turns if ws < float(t["start"]) < we] +
                                [float(t["end"]) for t in turns if ws < float(t["end"]) < we]))
            def boundary_mae(words):
                errs = []
                edges = sorted(set([float(w["start"]) for w in words] + [float(w["end"]) for w in words]))
                for b in bounds:
                    if edges:
                        errs.append(min(abs(e - b) for e in edges))
                return round(sum(errs) / max(1, len(errs)), 3) if errs else None
            dp_bmae = boundary_mae(dp_words)
            uni_bmae = boundary_mae(uni)
            # hypothesis divergence: whisper text vs reference text WER in window
            hyp_text = " ".join(getattr(w, "word", "") for w in ref_w)
            tgt_text = " ".join(target)
            div = calculate_wer_cer(tgt_text, hyp_text, normalizer)
            results.append({
                "window": [ws, we], "n_target": len(target), "n_whisper_ref": len(ref_w),
                "exact_head_match": exact, "exact_rate": round(exact / max(1, min(len(target), len(ref_w))), 3),
                "whisper_vs_ref_norm_wer": div["norm_wer"], "whisper_vs_ref_norm_cer": div["norm_cer"],
                "avg_dur": round(avg_d, 3), "max_dur": round(max_d, 3), "implausible": implausible,
                "dp_acc": round(dp_acc, 3), "dp_near_boundary_acc": round(dp_near, 3),
                "uni_acc": round(u_acc, 3), "uni_near_boundary_acc": round(u_near, 3),
                "dp_boundary_mae": dp_bmae, "uni_boundary_mae": uni_bmae,
                "n_scored": tot, "n_near": near_tot,
            })
            print(f"[{ws}-{we}] tgt={len(target)} ref={len(ref_w)} exact={exact} "
                  f"whisperWER={div['norm_wer']:.3f} avg_d={avg_d:.2f} max_d={max_d:.2f} impl={implausible} "
                  f"DP_bmae={dp_bmae} UNI_bmae={uni_bmae} DP_acc={dp_acc:.3f}/near={dp_near:.3f} UNI_acc={u_acc:.3f}/near={u_near:.3f}")
    OUT.write_text(json.dumps({"windows": results,
                               "conclusion_rule": "Integrate Whisper-DP only if DP near-boundary acc beats uniform by >=0.05 with implausible==False on >=2/3 windows, else keep uniform+low-confidence flag."},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
