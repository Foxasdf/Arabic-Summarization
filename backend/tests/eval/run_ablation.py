"""
Ablation Experiment Runner (Section 1 of Approved Plan).
Evaluates the 5 distinct conditions (A, B, C, D, E) against the Gold standard dataset:
Condition A: Raw audio, Continuous / VAD chunked
Condition B: Filtered (afftdn), Continuous / VAD chunked
Condition C: Raw audio, Chopped at Pyannote speaker turns
Condition D: Filtered (afftdn), Chopped at Pyannote speaker turns
Condition E: Raw audio, Chopped at Pyannote speaker turns with padding (0.75s) and acoustic context
"""
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path

from backend.app.models.asr.cohere_engine import CohereASREngine
from backend.tests.eval.metrics import compute_wer, compute_cer, compute_cpwer, normalize_arabic_text
from backend.tests.eval.gold_dataset import GOLD_TEST_SET

def apply_afftdn(input_wav: Path, output_wav: Path):
    """Apply balanced afftdn filter via FFmpeg."""
    cmd = [
        "backend/bin/ffmpeg.exe", "-y", "-loglevel", "error",
        "-i", str(input_wav),
        "-af", "afftdn=nf=-25:nr=10:tn=0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_wav)
    ]
    subprocess.run(cmd, check=True)

def run_ablation():
    gold = GOLD_TEST_SET["sample_meeting_45s"]
    raw_audio = Path(gold["audio_path"])
    ref_text = gold["ground_truth_text"]
    ref_segments = gold["ground_truth_segments"]
    turns = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"]} for s in ref_segments]

    print("=================================================================")
    print("        RUNNING ABLATION EXPERIMENT MATRIX (PHASE 1)             ")
    print("=================================================================")

    # Initialize Cohere engine
    eng = CohereASREngine()
    eng.load()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        filtered_audio = tmp_path / "filtered_afftdn.wav"
        apply_afftdn(raw_audio, filtered_audio)

        results = {}

        # -------------------------------------------------------------
        # Condition A: Raw audio, Continuous / VAD (~25s-45s)
        # -------------------------------------------------------------
        print("\n[Condition A] Raw audio, Continuous...")
        text_a = eng._transcribe_slice_internal(raw_audio, dur_sec=45.0).replace("@@@فراغ", "").strip()
        wer_a = compute_wer(ref_text, text_a)
        cer_a = compute_cer(ref_text, text_a)
        results["A"] = {"name": "Raw Continuous (VAD)", "text": text_a, "wer": wer_a, "cer": cer_a}
        print(f" -> WER: {wer_a*100:.2f}%, CER: {cer_a*100:.2f}%")

        # -------------------------------------------------------------
        # Condition B: Filtered (afftdn), Continuous
        # -------------------------------------------------------------
        print("\n[Condition B] Filtered (afftdn), Continuous...")
        text_b = eng._transcribe_slice_internal(filtered_audio, dur_sec=45.0).replace("@@@فراغ", "").strip()
        wer_b = compute_wer(ref_text, text_b)
        cer_b = compute_cer(ref_text, text_b)
        results["B"] = {"name": "Filtered (afftdn) Continuous", "text": text_b, "wer": wer_b, "cer": cer_b}
        print(f" -> WER: {wer_b*100:.2f}%, CER: {cer_b*100:.2f}%")

        # -------------------------------------------------------------
        # Condition C: Raw audio, Chopped at Pyannote speaker turns
        # -------------------------------------------------------------
        print("\n[Condition C] Raw audio, Chopped at speaker turns...")
        from scipy.io import wavfile
        sr, audio_pcm = wavfile.read(str(raw_audio))
        segs_c = []
        texts_c = []
        for i, t in enumerate(turns):
            s_idx = int(t["start"] * sr)
            e_idx = min(len(audio_pcm), int(t["end"] * sr))
            slice_file = tmp_path / f"c_turn_{i}.wav"
            wavfile.write(str(slice_file), sr, audio_pcm[s_idx:e_idx])
            dur = t["end"] - t["start"]
            t_txt = eng._transcribe_slice_internal(slice_file, dur_sec=dur).replace("@@@فراغ", "").strip()
            segs_c.append({**t, "text": t_txt})
            if t_txt:
                texts_c.append(t_txt)
        text_c = " ".join(texts_c)
        wer_c = compute_wer(ref_text, text_c)
        cer_c = compute_cer(ref_text, text_c)
        cpwer_c = compute_cpwer(ref_segments, segs_c)
        results["C"] = {"name": "Raw Chopped Turns", "text": text_c, "wer": wer_c, "cer": cer_c, "cpwer": cpwer_c}
        print(f" -> WER: {wer_c*100:.2f}%, CER: {cer_c*100:.2f}%, cpWER: {cpwer_c*100:.2f}%")

        # -------------------------------------------------------------
        # Condition D: Filtered (afftdn), Chopped at Pyannote speaker turns
        # -------------------------------------------------------------
        print("\n[Condition D] Filtered (afftdn), Chopped at speaker turns...")
        sr_f, audio_pcm_f = wavfile.read(str(filtered_audio))
        segs_d = []
        texts_d = []
        for i, t in enumerate(turns):
            s_idx = int(t["start"] * sr_f)
            e_idx = min(len(audio_pcm_f), int(t["end"] * sr_f))
            slice_file = tmp_path / f"d_turn_{i}.wav"
            wavfile.write(str(slice_file), sr_f, audio_pcm_f[s_idx:e_idx])
            dur = t["end"] - t["start"]
            t_txt = eng._transcribe_slice_internal(slice_file, dur_sec=dur).replace("@@@فراغ", "").strip()
            segs_d.append({**t, "text": t_txt})
            if t_txt:
                texts_d.append(t_txt)
        text_d = " ".join(texts_d)
        wer_d = compute_wer(ref_text, text_d)
        cer_d = compute_cer(ref_text, text_d)
        cpwer_d = compute_cpwer(ref_segments, segs_d)
        results["D"] = {"name": "Filtered Chopped Turns", "text": text_d, "wer": wer_d, "cer": cer_d, "cpwer": cpwer_d}
        print(f" -> WER: {wer_d*100:.2f}%, CER: {cer_d*100:.2f}%, cpWER: {cpwer_d*100:.2f}%")

        # -------------------------------------------------------------
        # Condition E: Raw audio, Chopped at Pyannote speaker turns with padding 0.75s
        # -------------------------------------------------------------
        print("\n[Condition E] Raw audio, Chopped with 0.75s padding...")
        segs_e = []
        texts_e = []
        for i, t in enumerate(turns):
            pad = 0.75
            s_pad = max(0.0, t["start"] - pad)
            e_pad = min(len(audio_pcm)/sr, t["end"] + pad)
            s_idx = int(s_pad * sr)
            e_idx = int(e_pad * sr)
            slice_file = tmp_path / f"e_turn_{i}.wav"
            wavfile.write(str(slice_file), sr, audio_pcm[s_idx:e_idx])
            dur = e_pad - s_pad
            t_txt = eng._transcribe_slice_internal(slice_file, dur_sec=dur).replace("@@@فراغ", "").strip()
            segs_e.append({**t, "text": t_txt})
            if t_txt:
                texts_e.append(t_txt)
        text_e = " ".join(texts_e)
        wer_e = compute_wer(ref_text, text_e)
        cer_e = compute_cer(ref_text, text_e)
        cpwer_e = compute_cpwer(ref_segments, segs_e)
        results["E"] = {"name": "Raw Chopped + 0.75s Padding", "text": text_e, "wer": wer_e, "cer": cer_e, "cpwer": cpwer_e}
        print(f" -> WER: {wer_e*100:.2f}%, CER: {cer_e*100:.2f}%, cpWER: {cpwer_e*100:.2f}%")

    eng.unload()

    # Print Summary Table
    print("\n" + "="*70)
    print("                 ABLATION MATRIX RESULTS SUMMARY                  ")
    print("="*70)
    print(f"{'Condition':<12} | {'Audio':<10} | {'Segmentation':<26} | {'WER':<8} | {'CER':<8}")
    print("-" * 70)
    print(f"{'A':<12} | {'Raw':<10} | {'Continuous / VAD':<26} | {wer_a*100:>5.1f}% | {cer_a*100:>5.1f}%")
    print(f"{'B':<12} | {'afftdn':<10} | {'Continuous / VAD':<26} | {wer_b*100:>5.1f}% | {cer_b*100:>5.1f}%")
    print(f"{'C':<12} | {'Raw':<10} | {'Chopped Pyannote Turns':<26} | {wer_c*100:>5.1f}% | {cer_c*100:>5.1f}%")
    print(f"{'D':<12} | {'afftdn':<10} | {'Chopped Pyannote Turns':<26} | {wer_d*100:>5.1f}% | {cer_d*100:>5.1f}%")
    print(f"{'E':<12} | {'Raw':<10} | {'Chopped Turns + 0.75s Pad':<26} | {wer_e*100:>5.1f}% | {cer_e*100:>5.1f}%")
    print("="*70)

    # Save to JSON
    out_file = Path("eval_results/ablation_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {out_file}")

if __name__ == "__main__":
    run_ablation()
