#!/usr/bin/env python3
"""
Runner for Pyannote Speaker Diarization + Cohere Transcribe Arabic ASR.
Uses tuned Pyannote 3.1 parameters (threshold=0.75, min_duration_off=0.45s)
and offline GPU-accelerated Cohere ASR (GGUF via transcribe_cpp / Vulkan).
Tests both standard test audio files:
  1. audio_nlp_1_5min.wav
  2. 20261011_133306.m4a
"""
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = WORKSPACE_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure venv packages and torch DLLs are in path
VENV_SITE = BACKEND_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

TORCH_LIB = VENV_SITE / "torch" / "lib"
if TORCH_LIB.exists():
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(TORCH_LIB))
        except Exception:
            pass
    if str(TORCH_LIB) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(TORCH_LIB) + ";" + os.environ.get("PATH", "")

from app.services.diarization import diarize, invalidate_pyannote_cache
from app.services.audio_processor import get_ffmpeg_binary
from app.models.asr import get_asr_engine
from app.config import settings

OUTPUT_DIR = Path(__file__).resolve().parent
COHERE_MODEL_NAME = "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"


def prepare_16k_wav(audio_path: Path) -> Path:
    """
    Ensure the audio file is converted to a clean 16kHz mono WAV for optimal
    neural feature extraction (speaker embeddings + acoustic model).
    """
    suffix = audio_path.suffix.lower()
    target_wav = OUTPUT_DIR / f"{audio_path.stem}_16k.wav"

    if suffix == ".wav":
        # Check if already 16kHz mono, otherwise resample cleanly
        return audio_path

    print(f"[*] Preprocessing {audio_path.name} -> 16kHz mono WAV...")
    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(audio_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        str(target_wav),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return target_wav


def run_pyannote_diarization(audio_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run Pyannote speaker diarization with tuned parameters.
    """
    print(f"[*] Running Pyannote Diarization (tuned threshold=0.75) for {audio_path.name}...")
    t0 = time.time()

    def on_progress(pct, msg):
        print(f"    [{pct:3d}%] {msg}", flush=True)

    turns = diarize(audio_path, num_speakers=num_speakers, on_progress=on_progress)
    elapsed = time.time() - t0
    speakers = sorted(list(set(t["speaker"] for t in turns)))
    print(f"[+] Pyannote Diarization completed in {elapsed:.2f}s ({len(speakers)} speakers: {speakers})")
    return turns


def run_cohere_transcription(audio_path: Path, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run Cohere Transcribe Arabic ASR on the diarized turns.
    """
    print(f"[*] Loading Cohere Transcribe Arabic engine ({COHERE_MODEL_NAME})...")
    engine = get_asr_engine(COHERE_MODEL_NAME)
    engine.load()

    def on_progress(pct, msg):
        print(f"    [{pct:3d}%] {msg}", flush=True)

    print(f"[*] Transcribing {len(turns)} turns with Cohere ASR...")
    t0 = time.time()
    try:
        segments = engine.transcribe_turns(
            audio_path,
            turns,
            on_progress=on_progress,
            progress_range=(0, 100)
        )
    finally:
        engine.unload()

    elapsed = time.time() - t0
    print(f"[+] Cohere ASR completed in {elapsed:.2f}s ({len(segments)} segments transcribed)")

    # Consolidate consecutive utterances of the same speaker
    merged = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = s.get("speaker", "SPEAKER_00")
        st = float(s.get("start", 0.0))
        en = float(s.get("end", 0.0))

        if merged and merged[-1]["speaker"] == spk and (st - merged[-1]["end"]) <= 2.0:
            merged[-1]["end"] = max(merged[-1]["end"], en)
            merged[-1]["text"] += " " + text
        else:
            merged.append({
                "speaker": spk,
                "start": round(st, 2),
                "end": round(en, 2),
                "text": text,
            })

    return merged


def process_audio_file(audio_path: Path):
    print("\n" + "=" * 70)
    print(f"PROCESSING: {audio_path.name} (Cohere + Tuned Pyannote)")
    print("=" * 70)

    if not audio_path.exists():
        print(f"[!] Error: File not found: {audio_path}")
        return

    # Invalidate cache so any updated config.yaml parameters take immediate effect
    invalidate_pyannote_cache()

    # Preprocess audio to 16kHz mono WAV
    audio_16k = prepare_16k_wav(audio_path)

    # 1. Pyannote Diarization (with tuned parameters)
    diar_turns = run_pyannote_diarization(audio_16k)

    # 2. Cohere ASR Transcription
    segments = run_cohere_transcription(audio_16k, diar_turns)

    # 3. Save results
    base_name = audio_path.stem
    output_json = OUTPUT_DIR / f"{base_name}_cohere_pyannote.json"
    output_txt = OUTPUT_DIR / f"{base_name}_cohere_pyannote.txt"

    full_text = " ".join(s["text"] for s in segments)
    payload = {
        "audio_file": audio_path.name,
        "engine": "Cohere Transcribe Arabic (Q5_K_M GGUF)",
        "diarization_engine": "Pyannote 3.1 (Tuned threshold=0.75)",
        "speakers_detected": sorted(list(set(s["speaker"] for s in segments))),
        "turns_count": len(segments),
        "full_text": full_text,
        "segments": segments
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Audio File: {audio_path.name}\n")
        f.write(f"ASR Model: {COHERE_MODEL_NAME}\n")
        f.write(f"Diarization: Pyannote 3.1 (Tuned: threshold=0.75, min_dur=0.15s)\n")
        f.write(f"Speakers Detected: {payload['speakers_detected']}\n")
        f.write("=" * 60 + "\n\n")
        f.write("--- DIARIZED TRANSCRIPT ---\n\n")
        for turn in segments:
            f.write(f"[{turn['speaker']}] ({turn['start']:.2f}s -> {turn['end']:.2f}s):\n{turn['text']}\n\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("--- FULL COHERE ASR TEXT ---\n\n")
        f.write(full_text + "\n")

    print(f"\n[✓] Results saved:")
    print(f"    - JSON: {output_json}")
    print(f"    - TXT:  {output_txt}")

    # 4. Display formatted dialogue output
    print("\n" + "-" * 70)
    print(f"DIARIZED TRANSCRIPT OUTPUT ({audio_path.name}):")
    print("-" * 70)
    for idx, turn in enumerate(segments, 1):
        print(f"{idx:02d}. [{turn['speaker']}] ({turn['start']:.2f}s - {turn['end']:.2f}s):")
        print(f"    {turn['text']}")
    print("-" * 70)


def main():
    test_files = [
        WORKSPACE_ROOT / "audio_nlp_1_5min.wav",
        WORKSPACE_ROOT / "20261011_133306.m4a"
    ]

    for audio_file in test_files:
        process_audio_file(audio_file)

    print("\n" + "=" * 70)
    print("ALL COHERE + PYANNOTE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
