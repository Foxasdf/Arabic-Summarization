#!/usr/bin/env python3
"""
Runner for Speaker Diarization (Pyannote 3.1 / NVIDIA Nemotron-3) + Cohere Transcribe Arabic ASR.
Supports switching between `pyannote` and `nemotron_3` via `--backend` CLI flag.
"""
import argparse
import gc
import json
import os
import sys
import time
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

# Ensure venv packages and torch DLLs are in path across Windows and Linux/macOS
_venv_candidates = [BACKEND_DIR / ".venv" / "Lib" / "site-packages"]
if (BACKEND_DIR / ".venv" / "lib").exists():
    _venv_candidates.extend((BACKEND_DIR / ".venv" / "lib").glob("python*/site-packages"))
for _venv_site in _venv_candidates:
    if _venv_site.exists() and str(_venv_site) not in sys.path:
        sys.path.insert(0, str(_venv_site))
    _torch_lib = _venv_site / "torch" / "lib"
    if _torch_lib.exists():
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_torch_lib))
            except Exception:
                pass
        if str(_torch_lib) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

from app.services.diarization import (
    diarize,
    diarization_backend_name,
    unload_diarization_models,
)
from app.services.audio_processor import ensure_16k_mono_wav
from app.models.asr import get_asr_engine
from app.config import settings

OUTPUT_DIR = Path(__file__).resolve().parent
COHERE_MODEL_NAME = "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"


def prepare_16k_wav(audio_path: Path) -> Path:
    """
    Ensure the audio file is converted to a verified 16 kHz mono float32/PCM WAV
    for optimal neural feature extraction (speaker embeddings + acoustic model).
    """
    target_wav = OUTPUT_DIR / f"{audio_path.stem}_16k.wav"
    print(f"[*] Verifying / converting {audio_path.name} -> 16kHz mono WAV...")
    return ensure_16k_mono_wav(audio_path, target_wav_path=target_wav)


def run_speaker_diarization(
    audio_path: Path,
    backend: str = "pyannote",
    num_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run speaker diarization using the selected backend (`pyannote` or `nemotron_3`).
    """
    label = diarization_backend_name(backend)
    print(f"[*] Running {label} Diarization for {audio_path.name}...")
    t0 = time.time()

    def on_progress(pct, msg):
        print(f"    [{pct:3d}%] {msg}", flush=True)

    turns = diarize(
        audio_path,
        num_speakers=num_speakers,
        on_progress=on_progress,
        backend=backend,
    )
    elapsed = time.time() - t0
    speakers = sorted(list(set(t["speaker"] for t in turns)))
    print(f"[+] {label} completed in {elapsed:.2f}s ({len(speakers)} speakers: {speakers})")
    return turns


def run_cohere_transcription(
    audio_path: Path,
    turns: List[Dict[str, Any]],
    engine: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Run Cohere Transcribe Arabic ASR on the diarized turns.
    Reuses an existing loaded `engine` when provided to avoid reloading inside loops.
    """
    own_engine = engine is None
    if own_engine:
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
            progress_range=(0, 100),
        )
    finally:
        if own_engine:
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


def process_audio_file(
    audio_path: Path,
    backend: str = "pyannote",
    engine: Optional[Any] = None,
):
    label = diarization_backend_name(backend)
    print("\n" + "=" * 70)
    print(f"PROCESSING: {audio_path.name} (Cohere + {label})")
    print("=" * 70)

    if not audio_path.exists():
        print(f"[!] Error: File not found: {audio_path}")
        return

    # Preprocess audio to verified 16kHz mono WAV
    audio_16k = prepare_16k_wav(audio_path)

    # 1. Speaker Diarization
    diar_turns = run_speaker_diarization(audio_16k, backend=backend)

    # 2. Cohere ASR Transcription
    segments = run_cohere_transcription(audio_16k, diar_turns, engine=engine)

    # 3. Save results
    base_name = audio_path.stem
    suffix = "pyannote" if backend == "pyannote" else "nemotron_3"
    output_json = OUTPUT_DIR / f"{base_name}_cohere_{suffix}.json"
    output_txt = OUTPUT_DIR / f"{base_name}_cohere_{suffix}.txt"

    full_text = " ".join(s["text"] for s in segments)
    payload = {
        "audio_file": audio_path.name,
        "engine": "Cohere Transcribe Arabic (Q5_K_M GGUF)",
        "diarization_backend": backend,
        "diarization_engine": label,
        "speakers_detected": sorted(list(set(s["speaker"] for s in segments))),
        "turns_count": len(segments),
        "full_text": full_text,
        "segments": segments,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Audio File: {audio_path.name}\n")
        f.write(f"ASR Model: {COHERE_MODEL_NAME}\n")
        f.write(f"Diarization: {label}\n")
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
    print(f"DIARIZED TRANSCRIPT OUTPUT ({audio_path.name} - {label}):")
    print("-" * 70)
    for idx, turn in enumerate(segments, 1):
        print(f"{idx:02d}. [{turn['speaker']}] ({turn['start']:.2f}s - {turn['end']:.2f}s):")
        print(f"    {turn['text']}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run Speaker Diarization + Cohere Arabic ASR")
    parser.add_argument(
        "--backend",
        choices=["pyannote", "nemotron_3", "both"],
        default=getattr(settings, "DIARIZATION_BACKEND", "pyannote"),
        help="Diarization backend to use (pyannote, nemotron_3, or both).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional audio file paths to process.",
    )
    args = parser.parse_args()

    if args.files:
        test_files = [Path(p) for p in args.files]
    else:
        test_files = [
            WORKSPACE_ROOT / "audio_nlp_1_5min.wav",
            WORKSPACE_ROOT / "20261011_133306.m4a",
        ]

    backends = ["pyannote", "nemotron_3"] if args.backend == "both" else [args.backend]

    for backend in backends:
        engine = get_asr_engine(COHERE_MODEL_NAME)
        engine.load()
        try:
            for audio_file in test_files:
                process_audio_file(audio_file, backend=backend, engine=engine)
        finally:
            engine.unload()
            unload_diarization_models(backend)
            gc.collect()

    print("\n" + "=" * 70)
    print("ALL COHERE + DIARIZATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
