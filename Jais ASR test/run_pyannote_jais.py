#!/usr/bin/env python3
"""
Simple Runner for Pyannote Speaker Diarization + Jais ASR API.
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
import requests

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path so we can reuse local offline Pyannote service
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = WORKSPACE_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.diarization import diarize
from app.services.audio_processor import get_ffmpeg_binary

# Configuration (reads from environment if set)
IFM_API_KEY = os.getenv("IFM_API_KEY", "")
IFM_API_URL = "https://api.ifm.ai/v1/audio/transcriptions"
IFM_MODEL = "IFM/Jais-ASR"
OUTPUT_DIR = Path(__file__).resolve().parent


def prepare_audio_for_jais(audio_path: Path) -> Path:
    """
    Ensure the audio file is in a format accepted by the Jais API (.wav or .mp3).
    If it's .m4a or unsupported container, convert to 16kHz mono .mp3 using ffmpeg.
    """
    suffix = audio_path.suffix.lower()
    if suffix in [".wav", ".mp3"]:
        return audio_path

    converted_path = OUTPUT_DIR / f"{audio_path.stem}_converted.mp3"
    print(f"[*] Converting {audio_path.name} to MP3 for Jais ASR API...")
    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(audio_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "128k",
        str(converted_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return converted_path


def run_jais_asr(audio_path: Path) -> Dict[str, Any]:
    """
    Send audio to IFM Jais-ASR transcription API.
    """
    audio_for_api = prepare_audio_for_jais(audio_path)
    content_type = "audio/wav" if audio_for_api.suffix.lower() == ".wav" else "audio/mpeg"

    headers = {
        "Authorization": f"Bearer {IFM_API_KEY}"
    }

    print(f"[*] Calling Jais ASR API ({IFM_MODEL}) for {audio_for_api.name}...")
    t0 = time.time()
    with open(audio_for_api, "rb") as f:
        files = {
            "file": (audio_for_api.name, f, content_type)
        }
        data = {
            "model": IFM_MODEL
        }
        resp = requests.post(IFM_API_URL, headers=headers, files=files, data=data, timeout=300)

    elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"Jais ASR API Error ({resp.status_code}): {resp.text}")

    result = resp.json()
    print(f"[+] Jais ASR completed in {elapsed:.2f}s (Detected language: {result.get('language')})")
    return result


def run_pyannote_diarization(audio_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run Pyannote speaker diarization using the local offline models.
    """
    print(f"[*] Running Pyannote Diarization for {audio_path.name}...")
    t0 = time.time()

    def on_progress(pct, msg):
        print(f"    [{pct:3d}%] {msg}", flush=True)

    turns = diarize(audio_path, num_speakers=num_speakers, on_progress=on_progress)
    elapsed = time.time() - t0
    speakers = sorted(list(set(t["speaker"] for t in turns)))
    print(f"[+] Pyannote Diarization completed in {elapsed:.2f}s ({len(speakers)} speakers: {speakers})")
    return turns


def align_segments_with_speakers(
    asr_segments: List[Dict[str, Any]],
    diar_turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assign each Jais ASR segment to the speaker with the maximum temporal overlap in Pyannote.
    Then merges consecutive segments of the same speaker.
    """
    if not asr_segments:
        return []
    if not diar_turns:
        return [{"speaker": "SPEAKER_00", "start": s["start"], "end": s["end"], "text": s["text"]} for s in asr_segments]

    aligned_segments = []
    for s in asr_segments:
        s_start = float(s.get("start", 0.0))
        s_end = float(s.get("end", 0.0))
        s_mid = (s_start + s_end) / 2.0
        text = s.get("text", "").strip()

        best_spk = None
        max_overlap = 0.0

        for t in diar_turns:
            t_start = float(t.get("start", 0.0))
            t_end = float(t.get("end", 0.0))
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = t.get("speaker")

        if not best_spk:
            # Fallback to midpoint containment or closest turn
            closest_dist = float("inf")
            for t in diar_turns:
                t_start = float(t.get("start", 0.0))
                t_end = float(t.get("end", 0.0))
                if t_start <= s_mid <= t_end:
                    best_spk = t.get("speaker")
                    break
                dist = min(abs(s_mid - t_start), abs(s_mid - t_end))
                if dist < closest_dist:
                    closest_dist = dist
                    best_spk = t.get("speaker")

        if not best_spk:
            best_spk = "SPEAKER_00"

        aligned_segments.append({
            "speaker": best_spk,
            "start": round(s_start, 2),
            "end": round(s_end, 2),
            "text": text
        })

    # Merge consecutive utterances of the same speaker
    merged = []
    for seg in aligned_segments:
        if not seg["text"]:
            continue
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))

    return merged


def process_audio_file(audio_path: Path):
    print("\n" + "=" * 70)
    print(f"PROCESSING: {audio_path.name}")
    print("=" * 70)

    if not audio_path.exists():
        print(f"[!] Error: File not found: {audio_path}")
        return

    # 1. Pyannote Diarization
    diar_turns = run_pyannote_diarization(audio_path)

    # 2. Jais ASR Transcription
    jais_res = run_jais_asr(audio_path)

    # 3. Align Diarization and ASR
    asr_segments = jais_res.get("segments", [])
    if not asr_segments and jais_res.get("text"):
        asr_segments = [{"start": 0.0, "end": jais_res.get("duration_seconds", 0.0), "text": jais_res.get("text")}]

    diarized_transcript = align_segments_with_speakers(asr_segments, diar_turns)

    # 4. Save results to disk
    base_name = audio_path.stem
    output_json = OUTPUT_DIR / f"{base_name}_jais_pyannote.json"
    output_txt = OUTPUT_DIR / f"{base_name}_jais_pyannote.txt"

    payload = {
        "audio_file": audio_path.name,
        "language": jais_res.get("language"),
        "duration_seconds": jais_res.get("duration_seconds"),
        "full_text": jais_res.get("text", ""),
        "diarization_raw_turns_count": len(diar_turns),
        "aligned_transcript": diarized_transcript
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Audio File: {audio_path.name}\n")
        f.write(f"Language: {jais_res.get('language')}\n")
        f.write(f"Duration: {jais_res.get('duration_seconds')}s\n")
        f.write("=" * 60 + "\n\n")
        f.write("--- DIARIZED TRANSCRIPT ---\n\n")
        for turn in diarized_transcript:
            f.write(f"[{turn['speaker']}] ({turn['start']:.2f}s -> {turn['end']:.2f}s):\n{turn['text']}\n\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("--- FULL JAIS ASR TEXT ---\n\n")
        f.write(jais_res.get("text", "") + "\n")

    print(f"\n[✓] Results saved:")
    print(f"    - JSON: {output_json}")
    print(f"    - TXT:  {output_txt}")

    # 5. Display formatted output
    print("\n" + "-" * 70)
    print(f"DIARIZED TRANSCRIPT OUTPUT ({audio_path.name}):")
    print("-" * 70)
    for idx, turn in enumerate(diarized_transcript, 1):
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
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
