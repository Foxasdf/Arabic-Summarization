import sys
import time
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.diarization import diarize


def progress_callback(pct, msg):
    print(f"[{pct}%] {msg}", flush=True)


def verify_file(audio_path: Path, ref_json_path: Path, num_speakers=None):
    print("=" * 60)
    print(f"Testing diarization on: {audio_path.name}")
    print("=" * 60)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return False

    ref_speakers = {}
    if ref_json_path.exists():
        with open(ref_json_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
            ref_speakers = ref_data.get("speakers", {})
        print(f"Reference ground truth speakers: {list(ref_speakers.keys())} ({len(ref_speakers)} speakers)")
        for k, v in ref_speakers.items():
            print(f"  {k}: {v}")

    t0 = time.time()
    turns = diarize(audio_path, num_speakers=num_speakers, on_progress=progress_callback)
    elapsed = time.time() - t0

    detected_speakers = sorted(list(set(t["speaker"] for t in turns)))
    print(f"\n[Result] Diarization completed in {elapsed:.2f}s")
    print(f"[Result] Total normalized turns: {len(turns)}")
    print(f"[Result] Detected speakers: {detected_speakers} ({len(detected_speakers)} speakers)")

    # Analyze short turns (interjections <= 1.8s)
    interjections = [t for t in turns if (t["end"] - t["start"]) <= 1.8]
    print(f"[Result] Preserved conversational interjections (<= 1.8s): {len(interjections)}")

    # Print first 10 turns
    print("\nFirst 10 normalized turns:")
    for i, t in enumerate(turns[:10]):
        dur = t["end"] - t["start"]
        print(f"  Turn {i+1:02d}: [{t['speaker']}] {t['start']:.2f}s - {t['end']:.2f}s (dur: {dur:.2f}s)")

    # Check for overlapping turns (should be none or minimal at boundaries)
    overlaps = 0
    for i in range(len(turns) - 1):
        if turns[i]["end"] > turns[i+1]["start"] + 0.05:
            overlaps += 1
    print(f"\n[Check] Overlapping turns count: {overlaps}")

    # Check that speaker count matches expected
    if ref_speakers:
        expected_count = len(ref_speakers)
        actual_count = len(detected_speakers)
        match = actual_count == expected_count
        print(f"[Check] Speaker count match: expected={expected_count}, actual={actual_count} -> {'PASS' if match else 'DIFFERENT'}")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", choices=["all", "nlp", "133306"], default="all")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.file in ("nlp", "all"):
        verify_file(
            root / "audio_nlp_1_5min.wav",
            root / "transcription_nlp_5min.json",
        )

    if args.file in ("133306", "all"):
        verify_file(
            root / "20261011_133306.m4a",
            root / "transcription_20261011_133306.json",
        )
