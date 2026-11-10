"""
Unit tests verifying the Step 1 & Step 2 audit fixes:
1. Stereo int16 -> mono float32 normalization (no [-32768, 32767] scaling bug).
2. Turn merging & micro-segment handling (Refinement #1: absorb adjacent/enclosed <0.5s,
   preserve isolated 0.25s-0.50s turns in silence, drop <0.20s blips).
3. Boundary padding symmetry and enclosed mono overlap suppression in BaseASREngine.plan_windows.
4. Overlapping interval deduplication in fallback_relative_alignment.
5. Cohere special token stripping (@@@فراغ, @@@ضحك, @@@ضوضاء, ###...) in guardrails.
6. Nemotron-3 frame-level probability extraction into standardized [{'speaker', 'start', 'end'}] turns.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_processor import normalize_pcm_to_float32
from app.services.alignment import clean_diarization_turns
from app.services.forced_alignment import fallback_relative_alignment
from app.services.diarization import (
    _extract_segments_from_nemotron_probs,
    resolve_diarization_backend,
)
from app.models.asr.base import BaseASREngine
from app.models.asr.guardrails import clean_and_guard_asr_text


class TestPipelineAuditFixes(unittest.TestCase):
    def test_stereo_int16_normalization(self):
        """Stereo int16 PCM must normalize to [-1.0, 1.0] float32 mono before averaging."""
        left = np.full(1600, 16384, dtype=np.int16)
        right = np.full(1600, -16384, dtype=np.int16)
        stereo = np.stack([left, right], axis=1)
        mono = normalize_pcm_to_float32(stereo)
        self.assertEqual(mono.dtype, np.float32)
        self.assertEqual(mono.ndim, 1)
        self.assertTrue(np.allclose(mono, 0.0, atol=1e-4))

        loud_stereo = np.full((1600, 2), 16384, dtype=np.int16)
        loud_mono = normalize_pcm_to_float32(loud_stereo)
        self.assertTrue(np.all(np.abs(loud_mono) <= 1.0))
        self.assertAlmostEqual(float(np.mean(loud_mono)), 0.5, places=3)

    def test_clean_diarization_turns_refinement_1(self):
        """
        Refinement #1:
        - Drop isolated blips < 0.20s
        - Preserve isolated short replies (0.25s-0.50s) surrounded by silence
        - Absorb < 0.50s micro-segments when adjacent to same speaker or enclosed in dominant turn
        - Merge consecutive same-speaker turns when gap <= 0.4s
        """
        raw_turns = [
            # Isolated blip (< 0.20s) -> should be dropped
            {"speaker": "SPEAKER_01", "start": 0.50, "end": 0.62},
            # Isolated genuine short reply (0.35s in silence) -> MUST BE KEPT
            {"speaker": "SPEAKER_01", "start": 2.00, "end": 2.35},
            # Dominant SPEAKER_00 turn split by a 0.30s gap and containing an enclosed 0.30s backchannel
            {"speaker": "SPEAKER_00", "start": 5.00, "end": 8.00},
            {"speaker": "SPEAKER_01", "start": 6.20, "end": 6.50},  # enclosed <0.5s micro-segment
            {"speaker": "SPEAKER_00", "start": 8.30, "end": 11.00},  # 0.30s gap from 8.00 -> merge!
        ]
        cleaned = clean_diarization_turns(
            raw_turns,
            max_same_speaker_gap=0.4,
            min_duration=0.5,
            isolated_min_duration=0.25,
            drop_blip_duration=0.20,
            absorb_enclosed_micro=True,
        )
        self.assertEqual(len(cleaned), 2)
        # First turn is the preserved isolated short reply (2.00 - 2.35)
        self.assertEqual(cleaned[0]["speaker"], "SPEAKER_01")
        self.assertAlmostEqual(cleaned[0]["start"], 2.00)
        self.assertAlmostEqual(cleaned[0]["end"], 2.35)
        # Second turn is the merged SPEAKER_00 turn (5.00 - 11.00)
        self.assertEqual(cleaned[1]["speaker"], "SPEAKER_00")
        self.assertAlmostEqual(cleaned[1]["start"], 5.00)
        self.assertAlmostEqual(cleaned[1]["end"], 11.00)

    def test_plan_windows_suppresses_enclosed_overlap_and_splits_at_silence(self):
        """plan_windows suppresses enclosed mono sub-turns and splits long turns at low energy."""
        turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 30.0},
            {"speaker": "SPEAKER_01", "start": 10.0, "end": 10.3},  # <0.45s enclosed backchannel -> dropped
        ]
        sr = 16000
        # Create 30s audio with a silent valley around 14.2s - 14.5s
        audio = np.full(30 * sr, 0.2, dtype=np.float32)
        audio[int(14.2 * sr) : int(14.5 * sr)] = 0.0

        windows = BaseASREngine.plan_windows(turns, max_len=25.0, audio_data=audio, sr=sr)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["speaker"], "SPEAKER_00")
        self.assertEqual(windows[1]["speaker"], "SPEAKER_00")
        # Cut point should land inside the 14.2s - 14.5s silent region instead of 15.0s
        self.assertGreaterEqual(windows[0]["end"], 14.15)
        self.assertLessEqual(windows[0]["end"], 14.55)

    def test_fallback_relative_alignment_merges_overlaps(self):
        """Overlapping intervals must not double-count active duration or move timestamps backwards."""
        words = ["هذا", "اختبار", "لمطابقة", "الكلمات", "زمنيا"]
        intervals = [
            {"start": 0.0, "end": 10.0},
            {"start": 2.0, "end": 8.0},  # completely overlapping
        ]
        aligned = fallback_relative_alignment(words, total_duration=10.0, active_intervals=intervals)
        self.assertEqual(len(aligned), 5)
        starts = [w["start"] for w in aligned]
        self.assertEqual(starts, sorted(starts))
        self.assertLessEqual(aligned[-1]["end"], 10.0)

    def test_guardrails_strips_cohere_tokens(self):
        """Guardrails must strip @@@ضحك, @@@ضوضاء, @@@فراغ, and ### tags."""
        raw = "###السؤال. مرحبا بكم @@@ضحك في الاجتماع @@@ضوضاء @@@فراغ"
        cleaned = clean_and_guard_asr_text(raw, dur_sec=4.0)
        self.assertNotIn("@@@", cleaned)
        self.assertNotIn("###", cleaned)
        self.assertIn("مرحبا بكم", cleaned)

        # Short-clip script hallucination suppression
        script_hallucination = "- بدر: يا بدر"
        self.assertEqual(clean_and_guard_asr_text(script_hallucination, dur_sec=0.8), "")

    def test_nemotron_frame_extraction_and_backend_resolution(self):
        """Nemotron-3 [T, 8] frame probabilities convert cleanly to standardized turn dicts."""
        self.assertEqual(resolve_diarization_backend("pyannote"), "pyannote")
        self.assertEqual(resolve_diarization_backend("nemotron_3"), "nemotron_3")
        self.assertEqual(resolve_diarization_backend("nvidia/Nemotron-3-Diarization"), "nemotron_3")

        # 500 frames at 10ms = 5.0s
        probs = np.zeros((500, 8), dtype=np.float32)
        probs[50:200, 0] = 0.9   # 0.50s - 2.00s: speaker 0
        probs[250:450, 2] = 0.85  # 2.50s - 4.50s: speaker 2
        raw_segs = _extract_segments_from_nemotron_probs(probs, threshold=0.5, frame_duration=0.01)
        self.assertEqual(len(raw_segs), 2)
        self.assertEqual(raw_segs[0]["speaker"], "SPEAKER_00")
        self.assertAlmostEqual(raw_segs[0]["start"], 0.50, places=2)
        self.assertAlmostEqual(raw_segs[0]["end"], 2.00, places=2)
        self.assertEqual(raw_segs[1]["speaker"], "SPEAKER_02")
        self.assertAlmostEqual(raw_segs[1]["start"], 2.50, places=2)
        self.assertAlmostEqual(raw_segs[1]["end"], 4.50, places=2)

    def test_resolve_turn_overlaps_carves_and_deoverlaps(self):
        """Enclosed interjections are carved, blips dropped, duplicates deduped, handovers split."""
        from app.services.alignment import resolve_turn_overlaps

        turns = [
            {"speaker": "A", "start": 61.64, "end": 72.15},   # dominant
            {"speaker": "B", "start": 64.36, "end": 65.31},   # 0.95s enclosed question -> carve
            {"speaker": "B", "start": 68.00, "end": 68.30},   # 0.30s enclosed blip -> drop
            {"speaker": "C", "start": 107.91, "end": 108.66}, # duplicate pair -> keep longer
            {"speaker": "A", "start": 108.06, "end": 108.54},
            {"speaker": "A", "start": 33.86, "end": 60.78},   # 26.92s monologue overlapping 2.54s question (<3.0s, >=2x) -> trim A to 59.39
            {"speaker": "B", "start": 59.39, "end": 61.93},
        ]
        out = resolve_turn_overlaps(turns)
        for x, y in zip(out, out[1:]):
            self.assertLessEqual(x["end"], y["start"] + 1e-6)
        spans = [(t["speaker"], t["start"], t["end"]) for t in out]
        self.assertIn(("B", 64.36, 65.31), spans)
        self.assertIn(("A", 65.31, 72.15), spans)
        self.assertNotIn(("B", 68.0, 68.3), spans)
        self.assertIn(("C", 107.91, 108.66), spans)
        self.assertFalse(any(s == "A" and st == 108.06 for s, st, _ in spans))
        self.assertIn(("A", 33.86, 59.39), spans)
        self.assertIn(("B", 59.39, 61.78), spans)

        # Comparable turns (both >= 1.5s, prev < 2x curr) split at midpoint
        handover_mid = resolve_turn_overlaps([
            {"speaker": "A", "start": 10.00, "end": 14.00},
            {"speaker": "B", "start": 13.00, "end": 16.00},
        ])
        self.assertEqual(
            [(t["speaker"], t["start"], t["end"]) for t in handover_mid],
            [("A", 10.0, 13.5), ("B", 13.5, 16.0)],
        )

        # Short turn (< 1.5s) overlapping a longer turn (>= 1.5s) at handover must keep its full bounds
        handover_short = resolve_turn_overlaps([
            {"speaker": "A", "start": 61.66, "end": 64.91},
            {"speaker": "B", "start": 64.36, "end": 65.31},
        ])
        self.assertEqual(
            [(t["speaker"], t["start"], t["end"]) for t in handover_short],
            [("A", 61.66, 64.36), ("B", 64.36, 65.31)],
        )

    def test_plan_windows_keeps_carved_interjection(self):
        turns = [
            {"speaker": "SPEAKER_00", "start": 61.64, "end": 72.15},
            {"speaker": "SPEAKER_02", "start": 64.36, "end": 65.32},
        ]
        windows = BaseASREngine.plan_windows(turns)
        self.assertEqual([w["speaker"] for w in windows], ["SPEAKER_00", "SPEAKER_02", "SPEAKER_00"])

    def test_filter_phantom_speakers(self):
        from app.services.diarization import filter_phantom_speakers

        turns = [
            {"speaker": "S0", "start": 0.0, "end": 5.0},
            {"speaker": "S1", "start": 5.0, "end": 9.0},
            {"speaker": "S3", "start": 9.0, "end": 9.8},      # cumulative < 1.5s -> drop
            {"speaker": "S4", "start": 395.8, "end": 397.8},  # single turn (len==1) < 2.5s -> drop
        ]
        self.assertEqual({t["speaker"] for t in filter_phantom_speakers(turns)}, {"S0", "S1"})


if __name__ == "__main__":
    unittest.main()
