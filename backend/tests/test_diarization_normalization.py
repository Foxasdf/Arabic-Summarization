import unittest
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.diarization import _normalize_turns


class TestDiarizationNormalization(unittest.TestCase):
    def test_empty_turns(self):
        self.assertEqual(_normalize_turns([]), [])

    def test_micro_noise_filtered(self):
        # All turns shorter than 0.15s should be filtered
        turns = [
            {"speaker": "SPK1", "start": 0.0, "end": 0.10},
            {"speaker": "SPK2", "start": 0.5, "end": 0.64},
        ]
        self.assertEqual(_normalize_turns(turns), [])

    def test_single_turn_renamed(self):
        turns = [{"speaker": "SPK_ORIG", "start": 1.0, "end": 5.0}]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["start"], 1.0)
        self.assertEqual(res[0]["end"], 5.0)

    def test_per_speaker_merging(self):
        # Same speaker intervals with gap <= 0.5s should merge
        turns = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "A", "start": 2.3, "end": 4.0},  # gap 0.3s <= 0.5s
        ]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["start"], 0.0)
        self.assertEqual(res[0]["end"], 4.0)

    def test_interjection_kept_as_overlapping_turn(self):
        # A short interjection nested inside another speaker's long turn is preserved
        # as-is (overlapping) rather than punching a hole in the primary turn.
        turns = [
            {"speaker": "SPK_A", "start": 10.0, "end": 20.0},
            {"speaker": "SPK_B", "start": 14.0, "end": 15.0},
        ]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["start"], 10.0)
        self.assertEqual(res[0]["end"], 20.0)

        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[1]["start"], 14.0)
        self.assertEqual(res[1]["end"], 15.0)

    def test_overlapping_turns_preserved(self):
        # Overlapping speech is kept intact; word alignment resolves it later.
        turns = [
            {"speaker": "SPK_A", "start": 0.0, "end": 5.0},
            {"speaker": "SPK_B", "start": 4.0, "end": 9.0},
        ]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["start"], 0.0)
        self.assertEqual(res[0]["end"], 5.0)

        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[1]["start"], 4.0)
        self.assertEqual(res[1]["end"], 9.0)

    def test_appearance_order_speaker_naming(self):
        # Speakers must be named SPEAKER_00, SPEAKER_01, SPEAKER_02 in order of first appearance
        turns = [
            {"speaker": "CHARLIE", "start": 1.0, "end": 3.0},
            {"speaker": "ALICE", "start": 3.5, "end": 5.0},
            {"speaker": "BOB", "start": 5.5, "end": 7.0},
            {"speaker": "CHARLIE", "start": 7.5, "end": 9.0},
        ]
        res = _normalize_turns(turns)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")  # CHARLIE
        self.assertEqual(res[1]["speaker"], "SPEAKER_01")  # ALICE
        self.assertEqual(res[2]["speaker"], "SPEAKER_02")  # BOB
        self.assertEqual(res[3]["speaker"], "SPEAKER_00")  # CHARLIE again


    def test_negative_start_clamped(self):
        # Turns with negative start should be clamped to 0.0
        turns = [{"speaker": "A", "start": -0.5, "end": 2.0}]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["start"], 0.0)
        self.assertEqual(res[0]["end"], 2.0)

    def test_inverted_and_zero_duration_filtered(self):
        # Inverted timestamps (start > end) and zero duration should be filtered
        turns = [
            {"speaker": "A", "start": 5.0, "end": 2.0},
            {"speaker": "B", "start": 3.0, "end": 3.0},
            {"speaker": "C", "start": 1.0, "end": 3.0},
        ]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["start"], 1.0)
        self.assertEqual(res[0]["end"], 3.0)

    def test_short_interjection_preserved(self):
        # Rapid conversational interjections (e.g. 0.35s) survive normalization
        turns = [
            {"speaker": "HOST", "start": 10.0, "end": 20.0},
            {"speaker": "GUEST", "start": 14.0, "end": 14.35},  # 0.35s interjection
        ]
        res = _normalize_turns(turns)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[1]["start"], 14.0)
        self.assertEqual(res[1]["end"], 14.35)


if __name__ == "__main__":
    unittest.main()
