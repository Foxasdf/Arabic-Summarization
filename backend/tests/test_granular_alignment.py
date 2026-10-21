import unittest
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.alignment import align_timestamps


class TestGranularAlignment(unittest.TestCase):
    def test_interjection_separation(self):
        # Simulated sequence where Student (SPEAKER_02) speaks, Doctor (SPEAKER_00) interjects
        words = [
            {"word": "يعني", "start": 80.0, "end": 80.5},
            {"word": "نحكي", "start": 80.5, "end": 81.0},
            {"word": "ولا", "start": 81.0, "end": 81.5},
            {"word": "لا؟", "start": 81.5, "end": 82.0},
            {"word": "اشاره", "start": 93.8, "end": 94.2},
            {"word": "الصوت", "start": 94.2, "end": 94.7},
            {"word": "يا", "start": 94.7, "end": 95.1},
            {"word": "سلام", "start": 95.1, "end": 95.6},
            {"word": "مشان", "start": 95.6, "end": 96.0},
            {"word": "شو", "start": 96.0, "end": 96.5},
            {"word": "بدك", "start": 96.5, "end": 97.0},
            {"word": "تستخدمه", "start": 97.0, "end": 97.4},
            {"word": "انت", "start": 97.4, "end": 97.9},
            {"word": "هذا", "start": 97.9, "end": 98.3},
            {"word": "التكست", "start": 98.3, "end": 98.8},
            {"word": "فورم؟", "start": 98.8, "end": 99.2},
            {"word": "نحن", "start": 99.3, "end": 99.7},
            {"word": "بدنا", "start": 99.7, "end": 100.1},
            {"word": "نعمل", "start": 100.1, "end": 100.6},
            {"word": "شيء", "start": 100.6, "end": 101.0},
        ]
        turns = [
            {"speaker": "SPEAKER_02", "start": 80.0, "end": 82.5},
            {"speaker": "SPEAKER_00", "start": 93.5, "end": 97.5},
            {"speaker": "SPEAKER_02", "start": 97.5, "end": 102.0},
        ]

        result = align_timestamps(words, turns)
        
        # Verify 100% word retention
        out_words = " ".join(s["text"] for s in result).split()
        self.assertEqual(len(out_words), len(words))

        # Verify doctor's question is in SPEAKER_00
        doc_segs = [s for s in result if s["speaker"] == "SPEAKER_00"]
        self.assertTrue(len(doc_segs) > 0)
        self.assertIn("سلام", doc_segs[0]["text"])
        self.assertIn("مشان شو", doc_segs[0]["text"])

        # Verify student reply is in SPEAKER_02
        student_segs = [s for s in result if s["speaker"] == "SPEAKER_02"]
        self.assertTrue(len(student_segs) >= 2)
        self.assertIn("نحن بدنا", student_segs[-1]["text"])

    def test_standalone_interjection_alternation(self):
        # Student explaining (SPEAKER_01), Doctor interjecting (SPEAKER_03) with "يا سلام" and "تماما"
        words = [
            {"word": "المشكلة", "start": 192.0, "end": 193.0},
            {"word": "بالتشكيل", "start": 193.0, "end": 194.0},
            {"word": "يا", "start": 198.0, "end": 198.4},
            {"word": "سلام", "start": 198.4, "end": 198.8},
            {"word": "بيقول", "start": 199.0, "end": 200.0},
            {"word": "المحافظ", "start": 200.0, "end": 201.0},
            {"word": "تماما", "start": 202.4, "end": 202.8},
            {"word": "فنحن", "start": 203.0, "end": 204.0},
            {"word": "لقينا", "start": 204.0, "end": 205.0},
            {"word": "الحل", "start": 205.0, "end": 206.0},
        ]
        # Raw tracks: long student track with Doctor interjections overlapping/nested
        tracks = [
            {"speaker": "SPEAKER_01", "start": 191.0, "end": 207.0},
            {"speaker": "SPEAKER_03", "start": 198.0, "end": 198.9},
            {"speaker": "SPEAKER_03", "start": 202.3, "end": 202.9},
        ]
        result = align_timestamps(words, tracks)

        # Verify 100% word retention
        out_words = " ".join(s["text"] for s in result).split()
        self.assertEqual(len(out_words), len(words))

        # Must alternate: Student -> Doctor ("يا سلام") -> Student -> Doctor ("تماما") -> Student
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["speaker"], "SPEAKER_01")
        self.assertIn("المشكلة بالتشكيل", result[0]["text"])

        self.assertEqual(result[1]["speaker"], "SPEAKER_03")
        self.assertEqual(result[1]["text"], "يا سلام")

        self.assertEqual(result[2]["speaker"], "SPEAKER_01")
        self.assertIn("بيقول المحافظ", result[2]["text"])

        self.assertEqual(result[3]["speaker"], "SPEAKER_03")
        self.assertEqual(result[3]["text"], "تماما")

        self.assertEqual(result[4]["speaker"], "SPEAKER_01")
        self.assertIn("فنحن لقينا الحل", result[4]["text"])

    def test_empty_inputs(self):
        self.assertEqual(align_timestamps([], []), [])
        self.assertEqual(align_timestamps([], [{"speaker": "SPEAKER_00", "start": 0, "end": 1}]), [])

    def test_single_speaker_fallback(self):
        words = [{"word": "مرحبا", "start": 0.0, "end": 0.5}]
        res = align_timestamps(words, [])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[0]["text"], "مرحبا")

    def test_question_preface_unity(self):
        # Image 1 & 2 case: "تماما طيب شو يعني..." and "يا سلام مشان شو بدك..."
        # When an interjection is followed by a question, it belongs to the questioner and should NOT split
        words = [
            {"word": "جزء", "start": 74.0, "end": 74.5},
            {"word": "من", "start": 74.5, "end": 75.0},
            {"word": "مشروعنا", "start": 75.0, "end": 76.0},
            {"word": "تماما", "start": 77.0, "end": 77.5},
            {"word": "طيب", "start": 77.5, "end": 78.0},
            {"word": "شو", "start": 78.0, "end": 78.5},
            {"word": "يعني", "start": 78.5, "end": 79.0},
            {"word": "تكست", "start": 79.0, "end": 79.5},
            {"word": "تفونيم؟", "start": 79.5, "end": 80.0},
            {"word": "يعني", "start": 80.0, "end": 80.5},
            {"word": "نحكي", "start": 80.5, "end": 81.0},
            {"word": "يا", "start": 94.7, "end": 95.1},
            {"word": "سلام", "start": 95.1, "end": 95.6},
            {"word": "مشان", "start": 95.6, "end": 96.0},
            {"word": "شو", "start": 96.0, "end": 96.5},
            {"word": "بدك", "start": 96.5, "end": 97.0},
            {"word": "تستخدمه؟", "start": 97.0, "end": 97.5},
        ]
        turns = [
            {"speaker": "SPEAKER_01", "start": 74.0, "end": 76.5},
            {"speaker": "SPEAKER_00", "start": 76.8, "end": 80.2},
            {"speaker": "SPEAKER_02", "start": 80.2, "end": 82.0},
            {"speaker": "SPEAKER_00", "start": 94.5, "end": 98.0},
        ]
        res = align_timestamps(words, turns)
        
        # Verify 100% word retention
        out_words = " ".join(s["text"] for s in res).split()
        self.assertEqual(len(out_words), len(words))

        # "تماما طيب شو يعني تكست تفونيم؟" must be a single segment of SPEAKER_00
        seg_q1 = [s for s in res if "تماما" in s["text"]]
        self.assertEqual(len(seg_q1), 1)
        self.assertEqual(seg_q1[0]["speaker"], "SPEAKER_00")
        self.assertIn("طيب شو يعني", seg_q1[0]["text"])

        # "يا سلام مشان شو بدك تستخدمه؟" must be a single segment of SPEAKER_00
        seg_q2 = [s for s in res if "سلام" in s["text"]]
        self.assertEqual(len(seg_q2), 1)
        self.assertEqual(seg_q2[0]["speaker"], "SPEAKER_00")
        self.assertIn("مشان شو", seg_q2[0]["text"])

    def test_consecutive_same_speaker_consolidation(self):
        from app.services.alignment import merge_consecutive_speaker_segments
        raw_segments = [
            {"speaker": "SPEAKER_02", "start": 13.0, "end": 14.5, "text": "سؤال"},
            {"speaker": "SPEAKER_02", "start": 15.0, "end": 18.2, "text": "بس انه انت مبين عليك مراحل الاكل"},
            {"speaker": "SPEAKER_01", "start": 19.0, "end": 20.0, "text": "صعبه صح"},
            {"speaker": "SPEAKER_02", "start": 20.5, "end": 22.0, "text": "ومطبات صعبه يعني"},
            {"speaker": "SPEAKER_01", "start": 36.0, "end": 40.0, "text": "بلشت هيك احس حالي بلشت"},
            {"speaker": "SPEAKER_01", "start": 42.5, "end": 46.0, "text": "مثلا كول اللقمه هي"},
        ]
        merged = merge_consecutive_speaker_segments(raw_segments)
        
        # Verify count: 6 raw segments -> 4 consolidated segments (no consecutive same speaker)
        self.assertEqual(len(merged), 4)

        # Segment 0: SPEAKER_02 (13.0 - 18.2)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_02")
        self.assertEqual(merged[0]["start"], 13.0)
        self.assertEqual(merged[0]["end"], 18.2)
        self.assertIn("سؤال", merged[0]["text"])
        self.assertIn("مراحل الاكل", merged[0]["text"])

        # Segment 1: SPEAKER_01
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[1]["text"], "صعبه صح")

        # Segment 2: SPEAKER_02
        self.assertEqual(merged[2]["speaker"], "SPEAKER_02")
        self.assertEqual(merged[2]["text"], "ومطبات صعبه يعني")

        # Segment 3: SPEAKER_01 (36.0 - 46.0) with gap > 2.0s (42.5 - 40.0 = 2.5s)
        self.assertEqual(merged[3]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[3]["start"], 36.0)
        self.assertEqual(merged[3]["end"], 46.0)
        self.assertEqual(merged[3]["text"], "بلشت هيك احس حالي بلشت، مثلا كول اللقمه هي")

    def test_gap_punctuation_formatting(self):
        from app.services.alignment import merge_consecutive_speaker_segments
        
        # Case A: Gap > 2.0s without punctuation -> inserts Arabic comma '، '
        segs_a = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.0, "text": "الجملة الأولى"},
            {"speaker": "SPEAKER_01", "start": 4.5, "end": 6.0, "text": "الجملة الثانية"},
        ]
        res_a = merge_consecutive_speaker_segments(segs_a)
        self.assertEqual(len(res_a), 1)
        self.assertEqual(res_a[0]["text"], "الجملة الأولى، الجملة الثانية")

        # Case B: Gap > 2.0s with punctuation already present -> does not duplicate comma
        segs_b = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.0, "text": "هل تفهم؟"},
            {"speaker": "SPEAKER_01", "start": 4.5, "end": 6.0, "text": "نعم بالتأكيد."},
        ]
        res_b = merge_consecutive_speaker_segments(segs_b)
        self.assertEqual(len(res_b), 1)
        self.assertEqual(res_b[0]["text"], "هل تفهم؟ نعم بالتأكيد.")

        # Case C: Gap <= 2.0s -> separates with standard space
        segs_c = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.0, "text": "مرحباً"},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 4.0, "text": "بكم"},
        ]
        res_c = merge_consecutive_speaker_segments(segs_c)
        self.assertEqual(len(res_c), 1)
        self.assertEqual(res_c[0]["text"], "مرحباً بكم")

    def test_short_confirmed_interjection_not_smoothed(self):
        # A 0.35s turn for SPEAKER_01 nested between SPEAKER_00 speech must be preserved
        words = [
            {"word": "المشروع", "start": 20.0, "end": 20.5},
            {"word": "صح", "start": 20.65, "end": 20.95},
            {"word": "تماماً", "start": 21.4, "end": 22.0},
        ]
        turns = [
            {"speaker": "SPEAKER_00", "start": 19.0, "end": 20.62},
            {"speaker": "SPEAKER_01", "start": 20.62, "end": 20.97},  # dur 0.35s
            {"speaker": "SPEAKER_00", "start": 21.0, "end": 23.0},
        ]
        res = align_timestamps(words, turns)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["speaker"], "SPEAKER_00")
        self.assertEqual(res[1]["speaker"], "SPEAKER_01")
        self.assertEqual(res[1]["text"], "صح")
        self.assertEqual(res[2]["speaker"], "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
