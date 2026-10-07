import unittest
import sys
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.alignment import align_timestamps
from app.services.summarization import build_transcript_lines
from app.models.registry import ModelRegistry, ModelInfo


class TestArabicSummarizerCore(unittest.TestCase):

    def test_build_transcript_lines(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "مرحباً بكم في الاجتماع.", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_00", "text": "سنناقش الخطة.", "start": 2.1, "end": 4.0},
            {"speaker": "SPEAKER_01", "text": "أهلاً وسهلاً.", "start": 4.5, "end": 6.0},
        ]
        speakers = {"SPEAKER_00": "أحمد", "SPEAKER_01": "خالد"}
        lines = build_transcript_lines(segments, speakers)
        self.assertEqual(len(lines), 2)
        self.assertIn("أحمد: مرحباً بكم في الاجتماع. سنناقش الخطة.", lines[0])
        self.assertIn("خالد: أهلاً وسهلاً.", lines[1])

    def test_alignment_empty(self):
        result = align_timestamps([], [])
        self.assertEqual(result, [])

    def test_alignment_no_diarization(self):
        words = [
            {"word": "مرحبا", "start": 0.0, "end": 0.5},
            {"word": "بكم", "start": 0.5, "end": 1.0},
        ]
        result = align_timestamps(words, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[0]["text"], "مرحبا بكم")

    def test_alignment_multi_speaker(self):
        words = [
            {"word": "السلام", "start": 0.0, "end": 0.4},
            {"word": "عليكم.", "start": 0.4, "end": 0.8},
            {"word": "وعليكم", "start": 2.0, "end": 2.4},
            {"word": "السلام.", "start": 2.4, "end": 2.8},
        ]
        diarization = [
            {"speaker": "متحدث 1", "start": 0.0, "end": 1.0},
            {"speaker": "متحدث 2", "start": 1.8, "end": 3.0},
        ]
        result = align_timestamps(words, diarization)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], "متحدث 1")
        self.assertEqual(result[0]["text"], "السلام عليكم.")
        self.assertEqual(result[1]["speaker"], "متحدث 2")
        self.assertEqual(result[1]["text"], "وعليكم السلام.")

    def test_model_registry(self):
        reg = ModelRegistry()
        llms = reg.list_llm_models()
        self.assertGreaterEqual(len(llms), 1)
        self.assertEqual(llms[0].model_type, "llm")

        # Test adding a custom LLM
        custom_model = ModelInfo(
            key="custom-arabic-llm",
            name="Custom Arabic LLM 14B",
            model_type="llm",
            gguf_filename="custom-model.gguf",
            description="موديل إضافي مخصص للاختبار",
            context_size=16384
        )
        reg.register(custom_model)
        fetched = reg.get_model("custom-arabic-llm")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "Custom Arabic LLM 14B")
        self.assertEqual(fetched.context_size, 16384)


    def test_asr_progress_calculation(self):
        lo, hi = 25, 92
        chunks_count = 10
        reported = []
        for i in range(chunks_count):
            start_pct = int(lo + (i / max(1, chunks_count)) * (hi - lo))
            end_pct = int(lo + ((i + 1) / max(1, chunks_count)) * (hi - lo))
            reported.append(start_pct)
            reported.append(end_pct)

        # Check bounds
        self.assertEqual(reported[0], 25)
        self.assertEqual(reported[-1], 92)
        # Check monotonicity
        for a, b in zip(reported[:-1], reported[1:]):
            self.assertLessEqual(a, b)


if __name__ == "__main__":
    unittest.main()
