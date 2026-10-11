"""
Unit and integration tests for audio enhancement, VAD, ASR guardrails, and modular architecture.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.services.audio_processor import build_audio_filter_chain
from app.services.vad import is_speech_present, compute_audio_rms_db
from app.models.asr.guardrails import clean_and_guard_asr_text
from app.models.asr import get_asr_engine, BaseASREngine, AudarASREngine, WhisperASREngine


class TestAudioEnhancementAndVAD(unittest.TestCase):

    def test_audio_filter_chain_modes(self):
        """Verify filter chain strings for various denoise modes."""
        off_chain = build_audio_filter_chain("off")
        self.assertEqual(off_chain, "")

        none_chain = build_audio_filter_chain("none")
        self.assertEqual(none_chain, "")

        no_proc_chain = build_audio_filter_chain("لا معالجة")
        self.assertEqual(no_proc_chain, "")

        auto_chain = build_audio_filter_chain("auto")
        self.assertIn("afftdn", auto_chain)
        # Verify agate and speechnorm are NEVER used (they cause speech cutoff)
        self.assertNotIn("agate", auto_chain)
        self.assertNotIn("speechnorm", auto_chain)

        dyn_chain = build_audio_filter_chain("dynamic")
        self.assertEqual(dyn_chain, auto_chain)

        low_chain = build_audio_filter_chain("low")
        self.assertIn("highpass=f=50", low_chain)

        mod_chain = build_audio_filter_chain("moderate")
        self.assertIn("nr=6", mod_chain)

        high_chain = build_audio_filter_chain("high")
        self.assertIn("nr=10", high_chain)

        # Arabic aliases
        self.assertEqual(build_audio_filter_chain("ديناميكي"), dyn_chain)
        self.assertEqual(build_audio_filter_chain("منخفضة"), low_chain)
        self.assertEqual(build_audio_filter_chain("متوسطة"), mod_chain)
        self.assertEqual(build_audio_filter_chain("عالية"), high_chain)

    def test_vad_silence_rejection(self):
        """VAD should reject digital silence."""
        silence = np.zeros(16000, dtype=np.float32)
        has_speech, rms = is_speech_present(silence)
        self.assertFalse(has_speech)
        self.assertLess(rms, -100.0)

    def test_vad_noise_rejection(self):
        """VAD should reject low-level background noise (typical ambient hiss)."""
        np.random.seed(42)
        noise = np.random.normal(0, 0.005, 16000).astype(np.float32)
        has_speech, rms = is_speech_present(noise)
        self.assertFalse(has_speech)
        self.assertLess(rms, -42.0)

    def test_vad_speech_acceptance(self):
        """VAD should accept genuine speech energy."""
        t = np.linspace(0, 1, 16000)
        # Synthesize audio with strong vocal formants (200Hz, 800Hz, 2500Hz)
        vocal = (0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
        has_speech, rms = is_speech_present(vocal)
        self.assertTrue(has_speech)
        self.assertGreater(rms, -30.0)

    def test_guardrails_official_text_preservation(self):
        """Guardrails should keep clean Arabic speech intact."""
        text = "مرحبا بكم في هذا الاجتماع لمناقشة خطة العمل لهذا العام"
        cleaned = clean_and_guard_asr_text(text, dur_sec=5.0)
        self.assertEqual(cleaned, text)

    def test_guardrails_hallucination_denylist(self):
        """Guardrails should completely reject known phantom hallucination phrases."""
        phantom = "شكراً لكم على حسن المتابعة"
        cleaned = clean_and_guard_asr_text(phantom, dur_sec=2.0)
        self.assertEqual(cleaned, "")

        phantom2 = "اشترك في القناة وتفعيل زر الجرس"
        cleaned2 = clean_and_guard_asr_text(phantom2, dur_sec=2.0)
        self.assertEqual(cleaned2, "")

    def test_guardrails_repetition_loop_breaker(self):
        """Guardrails should break repeating n-gram loops while preserving natural emphasis."""
        loop = "في هذا السياق في هذا السياق في هذا السياق في هذا السياق"
        cleaned = clean_and_guard_asr_text(loop, dur_sec=3.0)
        self.assertEqual(cleaned, "في هذا السياق")

        # Natural emphasis like "لا لا" or "نعم نعم" must be preserved
        emphasis = "لا لا هذا غير صحيح"
        cleaned_emp = clean_and_guard_asr_text(emphasis, dur_sec=2.0)
        self.assertEqual(cleaned_emp, "لا لا هذا غير صحيح")

    def test_guardrails_word_rate_burst_capping(self):
        """Guardrails should cap abnormal word-rate bursts on very short audio segments."""
        # 12 words in 1.0 second is physically impossible (~12 w/s vs normal max ~4.5 w/s)
        burst = "واحد اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة أحد عشر اثنا عشر"
        cleaned = clean_and_guard_asr_text(burst, dur_sec=1.0)
        words = cleaned.split()
        self.assertLessEqual(len(words), 5)

    def test_modular_asr_architecture(self):
        """Verify modular ASR engine hierarchy and factory instantiation."""
        audar = get_asr_engine("audar-asr-v1-turbo")
        self.assertIsInstance(audar, BaseASREngine)
        self.assertIsInstance(audar, AudarASREngine)

        whisper = get_asr_engine("whisper-large-v3-turbo")
        self.assertIsInstance(whisper, BaseASREngine)
        self.assertIsInstance(whisper, WhisperASREngine)

    def test_audar_prompt_integrity(self):
        """Verify that Audar engine strictly uses the official prompt without additions."""
        import inspect
        from app.models.asr.audar_engine import AudarASREngine
        source = inspect.getsource(AudarASREngine._transcribe_slice_internal)
        # Must contain exact official prompt
        self.assertIn("فرّغ الكلام العربي التالي.", source)
        # Must not contain negative prompt instructions
        self.assertNotIn("إذا كان المقطع", source)


if __name__ == "__main__":
    unittest.main()
