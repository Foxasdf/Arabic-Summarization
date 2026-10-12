"""
End-to-end integration test for routes.py with dual profile preprocessing (Production v3.1).
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from scipy.io import wavfile

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.audio_processor import process_dual_profile_audio
from app.services.preprocessing.manifest import ExecutionManifest


class TestE2EPreprocessingRoutes(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_e2e_routes_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dual_profile_pipeline_execution(self):
        """Verifies process_dual_profile_audio produces ASR, Speaker, and Manifest."""
        sr = 16000
        t = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
        sig = 0.4 * np.sin(2 * np.pi * 350 * t)

        source_file = self.temp_dir / "meeting_uploaded.wav"
        wavfile.write(str(source_file), sr, (sig * 32767).astype(np.int16))

        asr_out = self.temp_dir / "meeting.16k.wav"
        spk_out = self.temp_dir / "meeting.speaker.16k.wav"

        # Execute dual profile processing
        import asyncio
        ok, err = asyncio.run(
            process_dual_profile_audio(source_file, asr_out, spk_out, denoise_mode="light")
        )
        self.assertTrue(ok)
        self.assertIsNone(err)

        # Verify ASR output exists and has valid audio
        self.assertTrue(asr_out.exists())
        self.assertGreater(asr_out.stat().st_size, 1000)

        # Verify Speaker output exists and has valid audio
        self.assertTrue(spk_out.exists())
        self.assertGreater(spk_out.stat().st_size, 1000)

        # Verify Manifest exists and has schema 3.1.0
        manifest_file = self.temp_dir / "meeting.manifest.json"
        self.assertTrue(manifest_file.exists())
        manifest = ExecutionManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(manifest.schema_version, "3.1.0")
        self.assertIn("asr", manifest.outputs)
        self.assertIn("speaker", manifest.outputs)
        self.assertEqual(manifest.outputs["speaker"].channels, 1)
        self.assertEqual(manifest.outputs["asr"].channels, 1)


if __name__ == "__main__":
    unittest.main()
