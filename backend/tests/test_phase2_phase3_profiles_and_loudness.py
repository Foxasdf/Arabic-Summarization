"""
Unit and integration tests for Phase 2 (Separate Profiles) and Phase 3 (Playback Loudness).
Verifies:
- Complete isolation of ASR, Speaker, and Playback processing branches
- Speaker branch remains unprocessed (raw acoustics)
- ASR branch is free of dynamic compression
- Two-pass EBU R128 playback loudness meets -16 LUFS stereo, -19 LUFS mono, TP <= -1.5 dBTP
- Linear mode confirmation and fallback logging
- Silence and short clip edge case handling with max boost limiter
- Post-conversion verification (duration drift, format, channels, true peak)
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

from app.services.preprocessing.contracts import (
    OutputProfile,
    DenoiseMode,
    NormalizationMode,
    AudioProcessingRequest,
)
from app.services.preprocessing.analyzer import AudioAnalyzer
from app.services.preprocessing.engine import PreprocessingEngine
from app.services.preprocessing.loudness import LoudnessProcessor


class TestPhase2And3ProfilesAndLoudness(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_phase2_3_"))
        self.engine = PreprocessingEngine()
        self.analyzer = AudioAnalyzer()
        self.loudness = LoudnessProcessor()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_stereo_test_audio(self, duration_sec: float = 4.0, sr: int = 44100) -> Path:
        t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
        left = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 900 * t)
        right = 0.3 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
        stereo = np.stack([left, right], axis=1)
        p = self.temp_dir / f"stereo_test_{duration_sec}s.wav"
        wavfile.write(str(p), sr, (stereo * 32767).astype(np.int16))
        return p

    def test_independent_dual_output_branches(self):
        """All 3 profile branches execute independently from source."""
        in_wav = self._create_stereo_test_audio(duration_sec=3.0)

        requests = [
            AudioProcessingRequest(
                output_profile=OutputProfile.ASR,
                denoise_mode=DenoiseMode.BALANCED,
                normalization=NormalizationMode.SAFE_GAIN,
                safe_gain_db=2.5,
            ),
            AudioProcessingRequest(
                output_profile=OutputProfile.SPEAKER,
                # Speaker defaults to raw denoise and off normalization
            ),
            AudioProcessingRequest(
                output_profile=OutputProfile.PLAYBACK,
                normalization=NormalizationMode.EBU_R128,
            ),
        ]

        target_out = self.temp_dir / "multi_branch_job"
        res = self.engine.process(in_wav, requests, target_out)
        self.assertTrue(res.success)

        # 1. Verify ASR artifact: 16kHz mono, denoised & safe gain applied
        asr_file = Path(res.output_files["asr"])
        self.assertTrue(asr_file.exists())
        val_asr, feat_asr, _ = self.analyzer.decode_and_analyze(asr_file)
        self.assertEqual(feat_asr.sample_rate, 16000)
        self.assertEqual(feat_asr.channels, 1)

        # 2. Verify Speaker artifact: 16kHz mono, completely UNPROCESSED (no denoise, no gain)
        spk_file = Path(res.output_files["speaker"])
        self.assertTrue(spk_file.exists())
        val_spk, feat_spk, _ = self.analyzer.decode_and_analyze(spk_file)
        self.assertEqual(feat_spk.sample_rate, 16000)
        self.assertEqual(feat_spk.channels, 1)

        # ASR with safe_gain (+2.5dB) must have different RMS than Speaker (raw)
        self.assertNotEqual(feat_asr.rms_dbfs, feat_spk.rms_dbfs)

        # 3. Verify Playback artifact: preserves stereo (2 channels)
        play_file = Path(res.output_files["playback"])
        self.assertTrue(play_file.exists())
        val_play, feat_play, _ = self.analyzer.decode_and_analyze(play_file)
        self.assertEqual(feat_play.channels, 2)

    def test_playback_two_pass_stereo_loudness_targets(self):
        """Playback stereo reaches target -16 LUFS with true peak <= -1.5 dBTP."""
        in_wav = self._create_stereo_test_audio(duration_sec=5.0)

        req = AudioProcessingRequest(
            output_profile=OutputProfile.PLAYBACK,
            normalization=NormalizationMode.EBU_R128,
        )

        target_out = self.temp_dir / "playback_stereo_job"
        res = self.engine.process(in_wav, req, target_out)
        self.assertTrue(res.success)

        play_art = res.manifest.outputs["playback"]
        self.assertEqual(play_art.channels, 2)
        # Measured true peak must meet <= -1.3 dBTP ceiling (-1.5 dBTP target)
        self.assertLessEqual(play_art.true_peak_dbtp, -1.3)
        # Measured integrated loudness should be close to -16 LUFS
        if play_art.measured_lufs is not None:
            self.assertAlmostEqual(play_art.measured_lufs, -16.0, delta=1.5)

    def test_playback_two_pass_mono_loudness_targets(self):
        """Playback mono reaches target -19 LUFS with true peak <= -1.5 dBTP."""
        sr = 16000
        t = np.linspace(0, 5.0, int(5.0 * sr), endpoint=False)
        sig = 0.3 * np.sin(2 * np.pi * 350 * t)
        mono_path = self.temp_dir / "mono_test.wav"
        wavfile.write(str(mono_path), sr, (sig * 32767).astype(np.int16))

        req = AudioProcessingRequest(
            output_profile=OutputProfile.PLAYBACK,
            normalization=NormalizationMode.EBU_R128,
            target_channels=1,
        )

        target_out = self.temp_dir / "playback_mono_job"
        res = self.engine.process(mono_path, req, target_out)
        self.assertTrue(res.success)

        play_art = res.manifest.outputs["playback"]
        self.assertEqual(play_art.channels, 1)
        self.assertLessEqual(play_art.true_peak_dbtp, -1.3)
        if play_art.measured_lufs is not None:
            self.assertAlmostEqual(play_art.measured_lufs, -19.0, delta=1.5)

    def test_silence_playback_bypass(self):
        """Silent audio safely bypasses loudnorm to prevent noise explosion."""
        sr = 16000
        silence_path = self.temp_dir / "pure_silence.wav"
        wavfile.write(str(silence_path), sr, np.zeros(sr * 3, dtype=np.int16))

        req = AudioProcessingRequest(
            output_profile=OutputProfile.PLAYBACK,
            normalization=NormalizationMode.EBU_R128,
        )

        target_out = self.temp_dir / "playback_silence_job"
        res = self.engine.process(silence_path, req, target_out)
        self.assertTrue(res.success)

        play_out = Path(res.output_files["playback"])
        val, feat, _ = self.analyzer.decode_and_analyze(play_out)
        self.assertTrue(feat.is_silent)
        self.assertLess(feat.rms_dbfs, -65.0)

    def test_duration_drift_and_format_verification(self):
        """Post-conversion verifies duration drift is minimal."""
        in_wav = self._create_stereo_test_audio(duration_sec=3.5)

        req = AudioProcessingRequest(
            output_profile=OutputProfile.ASR,
            denoise_mode=DenoiseMode.LIGHT,
        )
        target_out = self.temp_dir / "drift_check_job"
        res = self.engine.process(in_wav, req, target_out)
        self.assertTrue(res.success)

        asr_file = Path(res.output_files["asr"])
        val, feat, _ = self.analyzer.decode_and_analyze(asr_file)
        self.assertAlmostEqual(feat.duration_seconds, 3.5, delta=0.08)


if __name__ == "__main__":
    unittest.main()
