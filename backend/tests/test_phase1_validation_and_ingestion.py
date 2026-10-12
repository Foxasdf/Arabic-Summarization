"""
Unit and integration tests for Phase 1: Request Validation, Ingestion, and Observability.
Verifies:
- Strict typed contract validation (HTTP 422 on invalid combinations)
- Deprecation notices for legacy aliases
- Bounded full decode & corruption detection
- Clipping, silence, rumble, and anti-phase stereo detection
- Versioned execution manifest generation
- Transactional atomic artifact publishing
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
    RequestValidationError422,
    UnsupportedMediaError415,
    get_legacy_alias_usage_stats,
)
from app.services.preprocessing.analyzer import AudioAnalyzer
from app.services.preprocessing.engine import PreprocessingEngine
from app.services.preprocessing.manifest import ExecutionManifest
from app.services.preprocessing.transaction import TransactionalOutputManager


class TestPhase1ValidationAndIngestion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_phase1_"))
        self.engine = PreprocessingEngine()
        self.analyzer = AudioAnalyzer()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_request_contract_valid_profiles(self):
        """Valid profile requests initialize cleanly."""
        r_asr = AudioProcessingRequest(output_profile="asr")
        self.assertEqual(r_asr.output_profile, OutputProfile.ASR)

        r_spk = AudioProcessingRequest(output_profile="speaker")
        self.assertEqual(r_spk.output_profile, OutputProfile.SPEAKER)

        r_play = AudioProcessingRequest(output_profile="playback")
        self.assertEqual(r_play.output_profile, OutputProfile.PLAYBACK)

    def test_request_contract_rejects_invalid_values(self):
        """Unknown or invalid profile/denoise/normalization values raise ValueError."""
        with self.assertRaises(ValueError):
            AudioProcessingRequest(output_profile="invalid_profile")

        with self.assertRaises(ValueError):
            AudioProcessingRequest(denoise_mode="super_ultra_denoise")

        with self.assertRaises(ValueError):
            AudioProcessingRequest(normalization="compressor_v2")

    def test_request_contract_profile_invariants(self):
        """safe_gain is ASR-only; ebu_r128 is Playback-only."""
        # safe_gain on ASR is valid
        r = AudioProcessingRequest(
            output_profile=OutputProfile.ASR,
            normalization=NormalizationMode.SAFE_GAIN,
            safe_gain_db=4.5,
        )
        self.assertEqual(r.normalization, NormalizationMode.SAFE_GAIN)

        # safe_gain on Speaker is INVALID
        with self.assertRaises(ValueError):
            AudioProcessingRequest(
                output_profile=OutputProfile.SPEAKER,
                normalization=NormalizationMode.SAFE_GAIN,
            )

        # safe_gain on Playback is INVALID
        with self.assertRaises(ValueError):
            AudioProcessingRequest(
                output_profile=OutputProfile.PLAYBACK,
                normalization=NormalizationMode.SAFE_GAIN,
            )

        # ebu_r128 on Playback is valid
        r_pb = AudioProcessingRequest(
            output_profile=OutputProfile.PLAYBACK,
            normalization=NormalizationMode.EBU_R128,
        )
        self.assertEqual(r_pb.normalization, NormalizationMode.EBU_R128)

        # ebu_r128 on ASR is INVALID
        with self.assertRaises(ValueError):
            AudioProcessingRequest(
                output_profile=OutputProfile.ASR,
                normalization=NormalizationMode.EBU_R128,
            )

        # ebu_r128 on Speaker is INVALID
        with self.assertRaises(ValueError):
            AudioProcessingRequest(
                output_profile=OutputProfile.SPEAKER,
                normalization=NormalizationMode.EBU_R128,
            )

    def test_legacy_aliases_mapping_and_deprecation_warnings(self):
        """Legacy aliases map to canonical presets with deprecation warnings."""
        # none -> raw
        r_none = AudioProcessingRequest(denoise_mode="none")
        self.assertEqual(r_none.denoise_mode, DenoiseMode.RAW)

        # low -> light
        r_low = AudioProcessingRequest(denoise_mode="low")
        self.assertEqual(r_low.denoise_mode, DenoiseMode.LIGHT)

        # moderate -> balanced
        r_mod = AudioProcessingRequest(denoise_mode="moderate")
        self.assertEqual(r_mod.denoise_mode, DenoiseMode.BALANCED)

        # high -> strong
        r_high = AudioProcessingRequest(denoise_mode="high")
        self.assertEqual(r_high.denoise_mode, DenoiseMode.STRONG)

        # dynamic -> auto
        r_dyn = AudioProcessingRequest(denoise_mode="dynamic")
        self.assertEqual(r_dyn.denoise_mode, DenoiseMode.AUTO)

        # Arabic alias: "لا معالجة" -> raw
        r_ar = AudioProcessingRequest(denoise_mode="لا معالجة")
        self.assertEqual(r_ar.denoise_mode, DenoiseMode.RAW)

        stats = get_legacy_alias_usage_stats()
        self.assertGreater(stats.get("none", 0), 0)
        self.assertGreater(stats.get("low", 0), 0)

    def test_empty_and_corrupt_files_validation(self):
        """Empty and corrupt files fail validation."""
        # Empty file
        empty_file = self.temp_dir / "empty.wav"
        empty_file.write_bytes(b"")
        val, feat, _ = self.analyzer.decode_and_analyze(empty_file)
        self.assertFalse(val.is_valid)
        self.assertIn("0 bytes", val.error_message.lower())

        # Corrupt file
        corrupt_file = self.temp_dir / "corrupt.wav"
        corrupt_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEbadcontent")
        val_c, feat_c, _ = self.analyzer.decode_and_analyze(corrupt_file)
        self.assertFalse(val_c.is_valid)

    def test_silence_and_clipping_detection(self):
        """Analyzer flags clipping and silence accurately."""
        sr = 16000

        # Silence
        silence_file = self.temp_dir / "silence.wav"
        wavfile.write(str(silence_file), sr, np.zeros(sr * 2, dtype=np.int16))
        val_s, feat_s, _ = self.analyzer.decode_and_analyze(silence_file)
        self.assertTrue(val_s.is_valid)
        self.assertTrue(feat_s.is_silent)
        self.assertLess(feat_s.rms_dbfs, -65.0)

        # Clipped audio
        t = np.linspace(0, 1, sr, endpoint=False)
        clipped_sig = np.clip(np.sin(2 * np.pi * 440 * t) * 3.0, -1.0, 1.0)
        clip_file = self.temp_dir / "clipped.wav"
        wavfile.write(str(clip_file), sr, (clipped_sig * 32767).astype(np.int16))
        val_c, feat_c, _ = self.analyzer.decode_and_analyze(clip_file)
        self.assertTrue(val_c.is_valid)
        self.assertTrue(feat_c.is_clipped)
        self.assertGreater(feat_c.clipped_sample_count, 100)

    def test_phase_aware_channel_selection(self):
        """Anti-phase stereo is detected and downmixed without cancellation."""
        sr = 16000
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 300 * t)
        # Left and Right are opposite polarity (corr = -1.0)
        anti_stereo = np.stack([sig, -sig], axis=1)
        stereo_path = self.temp_dir / "anti_stereo.wav"
        wavfile.write(str(stereo_path), sr, (anti_stereo * 32767).astype(np.int16))

        val, feat, _ = self.analyzer.decode_and_analyze(stereo_path)
        self.assertTrue(feat.anti_phase_detected)
        self.assertLess(feat.stereo_correlation, -0.9)

        # Process through ASR profile: must NOT result in silence
        req = AudioProcessingRequest(output_profile=OutputProfile.ASR)
        res = self.engine.process(stereo_path, req, self.temp_dir / "asr_out")
        self.assertTrue(res.success)
        out_wav = Path(res.output_files["asr"])
        val_out, feat_out, _ = self.analyzer.decode_and_analyze(out_wav)
        # Signal energy must be preserved (not canceled out to -90 dB)
        self.assertGreater(feat_out.rms_dbfs, -20.0)

    def test_transactional_output_and_manifest(self):
        """Engine writes artifacts and manifest atomically, cleaning up staging."""
        sr = 16000
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        sig = 0.4 * np.sin(2 * np.pi * 440 * t)
        in_file = self.temp_dir / "test_input.wav"
        wavfile.write(str(in_file), sr, (sig * 32767).astype(np.int16))

        req = AudioProcessingRequest(output_profile=OutputProfile.ASR, denoise_mode=DenoiseMode.LIGHT)
        target_out = self.temp_dir / "job_published_output"

        res = self.engine.process(in_file, req, target_out)
        self.assertTrue(res.success)
        self.assertTrue(target_out.exists())

        # Check published files
        asr_artifact = target_out / "asr.wav"
        manifest_artifact = target_out / "manifest.json"
        self.assertTrue(asr_artifact.exists())
        self.assertTrue(manifest_artifact.exists())

        # Verify manifest structure
        manifest_data = ExecutionManifest.model_validate_json(manifest_artifact.read_text(encoding="utf-8"))
        self.assertEqual(manifest_data.schema_version, "3.1.0")
        self.assertEqual(manifest_data.pipeline_version, "3.1.0")
        self.assertIn("asr", manifest_data.outputs)
        self.assertTrue(manifest_data.outputs["asr"].sha256_hash)
        self.assertGreater(manifest_data.performance["total_wall_time_ms"], 0)

        # Check no staging directories remain in parent
        stray_staging = list(self.temp_dir.glob(".staging_*"))
        self.assertEqual(len(stray_staging), 0)


if __name__ == "__main__":
    unittest.main()
