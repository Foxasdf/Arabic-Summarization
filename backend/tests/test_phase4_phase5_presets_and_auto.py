"""
Unit tests for Phase 4 (Denoiser Presets & Capabilities) and Phase 5 (Auto Mode).
Verifies:
- Frozen, immutable preset parameters (PRESET_VERSION = 3.1.0)
- Missing backend raises ServiceCapabilityError503 or engages declared fallback
- Calibrated auto mode decisions (short clips, clean speech, noisy environments)
- Static preset selection without dynamic parameter mutation
"""

import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.preprocessing.contracts import (
    DenoiseMode,
    ServiceCapabilityError503,
)
from app.services.preprocessing.capability import FFmpegCapabilities, FilterTestResult
from app.services.preprocessing.presets import (
    PRESETS,
    PRESET_VERSION,
    resolve_preset,
    resolve_auto_preset,
)
from app.services.preprocessing.analyzer import AudioFeatures


class TestPhase4And5PresetsAndAuto(unittest.TestCase):

    def test_frozen_presets_integrity(self):
        """All manual presets must have frozen version, parameters, and backends."""
        self.assertEqual(PRESET_VERSION, "3.1.0")

        # Raw preset
        self.assertIn("raw", PRESETS)
        self.assertEqual(PRESETS["raw"].backend, "bypass")
        self.assertEqual(PRESETS["raw"].filter_string, "")

        # Light preset
        self.assertIn("light", PRESETS)
        self.assertEqual(PRESETS["light"].backend, "afftdn")
        self.assertEqual(PRESETS["light"].parameters, {"nf": -30, "nr": 6, "tn": 0})
        self.assertEqual(PRESETS["light"].declared_fallback, "raw")

        # Balanced preset
        self.assertIn("balanced", PRESETS)
        self.assertEqual(PRESETS["balanced"].backend, "afftdn")
        self.assertEqual(PRESETS["balanced"].parameters, {"nf": -25, "nr": 10, "tn": 0})
        self.assertEqual(PRESETS["balanced"].declared_fallback, "light")

        # Strong preset
        self.assertIn("strong", PRESETS)
        self.assertEqual(PRESETS["strong"].backend, "afftdn")
        self.assertEqual(PRESETS["strong"].parameters, {"nf": -20, "nr": 15, "tn": 0})
        self.assertEqual(PRESETS["strong"].declared_fallback, "balanced")

    def test_missing_backend_raises_service_capability_error(self):
        """If required backend is unavailable and no fallback allowed, raises 503."""
        # Simulated capabilities with afftdn failing
        failing_caps = FFmpegCapabilities(
            binary_path="ffmpeg",
            available_filters={"afftdn"},
            filter_smoke_tests={
                "afftdn": FilterTestResult(
                    filter_name="afftdn",
                    is_supported=True,
                    live_smoke_passed=False,
                    error_message="Simulated filter init failure",
                )
            },
        )

        with self.assertRaises(ServiceCapabilityError503):
            resolve_preset(
                DenoiseMode.STRONG,
                capabilities=failing_caps,
                allow_fallback=False,
            )

    def test_declared_fallback_engaged_when_allowed(self):
        """When backend fails and fallback allowed, engages declared fallback."""
        failing_caps = FFmpegCapabilities(
            binary_path="ffmpeg",
            available_filters={"afftdn"},
            filter_smoke_tests={
                "afftdn": FilterTestResult(
                    filter_name="afftdn",
                    is_supported=True,
                    live_smoke_passed=False,
                    error_message="Simulated filter init failure",
                )
            },
        )

        # Light declares fallback to raw
        fallback_preset, reason = resolve_preset(
            DenoiseMode.LIGHT,
            capabilities=failing_caps,
            allow_fallback=True,
        )
        self.assertEqual(fallback_preset.preset_id, "raw")
        self.assertIn("failed live verification", reason)

    def test_auto_short_clip_conservative_selection(self):
        """Clips under 5.0 seconds select conservative light preset with low confidence."""
        feat_short = AudioFeatures(
            duration_seconds=3.2,
            estimated_snr_proxy_db=8.0,  # Even with noisy SNR!
            snr_confidence=0.35,
        )
        preset, meta, _ = resolve_auto_preset(features=feat_short)
        self.assertEqual(preset.preset_id, "light")
        self.assertIn("Short clip", meta["rationale"])

    def test_auto_clean_speech_selects_raw(self):
        """High SNR (>= 22 dB) audio selects raw bypass."""
        feat_clean = AudioFeatures(
            duration_seconds=12.0,
            estimated_snr_proxy_db=28.5,
            snr_confidence=0.85,
        )
        preset, meta, _ = resolve_auto_preset(features=feat_clean)
        self.assertEqual(preset.preset_id, "raw")
        self.assertIn("Clean speech", meta["rationale"])

    def test_auto_moderate_noise_selects_balanced(self):
        """Moderate noise (12 <= SNR < 22 dB) selects balanced preset."""
        feat_mod = AudioFeatures(
            duration_seconds=15.0,
            estimated_snr_proxy_db=16.5,
            snr_confidence=0.80,
        )
        preset, meta, _ = resolve_auto_preset(features=feat_mod)
        self.assertEqual(preset.preset_id, "balanced")
        self.assertIn("Moderate background noise", meta["rationale"])

    def test_auto_heavy_noise_selects_strong(self):
        """Heavy noise (< 12 dB) selects strong preset."""
        feat_noisy = AudioFeatures(
            duration_seconds=20.0,
            estimated_snr_proxy_db=7.2,
            snr_confidence=0.88,
        )
        preset, meta, _ = resolve_auto_preset(features=feat_noisy)
        self.assertEqual(preset.preset_id, "strong")
        self.assertIn("High noise environment", meta["rationale"])

    def test_auto_silence_selects_raw(self):
        """Silent audio selects raw bypass."""
        feat_silent = AudioFeatures(
            duration_seconds=10.0,
            is_silent=True,
            rms_dbfs=-75.0,
        )
        preset, meta, _ = resolve_auto_preset(features=feat_silent)
        self.assertEqual(preset.preset_id, "raw")


if __name__ == "__main__":
    unittest.main()
