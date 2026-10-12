# Progress Log

## Session Complete: 2026-10-11
- Audited repository: probed FFmpeg 7.1, filters (`afftdn`, `anlmdn`, `arnndn`, `loudnorm`), and models.
- Built modular preprocessing package `backend/app/services/preprocessing/`:
  - `contracts.py`: Strict typed request models, legacy alias mapping, invariant validators.
  - `capability.py`: FFmpeg capability prober and live filter execution smoke tests.
  - `presets.py`: Immutable presets (`raw`, `light`, `balanced`, `strong`, `PRESET_VERSION=3.1.0`), declared fallbacks, calibrated `resolve_auto_preset`.
  - `analyzer.py`: Bounded full decode, clipping and silence flags, stereo phase correlation check, rumble energy ratio, SNR proxy estimation.
  - `loudness.py`: Two-pass EBU R128 loudness engine (-16 LUFS stereo, -19 LUFS mono, TP <= -1.5 dBTP, post-resample, linear mode validation, silence bypass, max boost limiter).
  - `engine.py`: Unified pipeline adhering strictly to the pinned stage order, independent profile rendering, output verification.
  - `manifest.py`: Schema-versioned execution manifest (`manifest.json` v3.1.0).
  - `transaction.py`: Transactional staging directory with atomic publishing.
  - `normalizer.py`: Arabic text normalizer v1.0.0 with both raw and normalized WER/CER scoring.
- Built `backend/tools/eval_pipeline.py`:
  - 8-clip smoke set (clean, silence, clipped, anti-phase, stereo normal, rumble, corrupt, short).
  - Quality dataset from Arabic audio corpus across clean, fan noise, crowd noise, quiet speech.
  - Baseline matrix recorded in `eval_results/eval_baseline_matrix.json` and `eval_results/eval_baseline_report.md`.
- Integrated into backend services:
  - `backend/app/services/audio_processor.py` provides `process_dual_profile_audio` and backward-compatible `convert_to_wav`.
  - `backend/app/api/routes.py` connects `spk_wav_path` strictly to diarization (Pyannote) and `wav_path` to ASR (transcription).
- Test suites:
  - `test_phase1_validation_and_ingestion.py` (8 tests passing)
  - `test_phase2_phase3_profiles_and_loudness.py` (5 tests passing)
  - `test_phase4_phase5_presets_and_auto.py` (8 tests passing)
  - `test_e2e_preprocessing_routes.py` (1 test passing)
  - Full backend test suite: 74 of 74 tests passing.
