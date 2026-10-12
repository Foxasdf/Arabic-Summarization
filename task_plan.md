# Unified Audio Preprocessing Plan (Production v3.1) - Task Plan

## Overview
Implement the production-grade audio preprocessing pipeline (Production v3.1) with dual-output architecture, strict profile separation (ASR, Speaker, Playback), immutable versioned presets, two-pass EBU R128 playback loudness, phase-aware ingestion, atomic transactional publication, schema-versioned manifests, and an evaluation harness (`eval_pipeline.py`).

## Phases & Status

- [x] **Phase 0 — Repository Audit & Baseline Evaluation Harness**
  - [x] Audit current pipeline, FFmpeg version/build, filters, models, callers
  - [x] Implement capability prober with live filter/model smoke testing
  - [x] Build Arabic text normalizer (raw vs normalized WER/CER reporting)
  - [x] Create smoke set (6-8 clips: clean, silence, clipped, anti-phase, corrupt, short, stereo)
  - [x] Create quality test set from Arabic corpus recordings
  - [x] Build `backend/tools/eval_pipeline.py` with metrics: WER, CER, LUFS, LRA, sample peak, true peak, duration drift, RTF, SNR proxy
  - [x] Run and freeze baseline evaluation matrix

- [x] **Phase 1 — Request Validation, Ingestion, and Observability**
  - [x] Create typed schemas (`AudioProcessingRequest`, `OutputProfile`, `DenoiseMode`, `NormalizationMode`, etc.)
  - [x] Implement HTTP validation (reject invalid combos with 422, unknown settings, unsupported media 415/422)
  - [x] Implement legacy alias support with deprecation warnings (`none->raw`, `low->light`, `moderate->balanced`, `high->strong`, `dynamic->auto`)
  - [x] Separate metadata probe from bounded full decode validation
  - [x] Implement audio feature analyzer: clipping detector, silence analyzer, rumble detector, phase correlation analyzer
  - [x] Implement phase-aware channel downmixing for ASR (avoid anti-phase cancellation)
  - [x] Implement schema-versioned execution manifest (`manifest.json`)
  - [x] Implement transactional output manager (temp job dir -> atomic rename)
  - [x] Unit tests for Phase 1

- [x] **Phase 2 — Separate Profiles & Canonical Outputs**
  - [x] Implement independent output branch generators:
    - `ASR`: model rate/mono, optional rumble highpass, evaluated denoiser, optional static safe_gain
    - `Speaker`: canonical unprocessed 16kHz mono (or original), never denoised or compressed
    - `Playback`: preserve channels/layout, target rate, optional denoise, loudness normalization
  - [x] Enforce strict stage order: `decode -> channel policy -> rumble high-pass -> backend-rate -> denoise -> output-rate/layout -> gain/loudness -> encode/write -> verify`
  - [x] A/B test removal of fixed 7.8kHz lowpass filter
  - [x] Verification checks for each output (duration drift, format, channels, clipping)
  - [x] Unit tests for Phase 2

- [x] **Phase 3 — Playback Loudness Engine**
  - [x] Implement two-pass `loudnorm` using lossless intermediate
  - [x] First pass: measurement (`-af loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json -f null -`)
  - [x] Second pass: linear render with measured values (`measured_I`, `measured_TP`, `measured_LRA`, `measured_thresh`, `offset`, `linear=true`)
  - [x] Resample to final output rate after loudnorm
  - [x] Parse and record linear vs dynamic mode in manifest
  - [x] Edge cases: silence, short files (<3s), max boost limiter
  - [x] Independent true-peak metering verification (`<= -1.5 dBTP`)
  - [x] Unit tests for Phase 3

- [x] **Phase 4 — Denoiser Evaluation & Fixed Presets**
  - [x] Define immutable presets: `light`, `balanced`, `strong` with frozen parameters, backend, input rate
  - [x] Implement backend capability checks and declared fallback handling
  - [x] Service capability error (503/500) if required backend unavailable and no approved fallback
  - [x] Unit tests for Phase 4

- [x] **Phase 5 — Auto Mode & Quality Tuning**
  - [x] Implement multi-window non-speech VAD analysis for SNR proxy
  - [x] Confidence scoring and conservative fallback for low confidence / <5s clips
  - [x] Calibrated decision rules for preset selection
  - [x] Shadow mode / canary support
  - [x] Unit tests for Phase 5

- [x] **Phase 6 — API & Backend Service Integration**
  - [x] Wire unified preprocessor into `backend/app/services/audio_processor.py`
  - [x] Wire independent profiles into `backend/app/api/routes.py` (ASR gets ASR branch, diarization gets Speaker canonical branch)
  - [x] Update existing tests to maintain backward compatibility with legacy aliases
  - [x] Full end-to-end integration test suite

- [x] **Phase 7 — Verification, Quality Gates & Final Evaluation**
  - [x] Run full test suite (`backend/tests/`)
  - [x] Run `eval_pipeline.py` and produce final evaluation report
  - [x] Verify all acceptance criteria from Section 6
