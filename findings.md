# Findings & Architectural Notes

## Environment Findings
- OS: Windows 11
- Python: 3.13.14 (system) and 3.13 in `backend/.venv` (has PyTorch 2.6.0+cu124, torchaudio, faster-whisper 1.2.1, pyannote-audio 4.0.7, ctranslate2 4.8.2, etc.)
- FFmpeg: `backend/bin/ffmpeg.exe` is FFmpeg 7.1-essentials_build-www.gyan.dev
- Available FFmpeg filters: `afftdn` (FFT denoiser), `anlmdn` (non-local means denoiser), `arnndn` (RNN speech denoiser), `loudnorm` (EBU R128 loudness normalization), `astats`, `highpass`, `lowpass`.
- Audio test assets available: `audio_nlp_1_5min.wav`, `20261011_133306.m4a`, `podcast.mp3`, `denoised_audio_levels/`.

## Key Invariants to Respect
1. Source immutability: all outputs derived strictly from original decoded intermediate.
2. Separation of consumer paths:
   - ASR profile: speech-focused filtering (if requested), safe gain, 16kHz mono.
   - Speaker profile: canonical raw audio (16kHz mono, uncompressed, undenoised) so Pyannote / embedding features are untainted.
   - Playback profile: preserves spatial layout/channels, two-pass EBU R128 loudness normalization (-16 LUFS stereo, -19 LUFS mono, TP <= -1.5 dBTP).
3. Two-pass loudness must use a lossless intermediate and parse FFmpeg's linear vs dynamic normalization decision.
4. Channel downmixing must be phase-correlation aware: if correlation < -0.2 (anti-phase), do NOT average; pick best channel.
5. Ingestion must do a bounded full decode to catch stream corruption, clipping, silence, and rumble before applying filters.
6. Execution manifest must be schema-versioned and written atomically alongside output files.
7. Presets must be fixed and immutable. Missing backend must report capability error (503/500) or declared fallback in manifest.
