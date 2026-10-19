"""Run the production pipeline (or a declared variant) on 133306 and cache artifacts.

Every arm goes through the *production* entry points:
  ``process_dual_profile_audio`` -> ``diarize`` -> ``transcribe_with_speakers``

Arms differ only in the knobs the plan declares (timing source, per-turn vs
continuous, speaker count). Nothing under ``backend/app/**`` is edited: the
switches are applied as instance attributes / settings overrides inside the run
and restored in ``finally``.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import config as C
from . import patches


# ---------------------------------------------------------------------------
# Resource metering (adjustment #3: Arm C wall-clock + peak VRAM)
# ---------------------------------------------------------------------------
class ResourceMeter:
    """Wall clock plus peak GPU memory, sampled while a stage runs.

    Two sources are sampled because this pipeline spans two unrelated GPU stacks:
    Cohere runs in a transcribe-cpp/Vulkan *subprocess* and Arm C runs faster-whisper
    on CTranslate2/CUDA -- neither reports to ``torch.cuda``. The authoritative
    number is therefore the whole-device usage from ``nvidia-smi`` (out of process),
    with ``torch`` recorded alongside as the torch-only view.
    """

    def __init__(self, kind: str, interval: float = 0.25):
        self.kind = kind
        self.interval = interval
        self.peak_bytes: int = 0
        self.peak_device_mb: int = 0
        self.baseline_device_mb: int = 0
        self.samples = 0
        self.device_samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t0 = 0.0
        self.wall_s = 0.0
        self._smi = shutil.which("nvidia-smi")

    # -- GPU backends ------------------------------------------------------
    def _torch_peak(self) -> Optional[int]:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
        except Exception:
            return None
        return None

    def _device_used_mb(self) -> Optional[int]:
        if not self._smi:
            return None
        try:
            out = subprocess.run([self._smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode != 0 or not out.stdout.strip():
                return None
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _torch_reset(self) -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _sample(self) -> None:
        while not self._stop.is_set():
            v = self._torch_peak()
            if v is not None:
                self.peak_bytes = max(self.peak_bytes, v)
                self.samples += 1
            d = self._device_used_mb()
            if d is not None:
                self.peak_device_mb = max(self.peak_device_mb, d)
                self.device_samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMeter":
        self._torch_reset()
        base = self._device_used_mb()
        self.baseline_device_mb = base or 0
        if self._torch_peak() is not None or self._smi:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_s = round(time.perf_counter() - self.t0, 2)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        v = self._torch_peak()
        if v is not None:
            self.peak_bytes = max(self.peak_bytes, v)
        d = self._device_used_mb()
        if d is not None:
            self.peak_device_mb = max(self.peak_device_mb, d)

    def report(self) -> Dict[str, Any]:
        delta = (self.peak_device_mb - self.baseline_device_mb) if self.peak_device_mb else None
        out = {
            "wall_s": self.wall_s,
            "peak_device_vram_mb": self.peak_device_mb or None,
            "peak_vram_delta_mb": delta,
            "baseline_device_vram_mb": self.baseline_device_mb,
            "peak_torch_alloc_mb": round(self.peak_bytes / (1024 * 1024), 1) if self.peak_bytes else None,
            "vram_source": "nvidia-smi (whole device, covers Vulkan subprocess + CUDA)",
            "samples": max(self.samples, self.device_samples),
        }
        try:
            from app.services.hardware import get_hardware_profile

            out["vram_total_gb"] = round(get_hardware_profile().vram_total_gb, 2)
        except Exception:
            pass
        return out


# ---------------------------------------------------------------------------
# Canonical audio (shared by every arm, produced once)
# ---------------------------------------------------------------------------
def ensure_canonical_audio() -> Dict[str, Any]:
    """Decode once into the ASR + Speaker 16 kHz WAVs used by all arms."""
    from app.services.audio_processor import process_dual_profile_audio

    C.ensure_dirs()
    asr_wav = C.ARTIFACT_DIR / "asr.16k.wav"
    spk_wav = C.ARTIFACT_DIR / "speaker.16k.wav"
    meta_path = C.ARTIFACT_DIR / "canonical_meta.json"
    if asr_wav.exists() and spk_wav.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    ok, err = asyncio.run(
        process_dual_profile_audio(C.AUDIO_SOURCE, asr_wav, spk_wav, denoise_mode="raw")
    )
    wall = round(time.perf_counter() - t0, 2)
    if not ok:
        raise RuntimeError(f"dual-profile preprocessing failed: {err}")
    manifest = C.ARTIFACT_DIR / "asr.manifest.json"
    meta = {
        "ok": True,
        "wall_s": wall,
        "asr_wav": str(asr_wav),
        "speaker_wav": str(spk_wav),
        "asr_sha_hint": asr_wav.stat().st_size,
        "manifest": C.read_json(manifest) if manifest.exists() else None,
    }
    C.write_json(meta_path, meta)
    return meta


# ---------------------------------------------------------------------------
# Diarization (auto and oracle)
# ---------------------------------------------------------------------------
def diarize_arms(asr_wav: Path, spk_wav: Path) -> Dict[str, Any]:
    """Diarize the canonical speaker WAV; production path. Cached.

    Three runs: ``auto`` (production min/max range), ``oracle{N}`` (speaker count
    pinned to the reference's N) and ``min1`` (min_speakers=1), which reveals
    whether the auto speaker count is a natural clustering outcome or merely the
    floor of the production range.

    N comes from the reference, never a constant, so the suite covers recordings
    with different speaker counts.
    """
    from app.config import settings
    from app.services.diarization import diarize

    n_ref_spk = len({s["speaker"] for s in C.ref_segments()})
    oracle_label = f"oracle{n_ref_spk}"

    runs = {}
    for label, n in (("auto", None), (oracle_label, n_ref_spk), ("min1", None)):
        path = C.ARTIFACT_DIR / f"turns_{label}.json"
        if path.exists():
            runs[label] = {"turns": C.read_json(path), "wall_s": None, "source": "cache"}
            continue
        prev_min = settings.DIARIZATION_MIN_SPEAKERS
        try:
            if label == "min1":
                settings.DIARIZATION_MIN_SPEAKERS = 1
            t0 = time.perf_counter()
            with ResourceMeter(f"diar_{label}") as m:
                turns = diarize(spk_wav, num_speakers=n)
            timing = m.report()
        finally:
            settings.DIARIZATION_MIN_SPEAKERS = prev_min
        runs[label] = {"turns": turns, "wall_s": round(time.perf_counter() - t0, 2),
                       "resources": timing, "source": "fresh"}
        C.write_json(path, turns)
    return runs


# ---------------------------------------------------------------------------
# Production transcription (captured)
# ---------------------------------------------------------------------------
def _engine_for(model_identifier: Optional[str] = None):
    from app.models.asr import get_asr_engine

    mid = model_identifier
    if mid is None:
        from app.config import settings

        mid = getattr(settings, "COHERE_ASR_FILENAME", "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf")
    return get_asr_engine(mid)


def run_captured_asr(asr_wav: Path, timing: str = "uniform") -> Dict[str, Any]:
    """Continuous Cohere pass, instrumented for exact guardrail diffs."""
    engine = _engine_for()
    engine.load()
    try:
        with ResourceMeter(f"asr_{timing}") as m:
            res = asyncio.run(patches.run_transcription(engine, asr_wav, timing=timing))
        res["resources"] = m.report()
        return res
    finally:
        engine.unload()


def run_production_asr(asr_wav: Path) -> Dict[str, Any]:
    """Reference: the untouched production ``transcribe_with_speakers`` text path."""
    from app.services.transcription import transcribe_with_speakers

    engine = _engine_for()
    engine.load()
    try:
        with ResourceMeter("asr_production") as m:
            asr_res = engine.transcribe(asr_wav)
        info = {"text": (asr_res or {}).get("text", ""), "resources": m.report(),
                "n_text_words": len(((asr_res or {}).get("text", "") or "").split())}
        return info
    finally:
        engine.unload()


def run_segments_arm(
    asr_wav: Path,
    turns: list,
    mode: str = "continuous",
    num_speakers_note: str = "auto",
) -> Dict[str, Any]:
    """Run ``transcribe_with_speakers`` in either production or per-turn mode."""
    from app.config import settings
    from app.services.transcription import transcribe_with_speakers

    prev = settings.ASR_MODE
    if mode == "per_turn":
        settings.ASR_MODE = "per_turn"
    try:
        with ResourceMeter(f"segments_{mode}") as m:
            segs = transcribe_with_speakers(asr_wav, turns)
        return {"segments": segs, "resources": m.report(), "mode": mode,
                "num_speakers_note": num_speakers_note}
    finally:
        settings.ASR_MODE = prev
