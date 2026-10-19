"""Paths, constants and shared helpers for the 133306 diagnostic harness.

Nothing here mutates ``backend/app/**``. The harness imports production functions
the same way ``backend/app/api/routes.py`` does, and reaches into them only through
``contextlib``-safe monkeypatches declared in :mod:`patches`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
BACKEND = ROOT / "backend"
OUT_DIR = ROOT / "eval_results" / "diag_133306"
ARTIFACT_DIR = OUT_DIR / "artifacts"
CACHE_DIR = OUT_DIR / "cache"
WORD_DIR = OUT_DIR / "word_alignments"

AUDIO_SOURCE = ROOT / "20261011_133306.m4a"
REF_JSON = ROOT / "transcription_20261011_133306.json"
SHIPPED_MD = ROOT / "transcription_20261011_133306.md"
SHIPPED_IMPROVED_MD = ROOT / "transcription_20261011_133306_IMPROVED.md"
TERM_MAP = OUT_DIR / "term_map.json"
FROZEN = ROOT / "eval_results" / "frozen_v1"

COLLAR = 0.25  # fixed by eval_results/frozen_v1/RULES.md; never tune
FRAME = 0.01

# Production defaults under test (captured into every run for reproducibility)
SNAPSHOT_SETTINGS = [
    "ASR_CHUNK_SECONDS",
    "ASR_OVERLAP_SECONDS",
    "ASR_MIN_SILENCE_MS",
    "ASR_MODE",
    "ASR_GUARDRAILS_ENABLED",
    "VAD_ENABLED",
    "VAD_MIN_RMS_DB",
    "DIARIZATION_NUM_SPEAKERS",
    "DIARIZATION_MIN_SPEAKERS",
    "DIARIZATION_MAX_SPEAKERS",
    "DIARIZATION_BATCH_SIZE",
    "TURN_MERGE_SAME_SPEAKER_GAP",
    "TURN_PAD_MS",
    "PIPELINE_VERSION",
]

ARM_A = "A_pipeline"        # production functions unchanged
ARM_A_ORACLE = "A_oracle3"  # production functions, oracle speaker count
ARM_A_BUILDER = "A_builder"  # production diarization + uniform words -> ref-text builder check
ARM_B_BISECTION = "B_bisection"
ARM_C_WHISPER = "C_whisper_timing"
ARM_D_PER_TURN = "D_per_turn"


def ensure_dirs() -> None:
    for d in (OUT_DIR, ARTIFACT_DIR, CACHE_DIR, WORD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def bootstrap_sys_path() -> None:
    """Import production code exactly as the FastAPI app does."""
    for p in (str(BACKEND), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def add_torch_dll_dirs() -> None:
    """Windows: torch's bundled CUDA/cuDNN DLLs must be on the search path."""
    for base in [
        BACKEND / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
    ]:
        if not base.exists():
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(base))
            except Exception:
                pass
        if str(base) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(base) + ";" + os.environ.get("PATH", "")


def settings_snapshot() -> dict:
    from app.config import settings

    return {
        "asr_model": getattr(settings, "COHERE_ASR_FILENAME", None),
        "whisper_dialectal_dir": getattr(settings, "WHISPER_DIALECTAL_DIRNAME", None),
        **{k: getattr(settings, k, None) for k in SNAPSHOT_SETTINGS},
    }


def hardware_snapshot() -> dict:
    try:
        from app.services.hardware import get_hardware_profile

        hw = get_hardware_profile()
        out = hw.as_dict()
        out["asr_params"] = hw.asr_params()
        out["diarization_params"] = hw.diarization_params()
        return out
    except Exception as e:  # pragma: no cover - diagnostic only
        return {"error": repr(e)}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ref_segments() -> list[dict]:
    return read_json(REF_JSON)["segments"]


def ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def fmt_delta(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"
