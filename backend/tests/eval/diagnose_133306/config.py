"""Paths, constants and shared helpers for the diagnostic harness.

Supports a multi-file suite. Every measurement is parameterised by a
:class:`FileSpec`, and :func:`set_active` swaps the module-level globals so the
existing single-file call sites keep working unchanged.

Nothing here mutates ``backend/app/**``. The harness imports production functions
the same way ``backend/app/api/routes.py`` does, and reaches into them only through
``contextlib``-safe monkeypatches declared in :mod:`patches`.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
BACKEND = ROOT / "backend"
OUT_ROOT = ROOT / "eval_results"


@dataclass
class FileSpec:
    """One evaluated recording and where its artifacts live."""

    key: str
    audio: Path
    ref_json: Path
    out_dir: Path
    shipped_md: Optional[Path] = None
    shipped_improved_md: Optional[Path] = None
    _ref_cache: List[Dict] = field(default_factory=list, repr=False)

    @property
    def artifact_dir(self) -> Path:
        return self.out_dir / "artifacts"

    @property
    def word_dir(self) -> Path:
        return self.out_dir / "word_alignments"

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    def ensure_dirs(self) -> None:
        for d in (self.out_dir, self.artifact_dir, self.word_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def ref(self) -> List[Dict]:
        if not self._ref_cache:
            self._ref_cache.append(read_json(self.ref_json)["segments"])
        return self._ref_cache[0]

    def duration(self) -> float:
        segs = self.ref()
        if not segs:
            return 0.0
        d = read_json(self.ref_json).get("duration_seconds")
        if d:
            return float(d)
        return max(float(s["end"]) for s in segs)


def _spec(key: str, audio: str, ref: str, out: str,
          md: Optional[str] = None, md_imp: Optional[str] = None) -> FileSpec:
    return FileSpec(
        key=key,
        audio=ROOT / audio,
        ref_json=ROOT / ref,
        out_dir=OUT_ROOT / out,
        shipped_md=(ROOT / md) if md else None,
        shipped_improved_md=(ROOT / md_imp) if md_imp else None,
    )


SUITE: Dict[str, FileSpec] = {
    "133306": _spec("133306", "20261011_133306.m4a", "transcription_20261011_133306.json",
                    "diag_133306", "transcription_20261011_133306.md",
                    "transcription_20261011_133306_IMPROVED.md"),
    "nlp_5min": _spec("nlp_5min", "audio_nlp_1_5min.wav", "transcription_nlp_5min.json",
                      "diag_nlp_5min", "transcription_nlp_5min.md"),
}

PRIMARY = "133306"
VALIDATION = "nlp_5min"

# --- active-file globals (rebound by set_active) ---------------------------
ACTIVE: FileSpec = SUITE[PRIMARY]
OUT_DIR: Path = ACTIVE.out_dir
ARTIFACT_DIR: Path = ACTIVE.artifact_dir
CACHE_DIR: Path = ACTIVE.cache_dir
WORD_DIR: Path = ACTIVE.word_dir
AUDIO_SOURCE: Path = ACTIVE.audio
REF_JSON: Path = ACTIVE.ref_json
SHIPPED_MD: Optional[Path] = ACTIVE.shipped_md
SHIPPED_IMPROVED_MD: Optional[Path] = ACTIVE.shipped_improved_md


def set_active(key: str) -> FileSpec:
    """Point every module-level path global at one suite member."""
    global ACTIVE, OUT_DIR, ARTIFACT_DIR, CACHE_DIR, WORD_DIR
    global AUDIO_SOURCE, REF_JSON, SHIPPED_MD, SHIPPED_IMPROVED_MD
    spec = SUITE[key]
    spec.ensure_dirs()
    ACTIVE = spec
    OUT_DIR = spec.out_dir
    ARTIFACT_DIR = spec.artifact_dir
    CACHE_DIR = spec.cache_dir
    WORD_DIR = spec.word_dir
    AUDIO_SOURCE = spec.audio
    REF_JSON = spec.ref_json
    SHIPPED_MD = spec.shipped_md
    SHIPPED_IMPROVED_MD = spec.shipped_improved_md
    return spec


TERM_MAP = OUT_ROOT / "diag_133306" / "term_map.json"
FROZEN = OUT_ROOT / "frozen_v1"

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
ARM_A_ORACLE = "A_oracle"  # production functions, oracle speaker count
ARM_A_BUILDER = "A_builder"  # production diarization + uniform words -> ref-text builder check
ARM_B_BISECTION = "B_bisection"
ARM_C_WHISPER = "C_whisper_timing"
ARM_D_PER_TURN = "D_per_turn"


def ensure_dirs() -> None:
    ACTIVE.ensure_dirs()
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
    return ACTIVE.ref()


def ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def fmt_delta(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"
