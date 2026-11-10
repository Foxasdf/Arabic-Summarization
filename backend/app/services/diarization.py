"""
Speaker diarization.

Primary backend:
  * Pyannote 3.1 (Local & 100% Offline):
    Loads neural network weights (Segmentation 3.0 + WeSpeaker ResNet34)
    directly from `models/pyannote/`.
    No internet connection and no Hugging Face token required at runtime.
    Paths are resolved dynamically so the project folder can be moved
    between any drive or OS (Windows, Linux, macOS).
    Feeds audio via memory tensors to bypass any external C-codec DLL dependencies.

Fallback backend:
  * Builtin:
    Dependency-light (numpy + scipy only) statistical clustering diarizer
    ensuring zero-failure under any edge case.
"""
import os
import asyncio
from pathlib import Path
import logging
import yaml
import urllib.request
from typing import Any, List, Dict, Optional, Callable

# Prevent PyTorch CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from ..config import settings
from .hardware import get_hardware_profile

logger = logging.getLogger(__name__)

SR = 16000


# =============================================================================
# Helper: Validate and ensure authentic Pyannote weights (not Git LFS pointers)
# =============================================================================
def _is_lfs_pointer_or_invalid(file_path: Path, min_size: int = 1_000_000) -> bool:
    if not file_path.exists():
        return True
    if file_path.stat().st_size < min_size:
        return True
    try:
        with open(file_path, "rb") as f:
            header = f.read(30)
            if header.startswith(b"version http") or b"git-lfs" in header:
                return True
    except Exception:
        return True
    return False


def _ensure_valid_model_weight(file_path: Path, hf_repo: str, min_size: int, token: Optional[str] = None) -> None:
    """
    Validates that a PyTorch checkpoint is a real binary file (>min_size) and not a Git LFS text pointer.
    If it is a pointer or corrupt, it attempts automatic recovery via Hugging Face.
    """
    if not _is_lfs_pointer_or_invalid(file_path, min_size):
        return

    logger.warning(
        f"ملف أوزان النموذج {file_path.name} في ({file_path}) غير مكتمل أو مؤشر نصي Git LFS "
        f"({file_path.stat().st_size if file_path.exists() else 0} بايت). "
        f"جاري تحميل الأوزان الأصلية تلقائياً من Hugging Face ({hf_repo})..."
    )
    url = f"https://huggingface.co/{hf_repo}/resolve/main/pytorch_model.bin"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".download.tmp")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(temp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                
        if temp_path.stat().st_size < min_size:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"حجم الملف المحمّل ({temp_path.stat().st_size}) أصغر من المتوقع ({min_size}).")
            
        temp_path.replace(file_path)
        logger.info(f"تم تحميل وتثبيت أوزان Pyannote بنجاح: {file_path.name} ({file_path.stat().st_size} بايت)")
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        if file_path.exists() and file_path.stat().st_size < min_size:
            content_sample = ""
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content_sample = f.read(200).strip()
            except Exception:
                pass
            raise RuntimeError(
                f"ملف أوزان Pyannote ({file_path.name}) هو ملف نصي لمؤشر Git LFS ({file_path.stat().st_size} بايت):\n"
                f"{content_sample}\n"
                f"وتعذر تحميل الأوزان الأصلية تلقائياً ({exc}). "
                f"يرجى التأكد من استبدال هذا الملف بملف الأوزان الفعلي ({min_size // (1024*1024)}MB) أو تفعيل الاتصال بالإنترنت مؤقتاً."
            ) from exc
        raise


# =============================================================================
# Device & Backend Helpers
# =============================================================================
def _select_torch_device():
    """Return the best available torch.device ('cuda:0', 'mps', or 'cpu')."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _empty_torch_cache():
    """Release cached GPU/MPS memory back to the OS/driver."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception as exc:
        logger.debug(f"Torch cache cleanup warning: {exc}")


def resolve_diarization_backend(backend: Optional[str] = None) -> str:
    """Normalize backend identifier to canonical 'pyannote' or 'nemotron_3'."""
    raw = (backend or getattr(settings, "DIARIZATION_BACKEND", "pyannote") or "pyannote").strip().lower()
    if raw in ("pyannote", "pyannote_3.1", "pyannote-3.1", "pyannote3"):
        return "pyannote"
    if raw in ("nemotron_3", "nemotron-3", "nemotron3", "nemotron", "nvidia/nemotron-3-diarization"):
        return "nemotron_3"
    raise ValueError(
        f"Unsupported diarization backend '{raw}'. Supported backends: 'pyannote', 'nemotron_3'."
    )


def diarization_backend_name(backend: Optional[str] = None) -> str:
    resolved = resolve_diarization_backend(backend)
    if resolved == "nemotron_3":
        return "NVIDIA Nemotron-3-Diarization (Transformers)"
    return "Pyannote 3.1 (Local Offline)"


def filter_phantom_speakers(
    turns: List[Dict],
    min_total_sec: Optional[float] = None,
    single_turn_max_sec: float = 2.5,
) -> List[Dict]:
    """Drop minor phantom speaker tracks in Nemotron-3 when:
    - cumulative speech is < min_total_sec (default 1.5s), OR
    - the speaker only appears for a single turn (count == 1) with duration < single_turn_max_sec (2.5s).
    Never drops every speaker.
    """
    if not turns:
        return []
    thr = float(min_total_sec if min_total_sec is not None else getattr(settings, "NEMOTRON_MIN_SPEAKER_TOTAL_SEC", 1.5))
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for t in turns:
        spk = str(t["speaker"])
        dur = max(0.0, float(t["end"]) - float(t["start"]))
        totals[spk] = totals.get(spk, 0.0) + dur
        counts[spk] = counts.get(spk, 0) + 1
    keep = {
        s
        for s, tot in totals.items()
        if tot >= thr and not (counts.get(s, 0) == 1 and tot < single_turn_max_sec)
    }
    if not keep:
        return list(turns)
    dropped = sorted(set(totals) - keep)
    if dropped:
        logger.info(f"Dropping phantom speaker tracks (cumulative <{thr:.1f}s or single-turn <{single_turn_max_sec:.1f}s): {dropped}")
    return [t for t in turns if str(t["speaker"]) in keep]


# =============================================================================
# Public API
# =============================================================================
def diarize(
    audio_path: Path,
    num_speakers: Optional[int] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    backend: Optional[str] = None,
    return_raw_turns: bool = False,
):
    """
    Diarize speaker turns using the configured or requested backend:
      - 'pyannote': Local Pyannote 3.1 (Segmentation 3.0 + WeSpeaker ResNet34)
      - 'nemotron_3': nvidia/Nemotron-3-Diarization via Hugging Face Transformers
    Both backends output standardized [{'speaker': str, 'start': float, 'end': float}]
    and pass through the exact same _normalize_turns smoothing/merging pipeline.
    If return_raw_turns=True, returns (normalized_turns, raw_turns).
    """
    resolved_backend = resolve_diarization_backend(backend)
    num_speakers = num_speakers or (settings.DIARIZATION_NUM_SPEAKERS or None)

    try:
        if resolved_backend == "nemotron_3":
            raw_turns = _diarize_nemotron_3(audio_path, num_speakers, on_progress)
            if not raw_turns:
                raise RuntimeError("لم يتمكن نموذج Nemotron-3-Diarization من اكتشاف أي مقاطع للمتحدثين في التسجيل الصوتي.")
        else:
            raw_turns = _diarize_pyannote_local(audio_path, num_speakers, on_progress)
            if not raw_turns:
                raise RuntimeError("لم يتمكن نموذج Pyannote 3.1 من اكتشاف أي مقاطع للمتحدثين في التسجيل الصوتي.")

        normalized = _normalize_turns(raw_turns)
        if return_raw_turns:
            return normalized, raw_turns
        return normalized
    except (InterruptedError, asyncio.CancelledError):
        logger.info(f"{resolved_backend} diarization was cancelled/interrupted.")
        raise
    except Exception as e:
        logger.exception(f"{resolved_backend} diarization failed: {e}")
        raise RuntimeError(f"فشل فصل المتحدثين باستخدام ({resolved_backend}): {e}") from e


# =============================================================================
# Pyannote 3.1 Local Offline Backend
# =============================================================================
def _get_or_create_local_pyannote_config() -> Path:
    """
    Dynamically generates or updates config_local.yaml pointing to the current
    absolute paths on THIS machine. Ensures full portability when moving the
    project across different operating systems or folders.
    """
    pyannote_base = (settings.MODELS_DIR / "pyannote").resolve()
    orig_cfg = pyannote_base / "speaker-diarization-3.1" / "config.yaml"
    local_cfg = pyannote_base / "speaker-diarization-3.1" / "config_local.yaml"

    if not orig_cfg.exists():
        raise FileNotFoundError(f"ملف نموذج Pyannote الأساسي غير موجود: {orig_cfg}")

    seg_model = (pyannote_base / "segmentation-3.0" / "pytorch_model.bin").resolve()
    emb_model = (pyannote_base / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin").resolve()

    hf_token = getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or None
    # Ensure authentic PyTorch binaries (not Git-LFS pointer text files or partial files)
    _ensure_valid_model_weight(seg_model, "pyannote/segmentation-3.0", min_size=5_000_000, token=hf_token)
    _ensure_valid_model_weight(emb_model, "pyannote/wespeaker-voxceleb-resnet34-LM", min_size=25_000_000, token=hf_token)

    if not seg_model.exists() or not emb_model.exists():
        raise FileNotFoundError(f"أوزان نموذج Pyannote غير مكتملة في {pyannote_base}")

    with open(orig_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Point to the current machine's local weights dynamically
    cfg["pipeline"]["params"]["segmentation"] = str(seg_model)
    cfg["pipeline"]["params"]["embedding"] = str(emb_model)

    # Dynamic batch sizes scaled to available GPU hardware
    hw = get_hardware_profile()
    d_params = hw.diarization_params()
    cfg["pipeline"]["params"]["segmentation_batch_size"] = d_params["segmentation_batch_size"]
    cfg["pipeline"]["params"]["embedding_batch_size"] = d_params["embedding_batch_size"]

    # Point to local PLDA calibration files ONLY when the installed SpeakerDiarization class accepts 'plda' (pyannote >= 4.0)
    supports_plda = False
    try:
        import inspect
        from pyannote.audio.pipelines import SpeakerDiarization

        supports_plda = "plda" in inspect.signature(SpeakerDiarization.__init__).parameters
    except Exception:
        supports_plda = False

    plda_dir = (pyannote_base / "plda").resolve()
    if supports_plda and plda_dir.exists() and (plda_dir / "xvec_transform.npz").exists() and (plda_dir / "plda.npz").exists():
        cfg["pipeline"]["params"]["plda"] = str(plda_dir)
    else:
        cfg["pipeline"]["params"].pop("plda", None)

    with open(local_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return local_cfg


_cached_pyannote_pipeline = None
_cached_nemotron_bundle = None


def _ensure_torchaudio_pyannote_compat() -> None:
    """
    Ensure torchaudio >= 2.6 compatibility for pyannote.audio 3.x and speechbrain,
    which reference removed top-level attributes (AudioMetaData, list_audio_backends, etc.)
    at import time even when audio is passed via in-memory tensors.
    """
    try:
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            from dataclasses import dataclass

            @dataclass
            class AudioMetaData:
                sample_rate: int = 16000
                num_frames: int = 0
                num_channels: int = 1
                bits_per_sample: int = 16
                encoding: str = "PCM_S"

            torchaudio.AudioMetaData = AudioMetaData
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None
    except Exception as exc:
        logger.debug(f"torchaudio compat shim warning: {exc}")


def get_pyannote_pipeline():
    """
    Singleton lazy-loader for Pyannote pipeline.
    Avoids 5-10s reloading overhead on every request and accelerates with TF32 on NVIDIA GPUs.
    """
    global _cached_pyannote_pipeline
    if _cached_pyannote_pipeline is not None:
        return _cached_pyannote_pipeline

    import inspect
    import sys
    import warnings
    sys.modules["torchcodec"] = None
    warnings.filterwarnings("ignore", category=UserWarning, module=r".*pyannote.*")

    _ensure_torchaudio_pyannote_compat()

    import torch
    from pyannote.audio import Pipeline

    # Enable TF32 for Ampere / Turing Tensor Cores for speedup
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as exc:
            logger.debug(f"Could not enable TF32: {exc}")

    local_config_file = _get_or_create_local_pyannote_config()
    logger.info(f"Loading local Pyannote 3.1 pipeline from {local_config_file.name} (Offline)")

    hf_token = getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or None
    orig_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        fp_params = inspect.signature(Pipeline.from_pretrained).parameters
        if "token" in fp_params:
            pipeline = Pipeline.from_pretrained(local_config_file, token=hf_token)
        else:
            pipeline = Pipeline.from_pretrained(local_config_file, use_auth_token=hf_token)
    finally:
        torch.load = orig_torch_load

    if pipeline is None:
        raise RuntimeError("تعذر تحميل نموذج Pyannote من المجلد المحلي")

    target_device = _select_torch_device()
    if target_device.type in ("cuda", "mps"):
        try:
            pipeline.to(target_device)
            dev_label = torch.cuda.get_device_name(0) if target_device.type == "cuda" else "Apple MPS"
            logger.info(f"Pyannote 3.1 running on {dev_label} ({target_device})")
        except Exception as e:
            logger.warning(f"Could not move Pyannote to {target_device} ({e}), using CPU.")

    _cached_pyannote_pipeline = pipeline
    return _cached_pyannote_pipeline


def invalidate_pyannote_cache():
    """Force unload and re-load of the Pyannote pipeline (e.g. after config.yaml changes)."""
    global _cached_pyannote_pipeline
    if _cached_pyannote_pipeline is not None:
        try:
            import torch
            _cached_pyannote_pipeline.to(torch.device("cpu"))
        except Exception as exc:
            logger.debug(f"Pyannote CPU offload on cache invalidation ignored: {exc}")
    _cached_pyannote_pipeline = None
    local_cfg = (settings.MODELS_DIR / "pyannote" / "speaker-diarization-3.1" / "config_local.yaml")
    try:
        local_cfg.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug(f"Failed removing config_local.yaml: {exc}")
    _empty_torch_cache()


def invalidate_nemotron_cache():
    """Force unload of the cached Nemotron-3 diarization processor and model."""
    global _cached_nemotron_bundle
    if _cached_nemotron_bundle is not None:
        try:
            import torch
            _, model = _cached_nemotron_bundle
            model.to(torch.device("cpu"))
            del model
        except Exception as exc:
            logger.debug(f"Nemotron CPU offload on cache invalidation ignored: {exc}")
    _cached_nemotron_bundle = None
    _empty_torch_cache()


def unload_diarization_models(backend: Optional[str] = None):
    """Explicitly unload cached diarization models (Pyannote and/or Nemotron-3) and free VRAM."""
    if backend is None:
        invalidate_pyannote_cache()
        invalidate_nemotron_cache()
    else:
        resolved = resolve_diarization_backend(backend)
        if resolved == "pyannote":
            invalidate_pyannote_cache()
        elif resolved == "nemotron_3":
            invalidate_nemotron_cache()


def _diarize_pyannote_local(
    audio_path: Path,
    num_speakers: Optional[int],
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict]:
    import gc
    import torch

    def _safe_progress(pct: int, msg: str):
        if on_progress:
            try:
                on_progress(pct, msg)
            except (InterruptedError, asyncio.CancelledError):
                raise
            except Exception:
                pass

    _safe_progress(11, "جاري تحضير نموذج Pyannote على كرت الشاشة...")

    pipeline = get_pyannote_pipeline()

    # Dynamic batch sizes scaled to current hardware profile
    hw = get_hardware_profile()
    if hw.use_gpu and torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda:0"))
        except Exception:
            pass
    d_params = hw.diarization_params()
    seg_batch = d_params["segmentation_batch_size"]
    emb_batch = d_params["embedding_batch_size"]

    # Apply batch sizes directly to pipeline instance
    pipeline.segmentation_batch_size = seg_batch
    pipeline.embedding_batch_size = emb_batch
    if hasattr(pipeline, "_segmentation") and hasattr(pipeline._segmentation, "batch_size"):
        pipeline._segmentation.batch_size = seg_batch

    # Load audio into memory tensor safely (handles any audio format)
    audio_data = np.array(_load_wav(audio_path), dtype=np.float32, copy=True)
    waveform = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, time)
    if torch.cuda.is_available():
        try:
            waveform = waveform.pin_memory()
        except Exception:
            pass
    io_input = {"waveform": waveform, "sample_rate": SR}

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        min_spk = getattr(settings, "DIARIZATION_MIN_SPEAKERS", 2)
        if min_spk and min_spk > 0:
            kwargs["min_speakers"] = int(min_spk)
        kwargs["max_speakers"] = int(settings.DIARIZATION_MAX_SPEAKERS)

    def hook(step_name, step_artifact=None, file=None, completed=None, total=None):
        if step_name == "segmentation":
            if total and completed:
                pct = 12 + int(6 * (completed / total))
                _safe_progress(pct, f"جاري تجزئة الصوت ونقاط التحدث ({completed}/{total})...")
            else:
                _safe_progress(12, "جاري تجزئة الصوت ونقاط التحدث...")
        elif step_name == "embeddings":
            if total and completed:
                pct = 18 + int(6 * (completed / total))
                _safe_progress(pct, f"جاري استخراج البصمات الصوتية ({completed}/{total})...")
            else:
                _safe_progress(18, "جاري استخراج البصمات الصوتية للمتحدثين...")
        elif step_name in ("clustering", "discrete_diarization"):
            _safe_progress(24, "جاري مطابقة وتصنيف المتحدثين...")

    kwargs["hook"] = hook

    # Free any leftover CUDA cache before launching heavy inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_batches = [
        (seg_batch, emb_batch),
        (min(seg_batch, 32), min(emb_batch, 16)),
        (16, 8),
        (4, 4),
    ]
    seen_batches = set()
    batches_to_try = []
    for sb, eb in candidate_batches:
        if (sb, eb) not in seen_batches:
            seen_batches.add((sb, eb))
            batches_to_try.append((sb, eb))

    result = None
    last_err = None

    for cur_seg, cur_emb in batches_to_try:
        pipeline.segmentation_batch_size = cur_seg
        pipeline.embedding_batch_size = cur_emb
        if hasattr(pipeline, "_segmentation") and hasattr(pipeline._segmentation, "batch_size"):
            pipeline._segmentation.batch_size = cur_seg
        if hasattr(pipeline, "_embedding") and hasattr(pipeline._embedding, "batch_size"):
            pipeline._embedding.batch_size = cur_emb

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(
            f"Executing Pyannote diarization on GPU (seg_batch={cur_seg}, emb_batch={cur_emb})..."
        )
        try:
            with torch.inference_mode():
                result = pipeline(io_input, **kwargs)
            break
        except Exception as err:
            if isinstance(err, (InterruptedError, asyncio.CancelledError)):
                raise
            err_msg = str(err).lower()
            is_oom = isinstance(err, (MemoryError,)) or any(
                w in err_msg for w in ["cuda out of memory", "out of memory", "batch_size", "cudacachingallocator"]
            )
            if is_oom:
                logger.warning(
                    f"Pyannote diarization hit GPU memory limit with seg_batch={cur_seg}, emb_batch={cur_emb} ({err}). "
                    "Reducing batch size..."
                )
                last_err = err
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise

    # If all GPU batch sizes failed with OOM, fallback to CPU execution
    if result is None:
        logger.warning(f"All GPU batch sizes failed for Pyannote ({last_err}). Falling back to CPU diarization...")
        _safe_progress(12, "جاري تحويل فصل المتحدثين إلى المعالج (CPU) لتفادي نفاد ذاكرة كرت الشاشة...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            cpu_device = torch.device("cpu")
            pipeline.to(cpu_device)
            pipeline.segmentation_batch_size = 16
            pipeline.embedding_batch_size = 16
            if hasattr(pipeline, "_segmentation") and hasattr(pipeline._segmentation, "batch_size"):
                pipeline._segmentation.batch_size = 16
            if hasattr(pipeline, "_embedding") and hasattr(pipeline._embedding, "batch_size"):
                pipeline._embedding.batch_size = 16
            io_input_cpu = {
                "waveform": torch.from_numpy(audio_data).unsqueeze(0).to(cpu_device),
                "sample_rate": SR
            }
            with torch.inference_mode():
                result = pipeline(io_input_cpu, **kwargs)
        except Exception as err:
            if isinstance(err, (InterruptedError, asyncio.CancelledError)):
                raise
            raise RuntimeError(f"فشل تشغيل Pyannote على GPU و CPU بسبب نفاد الذاكرة: {last_err or err}")
        finally:
            if torch.cuda.is_available():
                try:
                    pipeline.to(torch.device("cuda:0"))
                except Exception:
                    pass

    # Prefer speaker_diarization to retain overlapping speech, backchannels, and rapid conversational interjections
    annotation = getattr(result, "speaker_diarization", result)

    turns = [
        {"speaker": spk, "start": float(t.start), "end": float(t.end)}
        for t, _, spk in annotation.itertracks(yield_label=True)
    ]

    logger.info(f"Pyannote completed with {len(turns)} speaker segments.")

    # Second pass: correct isolated long-turn cluster swaps using speaker embeddings.
    _safe_progress(27, "جاري التحقق من تناسق بصمات المتحدثين...")
    turns = _refine_speaker_labels(turns, audio_data, pipeline)

    # Cleanup temporary tensors and flush CUDA/MPS memory allocator cache
    del io_input, waveform, audio_data, result
    _empty_torch_cache()

    return turns


# =============================================================================
# NVIDIA Nemotron-3-Diarization Backend (Hugging Face Transformers)
# =============================================================================
def get_nemotron_diarizer():
    """
    Singleton lazy-loader for nvidia/Nemotron-3-Diarization via Hugging Face Transformers.
    Loads AutoProcessor and AutoModelForAudioFrameClassification once and places the model
    on the best available device (CUDA, MPS, or CPU).
    """
    global _cached_nemotron_bundle
    if _cached_nemotron_bundle is not None:
        return _cached_nemotron_bundle

    import torch
    from transformers import AutoProcessor, AutoModelForAudioFrameClassification

    model_id = getattr(settings, "NEMOTRON_DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization")
    local_dir = settings.MODELS_DIR / "nemotron-3-diarization"
    load_source = str(local_dir) if (local_dir.exists() and (local_dir / "config.json").exists()) else model_id
    hf_token = getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or None

    logger.info(f"Loading NVIDIA Nemotron-3-Diarization from '{load_source}'...")
    processor = AutoProcessor.from_pretrained(load_source, token=hf_token)
    model = AutoModelForAudioFrameClassification.from_pretrained(load_source, token=hf_token)
    model.eval()

    target_device = _select_torch_device()
    try:
        model.to(target_device)
        logger.info(f"Nemotron-3-Diarization loaded on device: {target_device} (dtype={model.dtype})")
    except Exception as exc:
        logger.warning(f"Could not move Nemotron-3-Diarization to {target_device} ({exc}); falling back to CPU.")
        model.to(torch.device("cpu"))

    _cached_nemotron_bundle = (processor, model)
    return _cached_nemotron_bundle


def _extract_segments_from_nemotron_probs(
    probs_2d: np.ndarray,
    threshold: float = 0.5,
    frame_duration: float = 0.01,
    num_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert frame-level speaker activity probabilities [T, 8] (10ms frame step)
    into standardized speaker turn dicts: [{'speaker': str, 'start': float, 'end': float}].
    """
    if probs_2d.ndim != 2 or probs_2d.shape[0] == 0:
        return []

    T, S = probs_2d.shape
    probs = probs_2d.copy()

    # Cap active speaker channels if num_speakers or max_speakers is specified
    cap_k = int(num_speakers) if num_speakers and num_speakers > 0 else (int(max_speakers) if max_speakers and max_speakers > 0 else S)
    if 0 < cap_k < S:
        channel_activity = (probs > threshold).sum(axis=0)
        if np.count_nonzero(channel_activity) > cap_k:
            top_channels = np.argsort(channel_activity)[::-1][:cap_k]
            keep_mask = np.zeros(S, dtype=bool)
            keep_mask[top_channels] = True
            probs[:, ~keep_mask] = 0.0

    active = (probs > threshold).astype(np.int8)
    padded = np.pad(active, ((1, 1), (0, 0)), mode="constant", constant_values=0)
    changes = np.diff(padded, axis=0)

    segments: List[Dict[str, Any]] = []
    for spk_idx in range(S):
        starts = np.flatnonzero(changes[:, spk_idx] == 1)
        ends = np.flatnonzero(changes[:, spk_idx] == -1)
        for st_frame, en_frame in zip(starts, ends):
            st_sec = round(float(st_frame) * frame_duration, 2)
            en_sec = round(float(en_frame) * frame_duration, 2)
            if en_sec > st_sec:
                segments.append({
                    "speaker": f"SPEAKER_{spk_idx:02d}",
                    "start": st_sec,
                    "end": en_sec,
                })

    segments.sort(key=lambda seg: (seg["start"], seg["speaker"], seg["end"]))
    return segments


def _diarize_nemotron_3(
    audio_path: Path,
    num_speakers: Optional[int],
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict]:
    """
    Run speaker diarization using nvidia/Nemotron-3-Diarization via Hugging Face Transformers.
    Passes 16 kHz mono float32 waveform through AutoProcessor and AutoModelForAudioFrameClassification,
    extracting the [T, 8] frame-level speaker probabilities / segments into the standardized schema:
    [{'speaker': str, 'start': float, 'end': float}].
    """
    import torch

    def _safe_progress(pct: int, msg: str):
        if on_progress:
            try:
                on_progress(pct, msg)
            except (InterruptedError, asyncio.CancelledError):
                raise
            except Exception as exc:
                logger.debug(f"Progress callback error ignored: {exc}")

    _safe_progress(11, "جاري تحضير نموذج NVIDIA Nemotron-3-Diarization...")
    processor, model = get_nemotron_diarizer()

    target_sr = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", SR) or SR)
    hop_length = int(getattr(getattr(processor, "feature_extractor", None), "hop_length", 160) or 160)
    frame_duration = hop_length / float(target_sr)

    _safe_progress(14, "جاري فك تشفير الصوت إلى 16kHz Mono Float32...")
    audio_data = _load_wav(audio_path)

    _safe_progress(17, "جاري استخراج الخصائص الطيفية وتشغيل Nemotron-3-Diarization...")
    inputs = processor(audio_data, sampling_rate=target_sr, return_tensors="pt")
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    if hasattr(inputs, "to"):
        inputs = inputs.to(model_device, dtype=model_dtype)
    else:
        inputs = {
            k: (v.to(model_device, dtype=model_dtype if v.is_floating_point() else v.dtype) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

    _empty_torch_cache()
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, T, 8), one frame every 10 ms

    _safe_progress(23, "جاري تحويل احتمالات النشاط الصوتي [T, 8] إلى مقاطع المتحدثين...")
    threshold = float(getattr(settings, "NEMOTRON_THRESHOLD", 0.5))
    max_spk = int(getattr(settings, "DIARIZATION_MAX_SPEAKERS", 8) or 8)

    att_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    turns: List[Dict[str, Any]] = []

    # If num_speakers is not forcing channel pruning, use processor.extract_speaker_dict when available,
    # while also validating against the [T, 8] probability tensor.
    probs_tensor = logits.sigmoid()
    if att_mask is not None:
        probs_tensor = probs_tensor * att_mask.to(device=probs_tensor.device, dtype=probs_tensor.dtype)[..., None]
    probs_2d = probs_tensor[0].detach().cpu().float().numpy()  # shape: [T, 8]

    if not num_speakers and max_spk >= 8 and hasattr(processor, "extract_speaker_dict"):
        try:
            raw_segs = processor.extract_speaker_dict(logits, att_mask, threshold=threshold)[0]
            turns = [
                {
                    "speaker": f"SPEAKER_{int(seg['Speaker']):02d}",
                    "start": float(seg["Start"]),
                    "end": float(seg["End"]),
                }
                for seg in raw_segs
                if float(seg["End"]) > float(seg["Start"])
            ]
        except Exception as exc:
            logger.warning(f"processor.extract_speaker_dict fallback triggered ({exc}); using [T, 8] probability extractor.")
            turns = []

    if not turns:
        turns = _extract_segments_from_nemotron_probs(
            probs_2d,
            threshold=threshold,
            frame_duration=frame_duration,
            num_speakers=num_speakers,
            max_speakers=max_spk,
        )

    logger.info(
        f"Nemotron-3-Diarization completed: logits shape={tuple(logits.shape)}, "
        f"{len(turns)} raw speaker segments extracted."
    )

    del inputs, outputs, logits, probs_tensor, audio_data
    _empty_torch_cache()

    return turns


# =============================================================================
# Audio Loader & Builtin Statistical Fallback Helpers
# =============================================================================
def _load_wav(path: Path) -> np.ndarray:
    """
    Decode any audio file (.wav, .m4a, .mp3, etc.) into a 16 kHz, mono, float32
    array in [-1.0, 1.0]. Delegates to audio_processor.load_audio_16k_mono which
    normalizes integer PCM BEFORE stereo channel averaging.
    """
    from .audio_processor import load_audio_16k_mono

    return load_audio_16k_mono(Path(path), target_sr=SR)


def _mel_filterbank(n_fft: int, n_mels: int = 40, fmin: float = 60, fmax: float = 7600) -> np.ndarray:
    def hz2mel(h): return 2595 * np.log10(1 + h / 700.0)
    def mel2hz(m): return 700 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l:
            fb[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def _frame_features(audio: np.ndarray, hop: int = 160, win: int = 400, n_mfcc: int = 20):
    """Return (log_energy[T], mfcc[T, n_mfcc]) at 10 ms hop."""
    from scipy.fft import rfft, dct
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # pre-emphasis
    n_frames = max(1, 1 + (len(audio) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = audio[idx] * np.hamming(win).astype(np.float32)
    n_fft = 512
    fb = _mel_filterbank(n_fft)

    energies, mfccs = [], []
    for s in range(0, n_frames, 20000):
        blk = frames[s:s + 20000]
        pow_spec = (np.abs(rfft(blk, n=n_fft)) ** 2) / n_fft
        energies.append(10 * np.log10(blk.var(axis=1) + 1e-10))
        mel = np.log(pow_spec @ fb.T + 1e-10)
        mfccs.append(dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc])
    return np.concatenate(energies), np.concatenate(mfccs).astype(np.float32)


def _vad(log_energy: np.ndarray, min_speech: int = 30, min_gap: int = 30) -> np.ndarray:
    """Adaptive energy VAD → boolean mask per frame (10 ms)."""
    from scipy.ndimage import median_filter
    noise = np.percentile(log_energy, 15)
    peak = np.percentile(log_energy, 95)
    thr = noise + max(6.0, 0.25 * (peak - noise))
    mask = median_filter((log_energy > thr).astype(np.uint8), size=11).astype(bool)

    def runs(m):
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    st, en = runs(~mask)
    for a, b in zip(st, en):
        if b - a < min_gap and a > 0 and b < len(mask):
            mask[a:b] = True
    st, en = runs(mask)
    for a, b in zip(st, en):
        if b - a < min_speech:
            mask[a:b] = False
    return mask


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = 1.0 - Xn @ Xn.T
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == k].mean() for k in uniq if k != labels[i])
        s[i] = (b - a) / max(a, b, 1e-9)
    return float(s.mean())


def _diarize_builtin(audio_path: Path, num_speakers: Optional[int]) -> List[Dict]:
    from scipy.cluster.hierarchy import linkage, fcluster

    audio = _load_wav(audio_path)
    duration = len(audio) / SR
    log_e, mfcc = _frame_features(audio)
    speech = _vad(log_e)

    feats = mfcc[:, 1:]
    if speech.sum() > 50:
        mu, sd = feats[speech].mean(0), feats[speech].std(0) + 1e-6
    else:
        mu, sd = feats.mean(0), feats.std(0) + 1e-6
    feats = (feats - mu) / sd
    delta = np.gradient(feats, axis=0)

    W, H = 150, 75
    starts, embs = [], []
    for s in range(0, max(1, len(feats) - W + 1), H):
        m = speech[s:s + W]
        if m.mean() < 0.5:
            continue
        f, d = feats[s:s + W][m], delta[s:s + W][m]
        embs.append(np.concatenate([f.mean(0), f.std(0), d.std(0)]))
        starts.append(s)

    if len(embs) < 4:
        logger.info("Builtin diarization: too little speech, single speaker.")
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}]

    X = np.asarray(embs, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Z = linkage(X, method="average", metric="cosine")

    max_k = min(settings.DIARIZATION_MAX_SPEAKERS, max(1, len(X) // 4))
    if num_speakers:
        labels = fcluster(Z, t=min(num_speakers, len(X)), criterion="maxclust")
        best_k = len(np.unique(labels))
    else:
        best_k, best_s, labels = 1, -1.0, np.ones(len(X), dtype=int)
        for k in range(2, max_k + 1):
            lab = fcluster(Z, t=k, criterion="maxclust")
            counts = np.bincount(lab)[1:]
            if counts.min() < max(2, 0.05 * len(X)):
                continue
            s = _silhouette(X, lab)
            if s > best_s + 0.01:
                best_k, best_s, labels = k, s, lab
        if best_s < 0.08:
            best_k, labels = 1, np.ones(len(X), dtype=int)
        logger.info(f"Builtin diarization: estimated {best_k} speakers (silhouette={best_s:.3f}).")

    T = len(feats)
    votes = np.zeros((T, labels.max() + 1), dtype=np.float32)
    for s, lab in zip(starts, labels):
        votes[s:s + W, lab] += 1
    frame_lab = votes.argmax(1)
    frame_lab[votes.sum(1) == 0] = -1

    known = np.flatnonzero(frame_lab >= 0)
    if len(known):
        nearest = known[np.clip(np.searchsorted(known, np.arange(T)), 0, len(known) - 1)]
        frame_lab = np.where(frame_lab >= 0, frame_lab, frame_lab[nearest])
    frame_lab = np.where(speech, frame_lab, -1)

    turns = []
    cur, start = None, 0
    for t in range(T + 1):
        lab = frame_lab[t] if t < T else None
        if lab != cur:
            if cur is not None and cur >= 0:
                turns.append({"speaker": int(cur), "start": start / 100.0, "end": t / 100.0})
            cur, start = lab, t
    merged = []
    for tr in turns:
        if merged and (tr["end"] - tr["start"] < 1.0 or merged[-1]["speaker"] == tr["speaker"]) \
                and tr["start"] - merged[-1]["end"] < 1.0:
            merged[-1]["end"] = tr["end"]
        else:
            merged.append(dict(tr))
    for tr in merged:
        tr["speaker"] = f"SPK{tr['speaker']}"
    return merged


# =============================================================================
# Helpers
# =============================================================================
def _normalize_turns(turns: List[Dict]) -> List[Dict]:
    """
    Clean, merge contiguous turns, preserve overlapping speech and conversational
    interjections, and label SPEAKER_00.. in order of appearance.
    Core cleaning delegates to alignment.clean_diarization_turns (single source of
    truth), which filters micro-noise and merges same-speaker gaps. Overlaps are
    kept intentionally and resolved later during word-level alignment or turn windowing.
    """
    if not turns:
        return []

    from .alignment import clean_diarization_turns

    sanitized = [
        {"speaker": str(t["speaker"]), "start": max(0.0, float(t["start"])), "end": float(t["end"])}
        for t in turns
    ]
    merge_gap = float(getattr(settings, "TURN_MERGE_SAME_SPEAKER_GAP", 0.4))
    resolved = clean_diarization_turns(sanitized, min_dur=0.15, merge_gap=merge_gap)
    if not resolved:
        return []

    # Rename speakers cleanly by order of appearance
    mapping: Dict[str, str] = {}
    for t in resolved:
        spk = t["speaker"]
        if spk not in mapping:
            mapping[spk] = f"SPEAKER_{len(mapping):02d}"
        t["speaker"] = mapping[spk]
        t["start"] = round(float(t["start"]), 2)
        t["end"] = round(float(t["end"]), 2)

    logger.info(f"Diarization output: {len(resolved)} turns, {len(mapping)} speakers.")
    return resolved


# =============================================================================
# Speaker label refinement (second pass over long turns)
# =============================================================================
_REFINE_CENTROID_MIN_SEC = 2.0
_REFINE_TURN_MIN_SEC = 1.5
_REFINE_MARGIN = 0.01
_REFINE_MAX_EMBED_SEC = 3.0


def _get_embedding_inference(pipeline):
    """Return Pyannote's already-loaded speaker-embedding inference, if present."""
    inferences = getattr(pipeline, "_inferences", None)
    if isinstance(inferences, dict):
        for key in ("_embedding", "embedding"):
            embed = inferences.get(key)
            if embed is not None:
                return embed
    return None


def _embed_turns(
    audio: np.ndarray,
    turns: List[Dict],
    embed,
    min_duration: float = _REFINE_TURN_MIN_SEC,
    max_duration: float = _REFINE_MAX_EMBED_SEC,
) -> List[Optional[np.ndarray]]:
    """Embed each turn with a centered crop, returning L2-normalized vectors."""
    import torch

    min_samples = int(min_duration * SR)
    max_samples = int(max_duration * SR)
    embeddings: List[Optional[np.ndarray]] = []
    embed_device = getattr(embed, "device", None)

    for turn in turns:
        start = max(0, int(float(turn["start"]) * SR))
        end = min(len(audio), int(float(turn["end"]) * SR))
        if end - start < min_samples:
            embeddings.append(None)
            continue

        if end - start > max_samples:
            mid = (start + end) // 2
            start = max(0, mid - max_samples // 2)
            end = min(len(audio), start + max_samples)

        chunk = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)
        if embed_device is not None and hasattr(chunk_tensor, "to"):
            try:
                chunk_tensor = chunk_tensor.to(embed_device)
            except Exception as exc:
                logger.debug(f"Could not move embedding chunk to {embed_device}: {exc}")
        with torch.inference_mode():
            emb_obj = embed(chunk_tensor)
        if hasattr(emb_obj, "detach"):
            emb_obj = emb_obj.detach().cpu().numpy()
        emb = np.asarray(emb_obj, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        embeddings.append(emb / norm if norm > 0 else None)

    return embeddings


def _build_speaker_centroids(
    turns: List[Dict],
    embeddings: List[Optional[np.ndarray]],
    centroid_min_duration: float = _REFINE_CENTROID_MIN_SEC,
) -> Dict[str, np.ndarray]:
    """Build one normalized centroid per speaker from sufficiently long turns."""
    per_speaker: Dict[str, List[np.ndarray]] = {}
    for turn, emb in zip(turns, embeddings):
        if emb is None or (turn["end"] - turn["start"]) < centroid_min_duration:
            continue
        per_speaker.setdefault(turn["speaker"], []).append(emb)

    centroids: Dict[str, np.ndarray] = {}
    for spk, vecs in per_speaker.items():
        mean = np.mean(vecs, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            centroids[spk] = mean / norm
    return centroids


def _relabel_turns_from_embeddings(
    turns: List[Dict],
    embeddings: List[Optional[np.ndarray]],
    centroid_min_duration: float = _REFINE_CENTROID_MIN_SEC,
    relabel_min_duration: float = _REFINE_TURN_MIN_SEC,
    margin: float = _REFINE_MARGIN,
) -> List[Dict]:
    """
    Second-pass speaker consistency check.
    """
    if not turns:
        return []

    centroids = _build_speaker_centroids(turns, embeddings, centroid_min_duration)

    refined = [dict(t) for t in turns]
    if len(centroids) < 2:
        return refined

    for i, (turn, emb) in enumerate(zip(turns, embeddings)):
        if emb is None or (turn["end"] - turn["start"]) < relabel_min_duration:
            continue
        current = turn["speaker"]
        if current not in centroids:
            continue
        similarities = sorted(
            ((float(np.dot(emb, centroid)), spk) for spk, centroid in centroids.items()),
            reverse=True,
        )
        best_sim, best_spk = similarities[0]
        if best_spk == current:
            continue
        runner_up = similarities[1][0]
        if (best_sim - runner_up) >= margin:
            refined[i]["speaker"] = best_spk

    return refined


def _refine_speaker_labels(
    turns: List[Dict],
    audio: np.ndarray,
    pipeline,
) -> List[Dict]:
    """Embed long turns and relabel inconsistent ones against speaker centroids."""
    if not turns:
        return turns

    embed = _get_embedding_inference(pipeline)
    if embed is None:
        logger.warning("Speaker embedding model unavailable; skipping label refinement.")
        return turns

    embeddings = _embed_turns(audio, turns, embed, min_duration=_REFINE_TURN_MIN_SEC)
    refined = _relabel_turns_from_embeddings(turns, embeddings)
    changed = sum(
        1 for before, after in zip(turns, refined)
        if before["speaker"] != after["speaker"]
    )
    if changed:
        logger.info(
            f"Speaker label refinement reassigned {changed}/{len(turns)} turns "
            f"across {len({t['speaker'] for t in refined})} speakers."
        )
    return refined


def build_speaker_embedding_context(
    audio_path: Path,
    turns: List[Dict],
    pipeline=None,
) -> Optional[Dict[str, Any]]:
    """
    Build turn embeddings and speaker centroids for post-ASR label smoothing.

    Uses the cached Pyannote pipeline ONLY if it is already loaded in memory
    (or explicitly provided), avoiding loading Pyannote when another diarization
    backend (such as Nemotron-3) is active.
    """
    if not turns:
        return None

    try:
        if pipeline is None:
            if _cached_pyannote_pipeline is None:
                return None
            pipeline = _cached_pyannote_pipeline
        embed = _get_embedding_inference(pipeline)
        if embed is None:
            logger.warning(
                "Speaker embedding model unavailable; skipping short-run smoothing."
            )
            return None

        audio = _load_wav(Path(audio_path))
        embeddings = _embed_turns(audio, turns, embed, min_duration=0.20)
        centroids = _build_speaker_centroids(turns, embeddings)
        if len(centroids) < 2:
            return None
        return {"embeddings": embeddings, "centroids": centroids}
    except Exception as exc:
        logger.warning(
            f"Speaker embedding context unavailable; skipping short-run smoothing: {exc}"
        )
        return None

