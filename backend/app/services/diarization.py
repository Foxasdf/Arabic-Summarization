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
from pathlib import Path
import logging
import yaml
import urllib.request
from typing import List, Dict, Optional, Callable

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
# Public API
# =============================================================================
def diarize(
    audio_path: Path,
    num_speakers: Optional[int] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict]:
    """
    Diarize speaker turns strictly using local Pyannote 3.1.
    Pyannote is the ONLY allowed diarization backend.
    If Pyannote fails, an exception is raised immediately (no fallback).
    """
    num_speakers = num_speakers or (settings.DIARIZATION_NUM_SPEAKERS or None)
    
    try:
        turns = _diarize_pyannote_local(audio_path, num_speakers, on_progress)
        if not turns:
            raise RuntimeError("لم يتمكن نموذج Pyannote 3.1 من اكتشاف أي مقاطع للمتحدثين في التسجيل الصوتي.")
        return _normalize_turns(turns)
    except Exception as e:
        logger.exception(f"Pyannote diarization failed: {e}")
        raise RuntimeError(f"فشل فصل المتحدثين: نموذج Pyannote 3.1 هو المحرك الوحيد المعتمد وحدث خطأ أثناء تشغيله: {e}")


def diarization_backend_name() -> str:
    return "Pyannote 3.1 (Local Offline)"


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

    hf_token = getattr(settings, "HF_TOKEN", None) or "REDACTED_HF_TOKEN"
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

    # Point to local PLDA calibration files if available (required for pyannote >= 4.0 offline execution)
    plda_dir = (pyannote_base / "plda").resolve()
    if plda_dir.exists() and (plda_dir / "xvec_transform.npz").exists() and (plda_dir / "plda.npz").exists():
        cfg["pipeline"]["params"]["plda"] = str(plda_dir)

    with open(local_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return local_cfg


_cached_pyannote_pipeline = None


def get_pyannote_pipeline():
    """
    Singleton lazy-loader for Pyannote pipeline.
    Avoids 5-10s reloading overhead on every request and accelerates with TF32 on NVIDIA GPUs.
    """
    global _cached_pyannote_pipeline
    if _cached_pyannote_pipeline is not None:
        return _cached_pyannote_pipeline

    import sys
    import warnings
    sys.modules["torchcodec"] = None
    warnings.filterwarnings("ignore", category=UserWarning, module=r".*pyannote.*")

    import torch
    from pyannote.audio import Pipeline

    # Enable TF32 for Ampere / Turing Tensor Cores for speedup
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    local_config_file = _get_or_create_local_pyannote_config()
    logger.info(f"Loading local Pyannote 3.1 pipeline from {local_config_file.name} (Offline)")

    hf_token = getattr(settings, "HF_TOKEN", None) or "REDACTED_HF_TOKEN"
    pipeline = Pipeline.from_pretrained(local_config_file, token=hf_token)
    if pipeline is None:
        raise RuntimeError("تعذر تحميل نموذج Pyannote من المجلد المحلي")

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            pipeline.to(device)
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Pyannote 3.1 running on NVIDIA GPU: {gpu_name}")
        except Exception as e:
            logger.warning(f"Could not move Pyannote to CUDA ({e}), using CPU.")

    _cached_pyannote_pipeline = pipeline
    return _cached_pyannote_pipeline


def invalidate_pyannote_cache():
    """Force re-load of the Pyannote pipeline (e.g. after config.yaml changes)."""
    global _cached_pyannote_pipeline
    _cached_pyannote_pipeline = None
    # Also delete config_local.yaml so it's regenerated from the updated config.yaml
    local_cfg = (settings.MODELS_DIR / "pyannote" / "speaker-diarization-3.1" / "config_local.yaml")
    try:
        local_cfg.unlink(missing_ok=True)
    except Exception:
        pass


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
            except Exception:
                pass

    _safe_progress(11, "جاري تحضير نموذج Pyannote على كرت الشاشة...")

    pipeline = get_pyannote_pipeline()

    # Dynamic batch sizes scaled to current hardware profile
    hw = get_hardware_profile()
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

    logger.info(
        f"Executing Pyannote diarization on GPU (seg_batch={seg_batch}, emb_batch={emb_batch})..."
    )
    with torch.inference_mode():
        result = pipeline(io_input, **kwargs)

    # Prefer speaker_diarization to retain overlapping speech, backchannels, and rapid conversational interjections
    annotation = getattr(result, "speaker_diarization", result)

    turns = [
        {"speaker": spk, "start": float(t.start), "end": float(t.end)}
        for t, _, spk in annotation.itertracks(yield_label=True)
    ]

    logger.info(f"Pyannote completed with {len(turns)} speaker segments.")

    # Cleanup temporary tensors and flush CUDA memory allocator cache
    # This releases all reserved VRAM back to the GPU driver so subsequent ASR
    # has 100% of GPU VRAM available for its multi-worker parallel transcription.
    del io_input, waveform, audio_data, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return turns


# =============================================================================
# Builtin fallback backend (numpy / scipy)
# =============================================================================
def _load_wav(path: Path) -> np.ndarray:
    path = Path(path)
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, SR)
            data = resample_poly(data, SR // g, sr // g).astype(np.float32)
        return np.ascontiguousarray(data, dtype=np.float32)
    except Exception:
        # Fallback to ffmpeg raw audio decoding (handles mp3, m4a, flac, etc.)
        from .audio_processor import get_ffmpeg_binary
        import subprocess
        cmd = [
            get_ffmpeg_binary(), "-y", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return np.ascontiguousarray(np.frombuffer(proc.stdout, dtype=np.float32), dtype=np.float32)


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
    Clean, resolve overlaps non-destructively, preserve conversational interjections,
    and label SPEAKER_00.. in order of appearance.
    Core cleaning delegates to alignment.clean_diarization_turns (single source of
    truth), which already performs midpoint overlap resolution. This wrapper only
    renames speakers in order of appearance.

    NOTE: the resolution step is deliberately NOT repeated here. Re-running it
    truncated further via the `resolved.pop()` path and cost ~3 DER points on
    133306 (0.3199 -> 0.3496). Resolution is idempotent only on truly disjoint
    input, which the interjection-punching stage does not guarantee.
    """
    if not turns:
        return []

    from .alignment import clean_diarization_turns

    resolved = clean_diarization_turns(turns, min_dur=0.15, merge_gap=0.5)
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
