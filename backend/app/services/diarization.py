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
from typing import List, Dict, Optional, Callable

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

SR = 16000


# =============================================================================
# Public API
# =============================================================================
def diarize(
    audio_path: Path,
    num_speakers: Optional[int] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict]:
    """
    Diarize speaker turns using Pyannote 3.1 local weights on GPU, with graceful
    fallback to builtin clustering if needed.
    """
    num_speakers = num_speakers or (settings.DIARIZATION_NUM_SPEAKERS or None)
    
    # 1. Try Pyannote (Primary & Required)
    try:
        turns = _diarize_pyannote_local(audio_path, num_speakers, on_progress)
        if turns:
            return _normalize_turns(turns)
    except Exception as e:
        logger.warning(
            f"Pyannote local diarization error ({type(e).__name__}: {e}); "
            "using builtin zero-failure diarizer fallback."
        )

    # 2. Builtin fallback (Zero Failure)
    try:
        return _normalize_turns(_diarize_builtin(audio_path, num_speakers))
    except Exception as e:
        logger.exception(f"Builtin diarization failed ({e}); falling back to single speaker.")
        from .audio_processor import get_audio_duration
        dur = get_audio_duration(audio_path) or 1e6
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float(dur)}]


def diarization_backend_name() -> str:
    pyannote_dir = settings.MODELS_DIR / "pyannote"
    if (pyannote_dir / "speaker-diarization-3.1" / "config.yaml").exists():
        return "Pyannote 3.1 (Local Offline)"
    return "Builtin (Scipy/Numpy)"


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

    if not seg_model.exists() or not emb_model.exists():
        raise FileNotFoundError(f"أوزان نموذج Pyannote غير مكتملة في {pyannote_base}")

    with open(orig_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Point to the current machine's local weights dynamically
    cfg["pipeline"]["params"]["segmentation"] = str(seg_model)
    cfg["pipeline"]["params"]["embedding"] = str(emb_model)

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

    import torch
    from pyannote.audio import Pipeline

    # Enable TF32 for Ampere (RTX 3060) Tensor Cores for 3x-5x speedup
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    local_config_file = _get_or_create_local_pyannote_config()
    logger.info(f"Loading local Pyannote 3.1 pipeline from {local_config_file.name} (Offline)")

    pipeline = Pipeline.from_pretrained(local_config_file)
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


def _diarize_pyannote_local(
    audio_path: Path,
    num_speakers: Optional[int],
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict]:
    import torch

    def _safe_progress(pct: int, msg: str):
        if on_progress:
            try:
                on_progress(pct, msg)
            except Exception:
                pass

    _safe_progress(11, "جاري تحضير نموذج Pyannote على كرت الشاشة...")

    pipeline = get_pyannote_pipeline()

    # Load audio into memory tensor safely (handles any audio format)
    audio_data = _load_wav(audio_path)
    waveform = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, time)
    io_input = {"waveform": waveform, "sample_rate": SR}

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
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

    logger.info("Executing Pyannote diarization on GPU...")
    result = pipeline(io_input, **kwargs)
    annotation = getattr(result, "speaker_diarization", result)

    turns = [
        {"speaker": spk, "start": float(t.start), "end": float(t.end)}
        for t, _, spk in annotation.itertracks(yield_label=True)
    ]

    logger.info(f"Pyannote completed with {len(turns)} speaker segments.")
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
        return data
    except Exception:
        # Fallback to ffmpeg raw audio decoding (handles mp3, m4a, flac, etc.)
        from .audio_processor import get_ffmpeg_binary
        import subprocess
        cmd = [
            get_ffmpeg_binary(), "-y", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32)


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
    """Sort, clip overlaps, merge same-speaker neighbours, absorb micro-blips, rename SPEAKER_00.. by first appearance."""
    # Filter out micro-blips < 250ms (mic pops, clicks, room noise)
    raw_turns = sorted((t for t in turns if float(t["end"]) - float(t["start"]) > 0.25), key=lambda t: float(t["start"]))
    
    # 1. Clip overlaps & initial merge of same-speaker close turns (< 1.2s gap)
    stage1: List[Dict] = []
    for t in raw_turns:
        t = {"speaker": str(t["speaker"]), "start": float(t["start"]), "end": float(t["end"])}
        if stage1 and t["start"] < stage1[-1]["end"]:
            if t["end"] <= stage1[-1]["end"]:
                continue
            t["start"] = stage1[-1]["end"]
        if stage1 and stage1[-1]["speaker"] == t["speaker"] and t["start"] - stage1[-1]["end"] < 1.2:
            stage1[-1]["end"] = max(stage1[-1]["end"], t["end"])
        else:
            stage1.append(t)

    # 2. Absorb micro-turns (< 1.0s) sandwiched between the same speaker
    # (e.g. Speaker A -> 0.4s Speaker B -> Speaker A)
    stage2: List[Dict] = []
    i = 0
    while i < len(stage1):
        cur = stage1[i]
        dur = cur["end"] - cur["start"]
        if dur < 1.0 and i > 0 and i + 1 < len(stage1) and stage2 and stage2[-1]["speaker"] == stage1[i + 1]["speaker"]:
            stage2[-1]["end"] = max(stage2[-1]["end"], cur["end"])
            i += 1
            continue
        stage2.append(cur)
        i += 1

    # 3. Final merge of same-speaker turns with gap < 1.2s
    out: List[Dict] = []
    for t in stage2:
        if out and out[-1]["speaker"] == t["speaker"] and t["start"] - out[-1]["end"] < 1.2:
            out[-1]["end"] = max(out[-1]["end"], t["end"])
        else:
            out.append(t)

    mapping: Dict[str, str] = {}
    for t in out:
        if t["speaker"] not in mapping:
            mapping[t["speaker"]] = f"SPEAKER_{len(mapping):02d}"
        t["speaker"] = mapping[t["speaker"]]
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
    logger.info(f"Diarization output: {len(out)} turns, {len(mapping)} speakers.")
    return out
