"""
Hardware adaptation layer.

Detects RAM, CPU threads, NVIDIA VRAM and whether the installed llama-cpp-python
build actually supports GPU offload, then derives safe runtime parameters
(context window, GPU layers, threads, batch size) for the ASR and LLM engines.
"""
import os
import sys
import shutil
import ctypes
import logging
import subprocess
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


def _ram_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb) without requiring psutil."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return vm.total / 1024**3, vm.available / 1024**3
    except Exception:
        pass
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024**3, stat.ullAvailPhys / 1024**3
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        return pages * size / 1024**3, avail * size / 1024**3
    except Exception:
        return 8.0, 4.0


def _vram_gb() -> tuple[Optional[str], float, float]:
    """Return (gpu_name, total_gb, free_gb) via nvidia-smi (0 if no NVIDIA GPU)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, 0.0, 0.0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if not out:
            return None, 0.0, 0.0
        name, total, free = [p.strip() for p in out[0].split(",")]
        return name, float(total) / 1024, float(free) / 1024
    except Exception:
        return None, 0.0, 0.0


def _llama_gpu_offload() -> bool:
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


@dataclass
class HardwareProfile:
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: Optional[str]
    vram_total_gb: float
    vram_free_gb: float
    llama_gpu_offload: bool
    tier: str  # "low" | "medium" | "high"

    def as_dict(self) -> dict:
        return asdict(self)

    # ----------------------------------------------------------------- threads
    @property
    def n_threads(self) -> int:
        # leave headroom for the OS / web server; physical cores ~ logical/2
        return max(2, min(16, self.cpu_threads - 2 if self.cpu_threads > 4 else self.cpu_threads))

    @property
    def use_gpu(self) -> bool:
        return self.llama_gpu_offload and self.vram_total_gb >= 2.0

    @property
    def recommended_asr_workers(self) -> int:
        override = getattr(settings, "ASR_NUM_WORKERS", 0)
        if override > 0:
            return min(4, max(1, int(override)))
        if not self.use_gpu:
            return 2 if (self.cpu_threads >= 8 and self.ram_available_gb >= 16.0) else 1

        # Dynamic VRAM scaling:
        # Each ASR model worker takes ~1.8 - 2.8 GB VRAM.
        # Preserve at least 2.0 GB headroom for CUDA context & OS.
        avail_vram = max(0.0, self.vram_free_gb - 2.0)
        if self.vram_total_gb >= 15.0 and avail_vram >= 8.5:
            return 4  # 24GB GPUs (RTX 3090, RTX 4090, A10G)
        elif self.vram_total_gb >= 10.0 and avail_vram >= 5.5:
            return 3  # High-throughput 3x parallel speedup for 11GB - 12GB GPUs (RTX 2080 Ti, RTX 3060 12GB)
        elif self.vram_total_gb >= 7.5 and avail_vram >= 2.5:
            return 2  # 8GB GPUs (RTX 2070, RTX 2080, RTX 3070)
        return 1

    # ------------------------------------------------------------- Diarization
    def diarization_params(self) -> dict:
        """
        Dynamic batch sizes for Pyannote 3.1 diarization.
        Larger batch sizes saturate NVIDIA CUDA / Tensor Cores on long recordings,
        reducing Python kernel launch overhead by 50-75%.

        NOTE: thresholds here deliberately key off vram_total_gb (a fixed property
        of the machine) and NOT vram_free_gb (a live measurement). Keying off free
        VRAM made the batch size depend on whatever else was resident on the GPU at
        the moment of the request, and combined with TF32 matmuls that flipped
        borderline speaker embeddings — the same file intermittently diarized to a
        different speaker count across processes. Batch size now stays stable for a
        given machine, so repeated runs on identical audio are reproducible.
        """
        batch_override = getattr(settings, "DIARIZATION_BATCH_SIZE", 0)
        if batch_override > 0:
            seg_batch = int(batch_override)
            emb_batch = max(16, seg_batch // 2)
        elif not self.use_gpu:
            seg_batch = 32
            emb_batch = 32
        elif self.vram_total_gb >= 15.0:
            # 24GB GPUs (RTX 3090, 4090, A10G)
            seg_batch = 256
            emb_batch = 128
        elif self.vram_total_gb >= 10.0:
            # 11GB - 12GB GPUs (RTX 2080 Ti, RTX 3060 12GB, RTX 3080)
            seg_batch = 192
            emb_batch = 96
        elif self.vram_total_gb >= 7.5:
            # 8GB GPUs (RTX 2070, RTX 2080, RTX 3070)
            seg_batch = 96
            emb_batch = 48
        else:
            # Low VRAM (<= 6GB)
            seg_batch = 32
            emb_batch = 32

        return {
            "segmentation_batch_size": seg_batch,
            "embedding_batch_size": emb_batch,
            "use_gpu": self.use_gpu,
        }

    # --------------------------------------------------------------------- ASR
    def asr_params(self) -> dict:
        gpu_layers = 0
        if self.use_gpu:
            gpu_layers = -1 if self.vram_free_gb >= 2.8 else 16

        batch_override = getattr(settings, "ASR_BATCH_SIZE", 0)
        if batch_override > 0:
            n_batch = int(batch_override)
        elif self.vram_total_gb >= 10.0 and self.use_gpu:
            n_batch = 1024
        elif self.tier != "low":
            n_batch = 512
        else:
            n_batch = 256

        return {
            "n_ctx": 2048,  # one 25 s window ≈ 750 audio tokens + 256 output
            "n_batch": n_batch,
            "n_gpu_layers": gpu_layers,
            "n_threads": self.n_threads,
            "use_gpu": gpu_layers != 0,
            "num_workers": self.recommended_asr_workers,
        }

    # --------------------------------------------------------------------- LLM
    def llm_params(self, model_size_gb: float, max_ctx: int = 32768) -> dict:
        """
        Pick a context window that fits in memory. Falcon-H1 is a hybrid
        Mamba/attention model so its KV cache is small; we budget ~0.12 GB per
        1K tokens to stay conservative on other GGUFs too.
        """
        override = settings.LLM_CONTEXT_SIZE_OVERRIDE
        gpu_layers = 0
        if self.use_gpu:
            if self.vram_free_gb >= model_size_gb + 0.6:
                gpu_layers = -1
            elif self.vram_free_gb >= 2.0:
                fraction = min(1.0, max(0.2, (self.vram_free_gb - 0.5) / model_size_gb))
                gpu_layers = max(16, int(32 * fraction))

        budget = self.ram_available_gb - (0 if gpu_layers == -1 else model_size_gb) - 1.5
        if gpu_layers == -1:
            budget = max(budget, self.vram_free_gb - model_size_gb - 0.5)
        per_1k = 0.12
        candidates = [32768, 16384, 12288, 8192, 6144, 4096]
        n_ctx = 4096
        for c in candidates:
            if c <= max_ctx and (c / 1000) * per_1k <= budget:
                n_ctx = c
                break
        if self.tier == "low":
            n_ctx = min(n_ctx, 4096)
        elif self.tier == "medium":
            n_ctx = min(n_ctx, 8192)
        else:
            n_ctx = min(n_ctx, 16384)
        if override:
            n_ctx = int(override)

        return {
            "n_ctx": n_ctx,
            "n_batch": 512 if self.tier != "low" else 256,
            "n_gpu_layers": gpu_layers,
            "n_threads": self.n_threads,
            # response length scales with context
            "max_tokens": 1024 if n_ctx <= 4096 else 1536,
        }


@lru_cache(maxsize=1)
def _static_info():
    return _llama_gpu_offload()


def get_hardware_profile() -> HardwareProfile:
    """Probe hardware (RAM/VRAM are re-read every call because they change)."""
    total, avail = _ram_gb()
    gpu, vtotal, vfree = _vram_gb()
    offload = _static_info()
    threads = os.cpu_count() or 4

    effective = avail + (vfree if offload else 0)
    if effective >= 12 and threads >= 8:
        tier = "high"
    elif effective >= 6:
        tier = "medium"
    else:
        tier = "low"

    prof = HardwareProfile(
        cpu_threads=threads,
        ram_total_gb=round(total, 1),
        ram_available_gb=round(avail, 1),
        gpu_name=gpu,
        vram_total_gb=round(vtotal, 1),
        vram_free_gb=round(vfree, 1),
        llama_gpu_offload=offload,
        tier=tier,
    )
    logger.info(f"Hardware profile: {prof}")
    if gpu and not offload:
        logger.warning(
            f"GPU '{gpu}' detected but llama-cpp-python was built without CUDA. "
            "Reinstall with CMAKE_ARGS=\"-DGGML_CUDA=on\" to enable GPU acceleration."
        )
    return prof
