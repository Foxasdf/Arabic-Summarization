"""
FFmpeg capability probe and filter smoke testing module.
Records FFmpeg version, build details, filter availability, and verifies live filter initialization.
"""

import os
import re
import sys
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def find_ffmpeg_binary() -> str:
    """Find FFmpeg binary across operating systems."""
    is_win = sys.platform.startswith("win")
    exe = "ffmpeg.exe" if is_win else "ffmpeg"
    # Look in backend/bin
    backend_bin = Path(__file__).resolve().parents[3] / "bin"
    plat = "windows" if is_win else ("macos" if sys.platform == "darwin" else "linux")

    for cand in (backend_bin / plat / exe, backend_bin / exe):
        if cand.exists() and (is_win or os.access(cand, os.X_OK)):
            return str(cand)

    system_ffmpeg = shutil.which(exe)
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return "ffmpeg"


class FilterTestResult(BaseModel):
    filter_name: str
    is_supported: bool
    live_smoke_passed: bool
    error_message: Optional[str] = None
    parameters_tested: str = ""


class ModelAssetVerification(BaseModel):
    model_name: str
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    file_path: Optional[str] = None
    exists: bool = False
    hash_matched: bool = False
    error: Optional[str] = None


class FFmpegCapabilities(BaseModel):
    binary_path: str
    version_string: str = ""
    build_date: str = ""
    compiler: str = ""
    configuration_flags: List[str] = Field(default_factory=list)
    available_filters: Set[str] = Field(default_factory=set)
    filter_smoke_tests: Dict[str, FilterTestResult] = Field(default_factory=dict)
    models_verified: Dict[str, ModelAssetVerification] = Field(default_factory=dict)

    def is_filter_ready(self, filter_name: str) -> bool:
        """Check if filter exists and passed live execution smoke test."""
        test = self.filter_smoke_tests.get(filter_name)
        return bool(test and test.live_smoke_passed)


_CACHED_CAPABILITIES: Optional[FFmpegCapabilities] = None


def probe_ffmpeg_capabilities(
    ffmpeg_bin: Optional[str] = None,
    force_refresh: bool = False,
    extra_models: Optional[Dict[str, Dict[str, str]]] = None,
) -> FFmpegCapabilities:
    """
    Probe FFmpeg capabilities with Python subprocesses, recording version, build,
    available filters, and running live filter smoke tests.
    """
    global _CACHED_CAPABILITIES
    if _CACHED_CAPABILITIES and not force_refresh:
        return _CACHED_CAPABILITIES

    bin_path = ffmpeg_bin or find_ffmpeg_binary()
    caps = FFmpegCapabilities(binary_path=bin_path)

    # 1. Probe version and build
    try:
        ver_proc = subprocess.run(
            [bin_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
            timeout=10,
        )
        first_line = ver_proc.stdout.splitlines()[0] if ver_proc.stdout else ""
        caps.version_string = first_line.strip()

        # Extract configuration
        for line in ver_proc.stdout.splitlines():
            if line.startswith("configuration:"):
                caps.configuration_flags = [flag.strip() for flag in line.replace("configuration:", "").split()]
            elif "built with" in line:
                caps.compiler = line.strip()

    except Exception as e:
        logger.error(f"Failed to probe FFmpeg version: {e}")
        caps.version_string = f"Error: {e}"

    # 2. Probe available filters
    try:
        filt_proc = subprocess.run(
            [bin_path, "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
            timeout=10,
        )
        avail = set()
        for line in filt_proc.stdout.splitlines():
            # Lines look like: " ... afftdn            A->A       Denoise audio samples using FFT."
            parts = line.strip().split()
            if len(parts) >= 2 and ("A->A" in parts or "A->N" in parts or "N->A" in parts):
                filter_name = parts[1]
                avail.add(filter_name)
        caps.available_filters = avail
    except Exception as e:
        logger.error(f"Failed to list FFmpeg filters: {e}")

    # 3. Live smoke test filters on synthetic 1-second sine test tone
    filters_to_test = {
        "afftdn": "afftdn=nf=-25:nr=6:tn=1",
        "anlmdn": "anlmdn=s=7:p=0.002:r=0.006",
        "highpass": "highpass=f=50",
        "lowpass": "lowpass=f=7800",
        "loudnorm": "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "aresample": "aresample=16000",
        "astats": "astats=metadata=1:reset=1",
    }

    # Generate test input: 1s of 16kHz sine wave via lavfi
    for fname, test_af in filters_to_test.items():
        is_in_list = fname in caps.available_filters
        if not is_in_list:
            caps.filter_smoke_tests[fname] = FilterTestResult(
                filter_name=fname,
                is_supported=False,
                live_smoke_passed=False,
                error_message="Filter not listed in ffmpeg -filters",
                parameters_tested=test_af,
            )
            continue

        cmd = [
            bin_path,
            "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=1000:duration=1",
            "-af", test_af,
            "-f", "null",
            "-",
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="ignore",
                timeout=10,
            )
            passed = (res.returncode == 0)
            caps.filter_smoke_tests[fname] = FilterTestResult(
                filter_name=fname,
                is_supported=True,
                live_smoke_passed=passed,
                error_message=None if passed else (res.stderr[:200] if res.stderr else "Non-zero exit"),
                parameters_tested=test_af,
            )
        except Exception as e:
            caps.filter_smoke_tests[fname] = FilterTestResult(
                filter_name=fname,
                is_supported=True,
                live_smoke_passed=False,
                error_message=str(e),
                parameters_tested=test_af,
            )

    # 4. Check rnnoise / arnndn filter and models
    # arnndn requires an external .rnnn model file
    arnndn_in_list = "arnndn" in caps.available_filters
    extra_models = extra_models or {}
    arnndn_model_path = extra_models.get("arnndn", {}).get("path")
    expected_hash = extra_models.get("arnndn", {}).get("sha256")

    if arnndn_model_path and Path(arnndn_model_path).exists():
        p = Path(arnndn_model_path)
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        hash_matched = (expected_hash is None or actual_hash.lower() == expected_hash.lower())
        caps.models_verified["arnndn"] = ModelAssetVerification(
            model_name="arnndn",
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            file_path=str(p),
            exists=True,
            hash_matched=hash_matched,
            error=None if hash_matched else "Hash mismatch",
        )

        # Smoke test arnndn with model
        if hash_matched:
            cmd = [
                bin_path,
                "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=1000:duration=1",
                "-af", f"arnndn=m='{arnndn_model_path}'",
                "-f", "null",
                "-",
            ]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                caps.filter_smoke_tests["arnndn"] = FilterTestResult(
                    filter_name="arnndn",
                    is_supported=True,
                    live_smoke_passed=(res.returncode == 0),
                    parameters_tested=f"arnndn=m='{arnndn_model_path}'",
                )
            except Exception as e:
                caps.filter_smoke_tests["arnndn"] = FilterTestResult(
                    filter_name="arnndn",
                    is_supported=True,
                    live_smoke_passed=False,
                    error_message=str(e),
                )
    else:
        # arnndn model is missing
        caps.models_verified["arnndn"] = ModelAssetVerification(
            model_name="arnndn",
            file_path=arnndn_model_path,
            exists=False,
            hash_matched=False,
            error="Model file not found on disk",
        )
        caps.filter_smoke_tests["arnndn"] = FilterTestResult(
            filter_name="arnndn",
            is_supported=arnndn_in_list,
            live_smoke_passed=False,
            error_message="arnndn model file not provided or missing",
            parameters_tested="arnndn (model missing)",
        )

    _CACHED_CAPABILITIES = caps
    return caps
