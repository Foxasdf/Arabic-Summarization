"""
Schema-versioned execution manifest generator for audio preprocessing (Production v3.1).
Records immutable pipeline audit trail, applied stages, analysis proxies, timings, and output hashes.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

MANIFEST_SCHEMA_VERSION = "3.1.0"
PIPELINE_VERSION = "3.1.0"


class AppliedStage(BaseModel):
    stage_name: str
    stage_order: int
    parameters: Dict[str, Any] = Field(default_factory=dict)
    filter_string: Optional[str] = None
    duration_ms: float = 0.0


class OutputArtifactInfo(BaseModel):
    profile: str
    file_name: str
    sample_rate: int
    channels: int
    duration_seconds: float
    file_size_bytes: int
    sha256_hash: str
    sample_peak_dbfs: Optional[float] = None
    true_peak_dbtp: Optional[float] = None
    measured_lufs: Optional[float] = None
    measured_lra: Optional[float] = None


class ExecutionManifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    job_id: str
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ffmpeg_version: str = ""
    ffmpeg_compiler: str = ""

    input_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input audio metadata, hashes, and quality flags (no private paths).",
    )
    analysis_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis proxies including estimated_snr_proxy_db, confidence, and rumble ratio.",
    )
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Requested vs effective settings, presets, backends, and fallback records.",
    )
    applied_stages: List[AppliedStage] = Field(default_factory=list)
    outputs: Dict[str, OutputArtifactInfo] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write_to_file(self, target_path: Path):
        target_path.write_text(self.to_json(), encoding="utf-8")


class ManifestManager:
    """Helper to build and compute artifact hashes for manifests."""

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
