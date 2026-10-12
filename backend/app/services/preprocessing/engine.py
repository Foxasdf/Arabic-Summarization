"""
Unified audio preprocessing engine (Production v3.1).
Executes independent ASR, Speaker, and Playback profile branches from immutable decoded source,
enforces strict stage ordering, runs verification checks, and publishes atomically with manifests.
"""

import os
import sys
import time
import uuid
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from pydantic import BaseModel, Field

from .contracts import (
    OutputProfile,
    DenoiseMode,
    NormalizationMode,
    ChannelStrategy,
    AudioProcessingRequest,
    AudioValidationResult,
    RequestValidationError422,
    UnsupportedMediaError415,
    ServiceCapabilityError503,
)
from .capability import probe_ffmpeg_capabilities, FFmpegCapabilities, find_ffmpeg_binary
from .presets import resolve_preset, PresetConfig
from .analyzer import AudioAnalyzer, AudioFeatures
from .loudness import LoudnessProcessor, LoudnessResult
from .manifest import (
    ExecutionManifest,
    AppliedStage,
    OutputArtifactInfo,
    ManifestManager,
    PIPELINE_VERSION,
)
from .transaction import TransactionalOutputManager, StagingJobContext

logger = logging.getLogger(__name__)


class ProcessingResult(BaseModel):
    success: bool
    job_id: str
    output_files: Dict[str, str] = Field(default_factory=dict)
    manifest_path: str = ""
    manifest: Optional[ExecutionManifest] = None
    input_duration_seconds: float = 0.0
    total_wall_time_ms: float = 0.0
    real_time_factor_rtf: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class PreprocessingEngine:
    """
    Main preprocessing pipeline orchestrator.
    """

    def __init__(
        self,
        ffmpeg_bin: Optional[str] = None,
        capabilities: Optional[FFmpegCapabilities] = None,
    ):
        self.ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()
        self.capabilities = capabilities or probe_ffmpeg_capabilities(self.ffmpeg_bin)
        self.analyzer = AudioAnalyzer(self.ffmpeg_bin)
        self.loudness_engine = LoudnessProcessor(self.ffmpeg_bin)

    def process(
        self,
        input_path: Path,
        requests: Union[AudioProcessingRequest, List[AudioProcessingRequest]],
        output_dir: Path,
        job_id: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process an uploaded audio file into one or more requested profiles.
        All profile outputs are generated independently from the immutable source.
        """
        start_time = time.perf_counter()
        req_list = [requests] if isinstance(requests, AudioProcessingRequest) else requests
        job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        all_warnings: List[str] = []
        all_errors: List[str] = []

        for req in req_list:
            all_warnings.extend(req.deprecation_warnings)

        # 1. Validation & Decode Pass
        val_result, features, audio_pcm = self.analyzer.decode_and_analyze(input_path)
        if not val_result.is_valid:
            err_msg = val_result.error_message or "Audio validation failed"
            if "empty" in err_msg.lower() or "0 bytes" in err_msg.lower() or "not exist" in err_msg.lower():
                raise RequestValidationError422(err_msg, details={"input_path": str(input_path)})
            raise UnsupportedMediaError415(err_msg, details={"input_path": str(input_path)})

        # Compute input SHA-256
        input_hash = ManifestManager.compute_file_sha256(input_path)

        # 2. Begin Transactional Staging Context
        with TransactionalOutputManager.begin_job(output_dir, job_id=job_id) as stage_ctx:
            published_files: Dict[str, str] = {}
            artifact_infos: Dict[str, OutputArtifactInfo] = {}
            applied_stages: List[AppliedStage] = []

            # Record Ingestion Stage
            applied_stages.append(
                AppliedStage(
                    stage_name="ingestion_and_decode",
                    stage_order=1,
                    parameters={
                        "input_channels": features.channels,
                        "input_sample_rate": features.sample_rate,
                        "duration_sec": features.duration_seconds,
                    },
                    filter_string=None,
                    duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )
            )

            # Process each requested profile independently
            for req in req_list:
                profile_name = req.output_profile.value
                logger.info(f"Processing profile '{profile_name}' for job '{job_id}'...")

                staged_filename = f"{profile_name}.wav"
                staged_path = stage_ctx.get_staging_path(staged_filename)

                prof_stages, art_info, prof_warn = self._process_single_profile(
                    input_path=input_path,
                    request=req,
                    features=features,
                    staged_output=staged_path,
                    stage_ctx=stage_ctx,
                )

                all_warnings.extend(prof_warn)
                applied_stages.extend(prof_stages)
                artifact_infos[profile_name] = art_info
                published_files[profile_name] = str(output_dir / staged_filename)

            # 3. Create Execution Manifest
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            total_dur_sec = max(0.001, features.duration_seconds)
            rtf = (total_duration_ms / 1000.0) / total_dur_sec

            manifest = ExecutionManifest(
                schema_version="3.1.0",
                pipeline_version=PIPELINE_VERSION,
                job_id=job_id,
                ffmpeg_version=self.capabilities.version_string,
                ffmpeg_compiler=self.capabilities.compiler,
                input_summary={
                    "sha256_hash": input_hash,
                    "duration_seconds": features.duration_seconds,
                    "sample_rate": features.sample_rate,
                    "channels": features.channels,
                    "codec": features.codec_name,
                    "file_size_bytes": val_result.file_size_bytes,
                    "is_clipped": features.is_clipped,
                    "is_silent": features.is_silent,
                    "anti_phase_detected": features.anti_phase_detected,
                    "rumble_detected": features.rumble_detected,
                },
                analysis_values={
                    "estimated_snr_proxy_db": features.estimated_snr_proxy_db,
                    "snr_confidence": features.snr_confidence,
                    "rumble_ratio_db": features.rumble_ratio_db,
                    "stereo_correlation": features.stereo_correlation,
                    "speech_ratio": features.speech_ratio,
                },
                configuration={
                    "requests": [r.model_dump() for r in req_list],
                },
                applied_stages=applied_stages,
                outputs=artifact_infos,
                performance={
                    "total_wall_time_ms": round(total_duration_ms, 2),
                    "audio_duration_seconds": features.duration_seconds,
                    "real_time_factor_rtf": round(rtf, 3),
                },
                warnings=all_warnings,
                errors=all_errors,
            )

            manifest_staging_path = stage_ctx.get_staging_path("manifest.json")
            manifest.write_to_file(manifest_staging_path)

            # 4. Atomic Publication
            stage_ctx.publish_atomically()

            manifest_final_path = str(output_dir / "manifest.json")

            return ProcessingResult(
                success=True,
                job_id=job_id,
                output_files=published_files,
                manifest_path=manifest_final_path,
                manifest=manifest,
                input_duration_seconds=features.duration_seconds,
                total_wall_time_ms=round(total_duration_ms, 2),
                real_time_factor_rtf=round(rtf, 3),
                warnings=all_warnings,
                errors=all_errors,
            )

    def _process_single_profile(
        self,
        input_path: Path,
        request: AudioProcessingRequest,
        features: AudioFeatures,
        staged_output: Path,
        stage_ctx: StagingJobContext,
    ) -> Tuple[List[AppliedStage], OutputArtifactInfo, List[str]]:
        """
        Execute audio conversion strictly following the pinned stage order:
        decode -> channel policy -> optional rumble high-pass -> backend-rate conversion
               -> denoise -> output-rate/layout conversion -> profile gain or loudness
               -> encode/write -> verify
        """
        stages: List[AppliedStage] = []
        warnings: List[str] = []
        prof = request.output_profile

        # Target layout & rate
        target_sr = request.target_sample_rate or (16000 if prof in (OutputProfile.ASR, OutputProfile.SPEAKER) else features.sample_rate)
        target_ch = request.target_channels or (1 if prof in (OutputProfile.ASR, OutputProfile.SPEAKER) else features.channels)

        # 1. Channel Policy
        # For mono downmixing: handle anti-phase stereo if detected
        channel_filter = ""
        if target_ch == 1 and features.channels > 1:
            if features.anti_phase_detected:
                # Anti-phase channels cancel if summed! Pick best measured channel
                best_ch = features.best_channel_index
                channel_filter = f"pan=mono|c0=c{best_ch}"
                warnings.append(
                    f"Anti-phase stereo detected. Downmixing selected channel {best_ch} "
                    f"to prevent destructive phase cancellation."
                )
            else:
                channel_filter = "pan=mono|c0=0.5*c0+0.5*c1" if features.channels == 2 else "aresample=matrix_encoding=none:async=1"

        # 2. Rumble Highpass Filter (conditional, speech profiles only)
        rumble_filter = ""
        if prof == OutputProfile.ASR and features.rumble_detected:
            rumble_filter = "highpass=f=50"
            warnings.append("Low-frequency rumble detected (<50Hz); gentle 50Hz high-pass applied.")

        # 3. Denoise Preset Resolution
        denoise_filter = ""
        resolved_preset: Optional[PresetConfig] = None
        fallback_msg: Optional[str] = None

        if prof == OutputProfile.SPEAKER:
            # INVARIANT: Speaker profile is NEVER denoised by default!
            pass
        elif request.denoise_mode != DenoiseMode.RAW:
            # Resolve immutable preset
            resolved_preset, fallback_msg = resolve_preset(
                request.denoise_mode,
                capabilities=self.capabilities,
                allow_fallback=True,
                features=features,
            )
            if fallback_msg:
                warnings.append(fallback_msg)
            denoise_filter = resolved_preset.filter_string

        # 4. Profile Gain / Normalization
        gain_filter = ""
        if prof == OutputProfile.ASR and request.normalization == NormalizationMode.SAFE_GAIN:
            gain_db = request.safe_gain_db or 3.0
            gain_filter = f"volume={gain_db}dB"
        elif prof == OutputProfile.SPEAKER:
            # INVARIANT: Speaker profile normalization is OFF
            pass

        # Branching for PLAYBACK vs Direct ASR/Speaker
        t0 = time.perf_counter()
        if prof == OutputProfile.PLAYBACK and request.normalization == NormalizationMode.EBU_R128:
            # Two-pass playback loudness requires intermediate post-channel, post-denoise file
            intermediate_path = stage_ctx.get_staging_path("playback_intermediate.wav")

            # Intermediate filter
            inter_filters = [f for f in (channel_filter, rumble_filter, denoise_filter) if f]
            inter_af = ",".join(inter_filters)

            cmd_inter = [self.ffmpeg_bin, "-y", "-i", str(input_path)]
            if inter_af:
                cmd_inter.extend(["-af", inter_af])
            cmd_inter.extend(["-ar", str(target_sr), "-c:a", "pcm_s16le", str(intermediate_path)])

            res_inter = subprocess.run(cmd_inter, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_inter.returncode != 0:
                raise RuntimeError(f"Failed to generate playback intermediate: {res_inter.stderr.decode('utf-8', errors='ignore')[:200]}")

            stages.append(
                AppliedStage(
                    stage_name="playback_intermediate_render",
                    stage_order=2,
                    parameters={"filters": inter_af, "target_sr": target_sr},
                    filter_string=inter_af,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
            )

            # Two-pass loudness normalization
            t_loud = time.perf_counter()
            loud_res: LoudnessResult = self.loudness_engine.process_playback_loudness(
                intermediate_wav=intermediate_path,
                output_path=staged_output,
                channels=target_ch,
                target_sample_rate=target_sr,
            )

            stages.append(
                AppliedStage(
                    stage_name="playback_two_pass_loudnorm",
                    stage_order=3,
                    parameters={
                        "target_i": loud_res.target_i,
                        "target_tp": loud_res.target_tp,
                        "mode_used": loud_res.normalization_mode_used,
                        "linear_confirmed": loud_res.linear_mode_confirmed,
                        "final_i": loud_res.final_output_i,
                        "final_tp": loud_res.final_output_tp,
                    },
                    filter_string="loudnorm (two-pass)",
                    duration_ms=round((time.perf_counter() - t_loud) * 1000, 2),
                )
            )

            if intermediate_path.exists():
                intermediate_path.unlink()

            out_i = loud_res.final_output_i
            out_tp = loud_res.final_output_tp
            out_lra = loud_res.final_output_lra

        else:
            # ASR, SPEAKER, or Playback without loudness: Single pinned filter chain
            pipeline_filters = [f for f in (channel_filter, rumble_filter, denoise_filter, gain_filter) if f]
            af_string = ",".join(pipeline_filters)

            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(input_path),
            ]
            if af_string:
                cmd.extend(["-af", af_string])
            cmd.extend([
                "-ar", str(target_sr),
                "-ac", str(target_ch),
                "-c:a", "pcm_s16le",
                str(staged_output),
            ])

            p_res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p_res.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg pipeline failed for profile '{prof.value}': {p_res.stderr.decode('utf-8', errors='ignore')[:300]}"
                )

            stages.append(
                AppliedStage(
                    stage_name=f"{prof.value}_render",
                    stage_order=2,
                    parameters={
                        "filters": af_string,
                        "target_sample_rate": target_sr,
                        "target_channels": target_ch,
                        "preset": resolved_preset.preset_id if resolved_preset else "none",
                    },
                    filter_string=af_string,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
            )

            out_i = None
            out_tp = None
            out_lra = None

        # 5. Verification stage
        val_out, feat_out, _ = self.analyzer.decode_and_analyze(staged_output)
        if not val_out.is_valid:
            raise RuntimeError(f"Output verification failed for profile '{prof.value}': {val_out.error_message}")

        # Check duration drift
        drift = abs(feat_out.duration_seconds - features.duration_seconds)
        if drift > 0.1:
            warnings.append(f"Duration drift of {drift:.2f}s detected between input and output.")

        out_hash = ManifestManager.compute_file_sha256(staged_output)

        art_info = OutputArtifactInfo(
            profile=prof.value,
            file_name=staged_output.name,
            sample_rate=feat_out.sample_rate,
            channels=feat_out.channels,
            duration_seconds=feat_out.duration_seconds,
            file_size_bytes=staged_output.stat().st_size,
            sha256_hash=out_hash,
            sample_peak_dbfs=feat_out.sample_peak_dbfs,
            true_peak_dbtp=out_tp or feat_out.true_peak_dbtp,
            measured_lufs=out_i,
            measured_lra=out_lra,
        )

        return stages, art_info, warnings
