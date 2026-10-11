"""
Abstract Base Class for ASR (Speech-to-Text) Engines.
Provides unified interface and shared pipeline logic (VAD gating, window planning,
progress reporting, and post-generation guardrails) across all speech recognition models.
"""
import os
import re
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path
from abc import abstractmethod
from typing import Optional, Callable, Dict, List, Any

from ..base import BaseModelEngine
from ...config import settings
from ...services.audio_processor import get_audio_duration, get_ffmpeg_binary
from ...services.vad import is_speech_present
from .guardrails import clean_and_guard_asr_text

logger = logging.getLogger(__name__)


class BaseASREngine(BaseModelEngine):
    """
    Abstract Base Class for all ASR Engines (Audar, Faster-Whisper, etc.).
    Encapsulates shared audio slicing, VAD energy checks, and post-filtering.
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)

    @abstractmethod
    def load(self):
        """Load model weights into GPU or CPU memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unload model from memory and release VRAM/RAM."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        pass

    @abstractmethod
    def _transcribe_slice_internal(self, slice_wav_path: Path, dur_sec: float) -> str:
        """
        Model-specific transcription of a single short audio segment (<= 30s).
        Must be implemented by concrete subclasses (e.g. AudarASREngine, WhisperASREngine).
        """
        pass

    def transcribe_slice(self, slice_wav_path: Path, dur_sec: float) -> str:
        """
        Shared guarded slice transcription:
        1. VAD & Energy Gate check: Rejects pure silence/noise before waking neural model.
        2. Dispatches to model-specific inference (_transcribe_slice_internal).
        3. Applies post-ASR guardrails and text sanitization.
        """
        # 1. Acoustic Energy & VAD Gate
        has_speech, rms_db = is_speech_present(slice_wav_path)
        if not has_speech:
            logger.debug(f"VAD Gate: Skipping non-speech slice {slice_wav_path.name} (RMS: {rms_db:.1f} dBFS)")
            return ""

        # 2. Model-specific transcription
        raw_text = self._transcribe_slice_internal(slice_wav_path, dur_sec=dur_sec)

        # 3. Post-generation guardrails & deduplication
        if getattr(settings, "ASR_GUARDRAILS_ENABLED", True):
            return clean_and_guard_asr_text(raw_text, dur_sec=dur_sec)
        return raw_text.strip()

    @staticmethod
    def plan_windows(
        turns: List[Dict[str, Any]],
        max_len: float = 24.0,
        min_len: float = 0.25,
        max_same_speaker_gap: float = 2.5,
    ) -> List[Dict[str, Any]]:
        """
        Convert speaker diarization turns into optimal ASR windows:
        1. Pre-merge consecutive turns of the SAME speaker when no other speaker intervenes
           (if gap <= 2.5s and combined span <= max_len).
        2. Filter out micro-segments (< 0.35s) which are typically noise clicks or mic bumps.
        3. Split any long turns that exceed max_len into equal windows <= max_len.
        """
        if not turns:
            return []

        sorted_turns = sorted(turns, key=lambda x: float(x.get("start", 0)))
        merged_turns: List[Dict[str, Any]] = []

        for t in sorted_turns:
            start = float(t.get("start", 0))
            end = float(t.get("end", 0))
            dur = end - start
            if dur < min_len:
                continue

            item = {
                "speaker": str(t.get("speaker", "SPEAKER_00")),
                "start": start,
                "end": end,
            }

            if not merged_turns:
                merged_turns.append(item)
                continue

            prev = merged_turns[-1]
            gap = item["start"] - prev["end"]
            combined_span = item["end"] - prev["start"]

            if (
                prev["speaker"] == item["speaker"]
                and 0.0 <= gap <= max_same_speaker_gap
                and combined_span <= max_len
            ):
                prev["end"] = max(prev["end"], item["end"])
            else:
                merged_turns.append(item)

        windows = []
        for t in merged_turns:
            dur = t["end"] - t["start"]
            n = max(1, int(-(-dur // max_len)))  # ceil division
            step = dur / n
            for k in range(n):
                windows.append({
                    "speaker": t["speaker"],
                    "start": round(t["start"] + k * step, 2),
                    "end": round(t["start"] + (k + 1) * step, 2),
                })

        return windows

    def transcribe(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[int, str], None]] = None,
        max_duration_seconds: Optional[float] = None,
        progress_range: tuple = (25, 92),
    ) -> Dict[str, Any]:
        """
        Transcribe continuous audio file of any duration.
        Automatically chunks audio into 25-second windows with 2-second overlap,
        applies VAD to skip silence/noise, and deduplicates the overlap region.
        """
        if not self.is_loaded():
            self.load()

        total_duration = get_audio_duration(audio_path)
        if max_duration_seconds and total_duration > max_duration_seconds:
            logger.info(f"Limiting transcription to first {max_duration_seconds}s of {total_duration}s")
            total_duration = max_duration_seconds

        chunk_len = 25.0
        overlap = 2.0
        step = chunk_len - overlap
        ffmpeg_bin = get_ffmpeg_binary()

        chunks_count = max(1, int((total_duration + step - 1) // step))
        logger.info(f"Starting ASR on {audio_path.name} (Duration: {total_duration}s, {chunks_count} chunks)")

        slice_info = []
        for i in range(chunks_count):
            start_sec = i * step
            dur_sec = min(chunk_len, total_duration - start_sec)
            if dur_sec <= 0.3:
                break
            slice_info.append({
                "idx": i,
                "start": start_sec,
                "dur": dur_sec,
            })

        # Load audio into memory once for high-speed slicing without invoking 100+ ffmpeg processes
        audio_data = None
        sr = 16000
        try:
            from scipy.io import wavfile
            sr_read, raw_data = wavfile.read(str(audio_path))
            if sr_read == 16000:
                audio_data = raw_data
                sr = sr_read
        except Exception:
            audio_data = None

        # Check worker configuration scaled to GPU VRAM and CPU cores
        from ...services.hardware import get_hardware_profile
        hw_params = get_hardware_profile().asr_params()
        target_workers = max(1, min(4, hw_params.get("num_workers", 1)))

        extra_engines = []
        if getattr(self, "allow_secondary_workers", True) and target_workers > 1 and len(slice_info) > 2:
            for w_idx in range(1, target_workers):
                try:
                    cls = self.__class__
                    if hasattr(self, "mmproj_path") and self.mmproj_path:
                        w = cls(self.model_path, mmproj_path=self.mmproj_path)
                    else:
                        w = cls(self.model_path)
                    w.load()
                    extra_engines.append(w)
                    logger.info(f"Initialized parallel ASR worker #{len(extra_engines)+1} for {cls.__name__}")
                except Exception as e:
                    logger.warning(f"Could not initialize secondary ASR worker #{w_idx+1} ({e}); continuing with {len(extra_engines)+1} workers.")
                    break

        all_engines = [self] + extra_engines
        actual_workers = len(all_engines)
        logger.info(f"Executing continuous ASR with {actual_workers} concurrent worker(s) across {len(slice_info)} chunks.")

        all_text_parts = []
        all_words = []
        lo, hi = progress_range

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                slice_paths: List[Optional[Path]] = []

                for item in slice_info:
                    i = item["idx"]
                    start = item["start"]
                    dur = item["dur"]
                    slice_wav = tmp_path / f"slice_{i:04d}.wav"

                    slice_created = False
                    if audio_data is not None:
                        try:
                            from scipy.io import wavfile
                            s_idx = int(start * sr)
                            e_idx = min(len(audio_data), int((start + dur) * sr))
                            if e_idx > s_idx:
                                wavfile.write(str(slice_wav), sr, audio_data[s_idx:e_idx])
                                slice_created = slice_wav.exists()
                        except Exception:
                            slice_created = False

                    if not slice_created:
                        cmd = [
                            ffmpeg_bin, "-y", "-loglevel", "error",
                            "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                            "-i", str(audio_path),
                            "-ar", "16000", "-ac", "1",
                            "-c:a", "pcm_s16le",
                            str(slice_wav),
                        ]
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        slice_created = (proc.returncode == 0 and slice_wav.exists())

                    slice_paths.append(slice_wav if slice_created else None)

                # Release master audio array from RAM immediately before starting ASR inference
                del audio_data
                audio_data = None
                import gc
                gc.collect()

                results: List[str] = [""] * len(slice_info)
                completed_count = 0

                if actual_workers > 1:
                    import queue
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    engine_q: queue.Queue = queue.Queue()
                    for eng in all_engines:
                        engine_q.put(eng)

                    def _task(idx: int, p: Optional[Path], dur: float):
                        if p is None or not p.exists():
                            return idx, ""
                        worker_eng = engine_q.get()
                        try:
                            try:
                                return idx, worker_eng.transcribe_slice(p, dur_sec=dur)
                            except Exception as worker_err:
                                logger.warning(f"ASR worker slice {idx} failed ({worker_err}), retrying on primary engine...")
                                try:
                                    return idx, self.transcribe_slice(p, dur_sec=dur)
                                except Exception as fallback_err:
                                    logger.error(f"Fallback slice {idx} transcription also failed: {fallback_err}")
                                    return idx, ""
                        finally:
                            if p and p.exists():
                                try:
                                    p.unlink()
                                except OSError:
                                    pass
                            engine_q.put(worker_eng)

                    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                        futures = [
                            executor.submit(_task, item["idx"], slice_paths[idx_k], item["dur"])
                            for idx_k, item in enumerate(slice_info)
                        ]
                        for fut in as_completed(futures):
                            try:
                                idx, text = fut.result()
                                results[idx] = text
                            except Exception as fut_err:
                                logger.error(f"Error retrieving slice transcription from parallel future: {fut_err}")
                            completed_count += 1
                            if completed_count % 25 == 0:
                                gc.collect()
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            if on_progress:
                                pct = int(lo + (completed_count / max(1, len(slice_info))) * (hi - lo))
                                tot_sec = int(round(total_duration))
                                on_progress(
                                    pct,
                                    f"جاري تفريغ الصوت: المقطع {completed_count} من {len(slice_info)} [تسريع متوازي ×{actual_workers}]..."
                                )
                else:
                    for idx_k, item in enumerate(slice_info):
                        idx = item["idx"]
                        p = slice_paths[idx_k]
                        dur = item["dur"]
                        if p is not None and p.exists():
                            results[idx] = self.transcribe_slice(p, dur_sec=dur)
                            try:
                                p.unlink()
                            except OSError:
                                pass
                        completed_count += 1
                        if completed_count % 25 == 0:
                            gc.collect()
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                        if on_progress:
                            pct = int(lo + (completed_count / max(1, len(slice_info))) * (hi - lo))
                            tot_sec = int(round(total_duration))
                            on_progress(pct, f"تم تفريغ المقطع {completed_count} من {len(slice_info)} ({int(round(item['start']))}/{tot_sec} ث)...")

                # Clean up any remaining temporary slice audio files
                for sp in slice_paths:
                    if sp and sp.exists():
                        try:
                            sp.unlink()
                        except OSError:
                            pass

                # Sequential overlap deduplication and word alignment across chunk seams
                for idx_k, item in enumerate(slice_info):
                    idx = item["idx"]
                    start_sec = item["start"]
                    dur_sec = item["dur"]
                    slice_text = results[idx]
                    if not slice_text:
                        continue

                    words = slice_text.split()
                    if not words:
                        continue

                    word_dur = dur_sec / len(words)
                    chunk_words = []
                    for w_idx, word in enumerate(words):
                        w_start = round(start_sec + w_idx * word_dur, 2)
                        w_end = round(start_sec + (w_idx + 1) * word_dur, 2)
                        chunk_words.append({
                            "word": word,
                            "start": w_start,
                            "end": w_end,
                        })

                    # Deduplicate overlap region
                    if idx > 0 and all_words:
                        overlap_boundary = start_sec + overlap
                        new_start_idx = 0
                        for c_idx, cw in enumerate(chunk_words):
                            if cw["start"] >= overlap_boundary - 0.2:
                                new_start_idx = c_idx
                                break
                        chunk_words = chunk_words[new_start_idx:]

                        # Drop duplicate boundary words (e.g. repeated word across the chunk seam)
                        while chunk_words and all_words:
                            last_w = re.sub(r"[،,.\?!؟;:\"\']", "", all_words[-1]["word"]).strip()
                            first_w = re.sub(r"[،,.\?!؟;:\"\']", "", chunk_words[0]["word"]).strip()
                            if last_w and first_w and last_w == first_w and abs(chunk_words[0]["start"] - all_words[-1]["start"]) <= 1.5:
                                logger.info(f"Deduplicated boundary word: '{chunk_words[0]['word']}' at {chunk_words[0]['start']}s")
                                chunk_words = chunk_words[1:]
                            else:
                                break

                    if chunk_words:
                        text_part = " ".join(cw["word"] for cw in chunk_words)
                        all_text_parts.append(text_part)
                        all_words.extend(chunk_words)

        finally:
            for extra_eng in extra_engines:
                try:
                    extra_eng.unload()
                except Exception as ex:
                    logger.debug(f"Error unloading extra ASR engine: {ex}")

        full_text = " ".join(all_text_parts).strip()
        logger.info(f"ASR complete. Total words: {len(all_words)}, Total characters: {len(full_text)}")
        return {
            "text": full_text,
            "words": all_words,
        }

    def transcribe_turns(
        self,
        audio_path: Path,
        turns: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, str], None]] = None,
        progress_range: tuple = (25, 95),
    ) -> List[Dict[str, Any]]:
        """
        Diarization-driven transcription: Transcribe each speaker turn separately.
        Integrates VAD to skip silent windows and supports multi-engine worker parallelism.
        """
        if not self.is_loaded():
            self.load()

        windows = self.plan_windows(turns)
        lo, hi = progress_range
        segments: List[Dict[str, Any]] = []

        # Load audio into memory once for fast slicing
        audio_data = None
        sr = 16000
        try:
            from scipy.io import wavfile
            sr_read, raw_data = wavfile.read(str(audio_path))
            if sr_read == 16000:
                audio_data = raw_data
                sr = sr_read
        except Exception:
            audio_data = None

        ffmpeg_bin = get_ffmpeg_binary()

        # Check worker configuration
        from ...services.hardware import get_hardware_profile
        hw_params = get_hardware_profile().asr_params()
        target_workers = max(1, min(4, hw_params.get("num_workers", 1)))

        extra_engines = []
        if getattr(self, "allow_secondary_workers", True) and target_workers > 1 and len(windows) > 2:
            for w_idx in range(1, target_workers):
                try:
                    cls = self.__class__
                    w = cls(self.model_path)
                    w.load()
                    extra_engines.append(w)
                    logger.info(f"Initialized parallel ASR turn worker #{len(extra_engines)+1}")
                except Exception as e:
                    logger.warning(f"Could not initialize secondary ASR worker #{w_idx+1} ({e}); continuing with {len(extra_engines)+1} workers.")
                    break

        all_engines = [self] + extra_engines
        actual_workers = len(all_engines)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                slice_paths: List[Optional[Path]] = []
                slice_durations: List[float] = []

                for i, w in enumerate(windows):
                    slice_wav = tmp_path / f"w_{i:05d}.wav"
                    start = max(0.0, w["start"] - 0.20)
                    dur = (w["end"] - w["start"]) + 0.40
                    slice_durations.append(dur)

                    slice_created = False
                    if audio_data is not None:
                        try:
                            from scipy.io import wavfile
                            s_idx = int(start * sr)
                            e_idx = min(len(audio_data), int((start + dur) * sr))
                            if e_idx > s_idx:
                                wavfile.write(str(slice_wav), sr, audio_data[s_idx:e_idx])
                                slice_created = slice_wav.exists()
                        except Exception:
                            slice_created = False

                    if not slice_created:
                        cmd = [
                            ffmpeg_bin, "-y", "-loglevel", "error",
                            "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                            "-i", str(audio_path),
                            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                            str(slice_wav),
                        ]
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        slice_created = (proc.returncode == 0 and slice_wav.exists())

                    slice_paths.append(slice_wav if slice_created else None)

                results: List[str] = [""] * len(windows)
                completed_count = 0

                if actual_workers > 1:
                    import queue
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    engine_q: queue.Queue = queue.Queue()
                    for eng in all_engines:
                        engine_q.put(eng)

                    def _task(idx: int, p: Optional[Path], dur: float):
                        if p is None:
                            return idx, ""
                        worker_eng = engine_q.get()
                        try:
                            return idx, worker_eng.transcribe_slice(p, dur_sec=dur)
                        finally:
                            engine_q.put(worker_eng)

                    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                        futures = [
                            executor.submit(_task, i, slice_paths[i], slice_durations[i])
                            for i in range(len(windows))
                        ]
                        for fut in as_completed(futures):
                            idx, text = fut.result()
                            results[idx] = text
                            completed_count += 1
                            if on_progress:
                                pct = int(lo + (completed_count / max(1, len(windows))) * (hi - lo))
                                on_progress(pct, f"تفريغ المقطع {completed_count} من {len(windows)} (متوازي ×{actual_workers})...")
                else:
                    for i, sp in enumerate(slice_paths):
                        if sp is not None:
                            results[i] = self.transcribe_slice(sp, dur_sec=slice_durations[i])
                        completed_count += 1
                        if on_progress:
                            pct = int(lo + (completed_count / max(1, len(windows))) * (hi - lo))
                            on_progress(pct, f"تفريغ المقطع {completed_count} من {len(windows)}...")

                # Clean up temporary slice audio files
                for sp in slice_paths:
                    if sp and sp.exists():
                        try:
                            sp.unlink()
                        except OSError:
                            pass

                # Assemble and merge consecutive same-speaker windows
                for i, w in enumerate(windows):
                    text = results[i].strip()
                    if text:
                        if (
                            segments
                            and segments[-1]["speaker"] == w["speaker"]
                            and w["start"] - segments[-1]["end"] < 2.5
                        ):
                            segments[-1]["text"] += " " + text
                            segments[-1]["end"] = w["end"]
                        else:
                            segments.append({**w, "text": text})

        finally:
            for extra_eng in extra_engines:
                try:
                    extra_eng.unload()
                except Exception as ex:
                    logger.debug(f"Error unloading extra ASR engine: {ex}")

        logger.info(f"Speaker-aware ASR complete: {len(windows)} windows -> {len(segments)} segments.")
        return segments
