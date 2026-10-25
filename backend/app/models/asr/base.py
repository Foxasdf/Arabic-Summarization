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
import numpy as np

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
        max_len: float = 25.0,
        min_len: float = 0.25,
        max_same_speaker_gap: float = 1.2,
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

    @staticmethod
    def plan_vad_chunks(
        audio_data: Optional[Any],
        sr: int,
        total_duration: float,
        target_chunk_len: float = 25.0,
        min_chunk_len: float = 18.0,
        max_chunk_len: float = 30.0,
        overlap: float = 2.0,
        min_silence_sec: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Build optimal ASR chunks guided by silence / lowest acoustic energy:
        1. Targets 20-30s chunks to maintain maximum dialectal language model context.
        2. Searches around target boundary for the longest silence run (>= min_silence_sec)
           or absolute minimal RMS energy frame to avoid severing words mid-utterance.
        3. Preserves consistent overlap between adjacent chunks for seam deduplication.
        """
        if total_duration <= max_chunk_len:
            return [{"idx": 0, "start": 0.0, "dur": round(total_duration, 2)}]

        if audio_data is None:
            step = max(1.0, target_chunk_len - overlap)
            n_chunks = max(1, int((total_duration + step - 1) // step))
            slices = []
            for i in range(n_chunks):
                st = i * step
                d = min(target_chunk_len, total_duration - st)
                if d <= 0.3:
                    break
                slices.append({"idx": i, "start": round(st, 2), "dur": round(d, 2)})
            return slices

        data = audio_data
        if data.ndim > 1:
            data = data.mean(axis=1)

        frame_ms = 0.05
        frame_len = max(1, int(frame_ms * sr))
        n_frames = len(data) // frame_len
        if n_frames < 2:
            return [{"idx": 0, "start": 0.0, "dur": round(total_duration, 2)}]

        scale = float(np.iinfo(data.dtype).max) if np.issubdtype(data.dtype, np.integer) else 1.0
        rms_arr = np.zeros(n_frames, dtype=np.float32)
        batch_frames = 4096
        for b_start in range(0, n_frames, batch_frames):
            b_end = min(n_frames, b_start + batch_frames)
            block = data[b_start * frame_len : b_end * frame_len].reshape(b_end - b_start, frame_len)
            block_f = block.astype(np.float32) / scale
            rms_arr[b_start:b_end] = np.sqrt(np.mean(block_f ** 2, axis=1) + 1e-9)
        rms_db = 20.0 * np.log10(rms_arr + 1e-9)

        chunks: List[Dict[str, Any]] = []
        cur_start = 0.0
        idx = 0
        silence_thresh = getattr(settings, "VAD_MIN_RMS_DB", -42.0)

        while cur_start < total_duration:
            if cur_start + max_chunk_len >= total_duration:
                dur = round(total_duration - cur_start, 2)
                if dur > 0.3:
                    chunks.append({"idx": idx, "start": round(cur_start, 2), "dur": dur})
                break

            search_st = cur_start + min_chunk_len
            search_en = min(total_duration, cur_start + max_chunk_len)
            f_st = int(search_st / frame_ms)
            f_en = min(n_frames, int(search_en / frame_ms))

            if f_en <= f_st:
                best_cut = cur_start + target_chunk_len
            else:
                win_db = rms_db[f_st:f_en]
                is_silent = win_db < silence_thresh
                longest_silence = 0
                cur_silence = 0
                silence_center_f = -1

                for fi, sil in enumerate(is_silent):
                    if sil:
                        cur_silence += 1
                        if cur_silence > longest_silence:
                            longest_silence = cur_silence
                            silence_center_f = fi - cur_silence // 2
                    else:
                        cur_silence = 0

                if longest_silence * frame_ms >= min_silence_sec and silence_center_f >= 0:
                    best_cut = (f_st + silence_center_f) * frame_ms
                else:
                    min_f = int(np.argmin(win_db))
                    best_cut = (f_st + min_f) * frame_ms

            dur = round(best_cut - cur_start, 2)
            chunks.append({"idx": idx, "start": round(cur_start, 2), "dur": dur})
            idx += 1
            cur_start = max(0.0, round(best_cut - overlap, 2))

        return chunks

    @staticmethod
    def _find_word_overlap(tail_words: List[str], head_words: List[str]) -> int:
        """
        Find number of overlapping boundary words between tail of previous chunk
        and head of current chunk using Arabic-normalized string matching.
        """
        def _norm_w(w: str) -> str:
            w = re.sub(r"[،,.\?!؟;:\"\'\-\(\)\[\]]", "", w).strip()
            w = re.sub(r"[إأآٱ]", "ا", w)
            w = re.sub(r"ة", "ه", w)
            w = re.sub(r"ى", "ي", w)
            return w

        tail_norm = [_norm_w(w) for w in tail_words]
        head_norm = [_norm_w(w) for w in head_words]

        # 1. Exact normalized match from largest to smallest
        max_k = min(len(tail_norm), len(head_norm))
        for k in range(max_k, 0, -1):
            if tail_norm[-k:] == head_norm[:k]:
                return k

        # 2. Fuzzy match for k >= 3 words if one word had a minor variation
        for k in range(min(max_k, 12), 2, -1):
            t_sub = tail_norm[-k:]
            h_sub = head_norm[:k]
            matches = sum(1 for a, b in zip(t_sub, h_sub) if a == b or a.startswith(b) or b.startswith(a))
            if matches >= k - 1 and (matches / k) >= 0.75:
                return k
        return 0

    def transcribe(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[int, str], None]] = None,
        max_duration_seconds: Optional[float] = None,
        progress_range: tuple = (25, 92),
    ) -> Dict[str, Any]:
        """
        Transcribe continuous audio file of any duration.
        Automatically chunks audio into windows with overlap,
        applies VAD to skip silence/noise, and deduplicates the overlap region.
        """
        if not self.is_loaded():
            self.load()

        total_duration = get_audio_duration(audio_path)
        if max_duration_seconds and total_duration > max_duration_seconds:
            logger.info(f"Limiting transcription to first {max_duration_seconds}s of {total_duration}s")
            total_duration = max_duration_seconds

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

        chunk_len = float(getattr(self, "chunk_len", getattr(settings, "ASR_CHUNK_SECONDS", 25.0)))
        overlap = float(getattr(self, "overlap", getattr(settings, "ASR_OVERLAP_SECONDS", 2.5)))
        min_silence_ms = float(getattr(settings, "ASR_MIN_SILENCE_MS", 300))
        ffmpeg_bin = get_ffmpeg_binary()

        slice_info = self.plan_vad_chunks(
            audio_data=audio_data,
            sr=sr,
            total_duration=total_duration,
            target_chunk_len=chunk_len,
            min_chunk_len=max(8.0, round(chunk_len * 0.72, 1)),
            max_chunk_len=min(30.0, round(chunk_len + 5.0, 1)),
            overlap=overlap,
            min_silence_sec=min_silence_ms / 1000.0,
        )
        logger.info(f"Starting ASR on {audio_path.name} (Duration: {total_duration}s, {len(slice_info)} chunks)")
        lo, hi = progress_range
        if on_progress:
            on_progress(lo, f"جاري تجهيز محركات التفريغ الصوتي ({len(slice_info)} مقطع)...")

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
        if on_progress:
            on_progress(lo + 1, f"جاري تفريغ الصوت: بدء معالجة {len(slice_info)} مقطع...")

        all_text_parts = []
        all_words = []

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                import gc
                import threading
                cancel_flag = threading.Event()

                def _materialize_slice(item: Dict[str, Any]) -> Optional[Path]:
                    if cancel_flag.is_set():
                        return None
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

                    return slice_wav if slice_created else None

                results: List[str] = [""] * len(slice_info)
                completed_count = 0

                try:
                    if actual_workers > 1:
                        import queue
                        from concurrent.futures import ThreadPoolExecutor, as_completed

                        engine_q: queue.Queue = queue.Queue()
                        for eng in all_engines:
                            engine_q.put(eng)

                        def _task(item: Dict[str, Any]):
                            if cancel_flag.is_set():
                                return item["idx"], ""
                            idx = item["idx"]
                            dur = item["dur"]
                            p = _materialize_slice(item)
                            if p is None or not p.exists():
                                return idx, ""
                            worker_eng = engine_q.get()
                            try:
                                if cancel_flag.is_set():
                                    return idx, ""
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

                        executor = ThreadPoolExecutor(max_workers=actual_workers)
                        try:
                            futures = [
                                executor.submit(_task, item)
                                for item in slice_info
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
                        except InterruptedError:
                            cancel_flag.set()
                            for f in futures:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        finally:
                            if not cancel_flag.is_set():
                                executor.shutdown(wait=True)
                    else:
                        for item in slice_info:
                            idx = item["idx"]
                            dur = item["dur"]
                            p = _materialize_slice(item)
                            if p is not None and p.exists():
                                try:
                                    results[idx] = self.transcribe_slice(p, dur_sec=dur)
                                finally:
                                    try:
                                        p.unlink(missing_ok=True)
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
                finally:
                    del audio_data
                    audio_data = None
                    gc.collect()

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

                    # Deduplicate seam words with previous chunks using smart suffix-prefix match
                    words_to_add = words
                    if idx > 0 and all_words:
                        tail_words = [w["word"] for w in all_words[-15:]]
                        head_words = words[:15]
                        overlap_k = self._find_word_overlap(tail_words, head_words)
                        if overlap_k > 0:
                            logger.info(f"Deduplicated {overlap_k} boundary words between chunks: {' '.join(words[:overlap_k])}")
                            words_to_add = words[overlap_k:]

                    if not words_to_add:
                        continue

                    word_dur = dur_sec / max(1, len(words))
                    offset_idx = len(words) - len(words_to_add)
                    chunk_words = []
                    for w_idx, word in enumerate(words_to_add):
                        w_start = round(start_sec + (offset_idx + w_idx) * word_dur, 2)
                        w_end = round(start_sec + (offset_idx + w_idx + 1) * word_dur, 2)
                        chunk_words.append({
                            "word": word,
                            "start": w_start,
                            "end": w_end,
                        })

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

                pad_sec = float(getattr(settings, "TURN_PAD_MS", 100)) / 1000.0
                for i, w in enumerate(windows):
                    slice_wav = tmp_path / f"w_{i:05d}.wav"
                    start = max(0.0, float(w["start"]) - pad_sec)
                    dur = (float(w["end"]) - float(w["start"])) + (pad_sec * 2.0)
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
                            try:
                                sp.unlink()
                            except OSError:
                                pass
                        completed_count += 1
                        if completed_count % 25 == 0:
                            import gc
                            gc.collect()
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
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
                max_utterance_dur = 15.0
                for i, w in enumerate(windows):
                    text = results[i].strip()
                    if text:
                        if (
                            segments
                            and segments[-1]["speaker"] == w["speaker"]
                            and (w["start"] - segments[-1]["end"]) <= 1.2
                            and (w["end"] - segments[-1]["start"]) <= max_utterance_dur
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

        from ...services.alignment import merge_consecutive_speaker_segments
        segments = merge_consecutive_speaker_segments(segments)

        logger.info(f"Speaker-aware ASR complete: {len(windows)} windows -> {len(segments)} segments.")
        return segments
