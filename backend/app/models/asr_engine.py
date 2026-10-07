import os
import re
import logging
import subprocess
import shutil
import tempfile
import ctypes
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any

from .base import BaseModelEngine
from ..config import settings
from ..services.audio_processor import get_audio_duration, get_ffmpeg_binary

HAS_CUDA = False  # GPU handling is done by llama.cpp itself (see services/hardware.py)

logger = logging.getLogger(__name__)


class ASREngine(BaseModelEngine):
    """
    Production ASR Engine using Audar-ASR-V1-Turbo GGUF model via llama.cpp MTMD.
    
    Audar-ASR-V1-Turbo is an Arabic-first generative speech-recognition model that recasts
    transcription as audio-conditioned next-token prediction over a unified text vocabulary.
    It supports MSA, Gulf/Emirati, Egyptian, Levantine, Maghrebi speech, code-switching, and English.
    """

    def __init__(self, model_path: Path, mmproj_path: Optional[Path] = None):
        self.model_path = model_path
        self.mmproj_path = mmproj_path or self._find_mmproj_path()
        self.model = None
        self.handler = None

    def _find_mmproj_path(self) -> Path:
        """Find the corresponding multimodal projector (mmproj) for Audar ASR."""
        candidates = [
            settings.MODELS_DIR / settings.ASR_MMPROJ_FILENAME,
            self.model_path.parent / f"mmproj-{self.model_path.name}",
            self.model_path.parent / "mmproj-Audar-ASR-V1-Turbo.gguf",
            self.model_path.parent / "mmproj-Audar-ASR-V1-Flash.gguf",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Default fallback
        return settings.MODELS_DIR / "mmproj-Audar-ASR-V1-Turbo.gguf"

    def load(self):
        if self.is_loaded():
            return

        if not self.model_path.exists():
            msg = f"ملف نموذج الصوت غير موجود: {self.model_path.name}. يرجى تشغيل سكربت التحميل أولاً."
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not self.mmproj_path.exists():
            msg = f"ملف بروجكتور الصوت (mmproj) غير موجود: {self.mmproj_path.name}. يرجى تشغيل سكربت التحميل أولاً."
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading Audar ASR model from {self.model_path.name} with mmproj {self.mmproj_path.name}")
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler
            from ..services.hardware import get_hardware_profile

            p = get_hardware_profile().asr_params()
            logger.info(f"ASR runtime params: {p}")
            n_threads = p["n_threads"]
            self.model = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=p["n_gpu_layers"],
                n_ctx=p["n_ctx"],
                n_batch=p["n_batch"],
                n_threads=n_threads,
                verbose=False,
            )

            self.handler = MTMDChatHandler(
                str(self.mmproj_path),
                verbose=False,
                use_gpu=p["use_gpu"],
            )

            ctx_params = self.handler._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = p["use_gpu"]
            ctx_params.print_timings = False
            ctx_params.n_threads = n_threads

            self.handler.mtmd_ctx = self.handler._mtmd_cpp.mtmd_init_from_file(
                str(self.mmproj_path).encode("utf-8"),
                self.model.model,
                ctx_params,
            )

            if not self.handler.mtmd_ctx:
                raise RuntimeError("Failed to initialize MTMD context for Audar ASR")

            logger.info("Audar ASR model & MTMD context successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load Audar ASR model: {e}")
            self.unload()
            raise

    def unload(self):
        """Unload both the MTMD context and Llama model, clearing CUDA VRAM."""
        if self.handler and getattr(self.handler, "mtmd_ctx", None):
            try:
                self.handler._mtmd_cpp.mtmd_free(self.handler.mtmd_ctx)
            except Exception as e:
                logger.warning(f"Error freeing mtmd context: {e}")
            self.handler.mtmd_ctx = None

        if self.handler:
            del self.handler
            self.handler = None

        if self.model:
            del self.model
            self.model = None

        if HAS_CUDA:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("Audar ASR engine completely unloaded from memory.")

    def is_loaded(self) -> bool:
        return self.model is not None and self.handler is not None and self.handler.mtmd_ctx is not None

    def _transcribe_slice(self, slice_wav_path: Path) -> str:
        """Transcribe a single audio chunk (<= 30 seconds) via MTMD."""
        import llama_cpp

        bitmap = self.handler._mtmd_cpp.mtmd_helper_bitmap_init_from_file(
            self.handler.mtmd_ctx,
            str(slice_wav_path).encode("utf-8"),
            False,
        )
        if not bitmap:
            logger.warning(f"Could not initialize audio bitmap for slice {slice_wav_path}")
            return ""

        try:
            media_marker = self.handler._mtmd_cpp.mtmd_default_marker().decode("utf-8")
            prompt_text = (
                "<|im_start|>system\nفرّغ الكلام العربي التالي.<|im_end|>\n"
                f"<|im_start|>user\n{media_marker}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            input_text = self.handler._mtmd_cpp.mtmd_input_text()
            input_text_bytes = prompt_text.encode("utf-8")
            input_text.text = input_text_bytes
            input_text.text_len = len(input_text_bytes)
            input_text.add_special = True
            input_text.parse_special = True

            chunks = self.handler._mtmd_cpp.mtmd_input_chunks_init()
            bitmaps = [bitmap]
            bitmap_array = (self.handler._mtmd_cpp.mtmd_bitmap_p_ctypes * 1)(*bitmaps)

            res = self.handler._mtmd_cpp.mtmd_tokenize(
                self.handler.mtmd_ctx,
                chunks,
                ctypes.byref(input_text),
                bitmap_array,
                1,
            )
            if res != 0:
                logger.warning(f"mtmd_tokenize failed with code {res}")
                return ""

            self.model.reset()
            self.model._ctx.kv_cache_clear()

            n_chunks = self.handler._mtmd_cpp.mtmd_input_chunks_size(chunks)
            for i in range(n_chunks):
                chunk = self.handler._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                if chunk is None:
                    continue

                chunk_type = self.handler._mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                if chunk_type == self.handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n_tokens_out = ctypes.c_size_t()
                    tokens_ptr = self.handler._mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                        chunk, ctypes.byref(n_tokens_out)
                    )
                    if tokens_ptr and n_tokens_out.value > 0:
                        self.model.eval([tokens_ptr[j] for j in range(n_tokens_out.value)])
                elif chunk_type in [
                    self.handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    self.handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                ]:
                    new_n_past = llama_cpp.llama_pos(0)
                    self.handler._mtmd_cpp.mtmd_helper_eval_chunk_single(
                        self.handler.mtmd_ctx,
                        self.model._ctx.ctx,
                        chunk,
                        llama_cpp.llama_pos(self.model.n_tokens),
                        llama_cpp.llama_seq_id(0),
                        self.model.n_batch,
                        False,
                        ctypes.byref(new_n_past),
                    )
                    self.model.n_tokens = new_n_past.value

            self.handler._mtmd_cpp.mtmd_input_chunks_free(chunks)

            # Generate transcribed tokens with anti-repetition decoding
            out = self.model.create_completion(
                prompt=self.model.input_ids[: self.model.n_tokens].tolist(),
                max_tokens=256,
                temperature=0.0,
                repeat_penalty=1.2,
                frequency_penalty=0.3,
                presence_penalty=0.1,
            )
            raw_text = out["choices"][0]["text"].strip()
            return self._clean_asr_text(raw_text)

        finally:
            self.handler._mtmd_cpp.mtmd_bitmap_free(bitmap)

    def _clean_asr_text(self, text: str) -> str:
        """Strip internal ASR prompt tags, language indicators, bracket noise, and repetition loops."""
        if not text:
            return ""
        # Remove special tokens
        for tag in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<asr_text>", "language Arabic", "language None"]:
            text = text.replace(tag, "")

        # Normalize bracketed words / artifacts
        replacements = {
            r"\[podcasts?\]": "بودكاست",
            r"\[benchmarks?\]": "بنشمارك",
            r"جهد تهني": "جهد ذهني",
        }
        for pat, rep in replacements.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

        # Remove audio annotations like [موسيقى], [ضحك], [تصفيق], [music], etc.
        text = re.sub(r"\[(موسيقى|ضحك|تصفيق|صوت|صمت|music|applause|laughter|noise|silence)\]", "", text, flags=re.IGNORECASE)

        # Clean lingering brackets around english words: e.g. [word] -> word
        text = re.sub(r"\[([a-zA-Z0-9_\-\s]+)\]", r"\1", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Phrase-level deduplication:
        # Detect consecutive repetitions of n-grams (from 1 up to 8 words)
        words = text.split()
        if not words:
            return ""

        cleaned_words = []
        i = 0
        n = len(words)

        def norm_w(w):
            return re.sub(r"[،,.\?!؟;:\"\']", "", w)

        while i < n:
            best_plen = 0
            best_reps = 0
            max_collapsed_tokens = 0

            # Try phrase length from 1 up to 8 words
            for plen in range(1, min(9, (n - i) // 2 + 1)):
                phrase_norm = [norm_w(w) for w in words[i:i + plen]]
                if not any(phrase_norm):
                    continue
                reps = 1
                while i + (reps + 1) * plen <= n:
                    next_phrase_norm = [norm_w(w) for w in words[i + reps * plen : i + (reps + 1) * plen]]
                    if next_phrase_norm == phrase_norm:
                        reps += 1
                    else:
                        break

                collapsed_tokens = (reps - 1) * plen
                if reps >= 2 and collapsed_tokens > max_collapsed_tokens:
                    max_collapsed_tokens = collapsed_tokens
                    best_plen = plen
                    best_reps = reps

            if best_reps >= 2:
                # If a single word is repeated twice (e.g. "لا لا" or "عالي عالي"), allow natural emphasis
                if best_plen == 1 and best_reps == 2 and words[i] in ["لا", "نعم", "عالي", "جدا"]:
                    cleaned_words.append(words[i])
                    cleaned_words.append(words[i + 1])
                    i += 2
                else:
                    # Keep 1 instance of the repeated phrase
                    cleaned_words.extend(words[i:i + best_plen])
                    i += best_reps * best_plen
            else:
                cleaned_words.append(words[i])
                i += 1

        cleaned_text = " ".join(cleaned_words)
        cleaned_text = re.sub(r"([،,.\?!؟])\s*\1+", r"\1", cleaned_text)
        return cleaned_text.strip()

    def transcribe(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[int, str], None]] = None,
        max_duration_seconds: Optional[float] = None,
        progress_range: tuple = (25, 92),
    ) -> Dict[str, Any]:
        """
        Transcribe audio file of any duration.
        Automatically chunks audio into 25-second windows and aggregates word timestamps.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            on_progress: Callback taking (progress_percent: int, message: str)
            max_duration_seconds: Optional limit for testing/speed
            progress_range: Tuple of (start_percent, end_percent) for progress mapping
            
        Returns:
            dict with:
                - "text": complete aggregated text
                - "words": list of {"word": str, "start": float, "end": float}
        """
        if not self.is_loaded():
            self.load()

        total_duration = get_audio_duration(audio_path)
        if max_duration_seconds and total_duration > max_duration_seconds:
            logger.info(f"Limiting transcription to first {max_duration_seconds}s of {total_duration}s")
            total_duration = max_duration_seconds

        chunk_len = 25.0  # seconds per ASR window (safe under Audar's 30s max)
        ffmpeg_bin = get_ffmpeg_binary()

        chunks_count = max(1, int((total_duration + chunk_len - 1) // chunk_len))
        logger.info(f"Starting ASR on {audio_path.name} (Duration: {total_duration}s, {chunks_count} chunks)")

        all_text_parts = []
        all_words = []
        lo, hi = progress_range

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for i in range(chunks_count):
                start_sec = i * chunk_len
                dur_sec = min(chunk_len, total_duration - start_sec)
                if dur_sec <= 0.5:
                    break

                if on_progress:
                    start_pct = int(lo + (i / max(1, chunks_count)) * (hi - lo))
                    cur_sec = int(round(start_sec))
                    tot_sec = int(round(total_duration))
                    on_progress(start_pct, f"جاري تفريغ المقطع {i+1} من {chunks_count} ({cur_sec}/{tot_sec} ثانية)...")

                slice_wav = tmp_path / f"slice_{i:04d}.wav"

                # Extract 16kHz mono WAV slice
                cmd = [
                    ffmpeg_bin, "-y",
                    "-ss", str(round(start_sec, 2)),
                    "-t", str(round(dur_sec, 2)),
                    "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(slice_wav),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0 or not slice_wav.exists():
                    logger.warning(f"Failed to slice audio at {start_sec}s: {proc.stderr.decode(errors='ignore')}")
                    continue

                # Transcribe slice
                slice_text = self._transcribe_slice(slice_wav)
                if slice_text:
                    all_text_parts.append(slice_text)
                    words = slice_text.split()
                    if words:
                        word_dur = dur_sec / len(words)
                        for w_idx, word in enumerate(words):
                            all_words.append({
                                "word": word,
                                "start": round(start_sec + w_idx * word_dur, 2),
                                "end": round(start_sec + (w_idx + 1) * word_dur, 2),
                            })

                if on_progress:
                    pct = int(lo + ((i + 1) / max(1, chunks_count)) * (hi - lo))
                    processed_sec = min(int(round(total_duration)), int(round(start_sec + dur_sec)))
                    tot_sec = int(round(total_duration))
                    msg = f"تم تفريغ المقطع {i+1} من {chunks_count} ({processed_sec}/{tot_sec} ثانية)..."
                    on_progress(pct, msg)

        full_text = " ".join(all_text_parts).strip()
        logger.info(f"ASR complete. Total words: {len(all_words)}, Total characters: {len(full_text)}")
        return {
            "text": full_text,
            "words": all_words,
        }

    # ------------------------------------------------------------------
    # Speaker-aware transcription
    # ------------------------------------------------------------------
    @staticmethod
    def plan_windows(turns: List[Dict[str, Any]], max_len: float = 25.0, min_len: float = 0.4) -> List[Dict[str, Any]]:
        """
        Convert diarization turns into ASR windows that never cross a speaker
        change and never exceed Audar's 30 s limit. Long turns are split into
        equal parts; tiny turns are dropped (usually breaths / noise).
        """
        windows = []
        for t in turns:
            dur = t["end"] - t["start"]
            if dur < min_len:
                continue
            n = max(1, int(-(-dur // max_len)))  # ceil
            step = dur / n
            for k in range(n):
                windows.append({
                    "speaker": t["speaker"],
                    "start": round(t["start"] + k * step, 2),
                    "end": round(t["start"] + (k + 1) * step, 2),
                })
        return windows

    def transcribe_turns(
        self,
        audio_path: Path,
        turns: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, str], None]] = None,
        progress_range: tuple = (35, 95),
    ) -> List[Dict[str, Any]]:
        """
        Transcribe each speaker turn separately. Returns segments:
            [{"speaker": "SPEAKER_00", "text": "...", "start": 0.0, "end": 4.2}, ...]
        Consecutive windows from the same speaker are merged afterwards.
        """
        if not self.is_loaded():
            self.load()

        windows = self.plan_windows(turns)
        ffmpeg_bin = get_ffmpeg_binary()
        lo, hi = progress_range
        segments: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i, w in enumerate(windows):
                slice_wav = tmp_path / f"w_{i:05d}.wav"
                # small padding so word edges aren't clipped
                start = max(0.0, w["start"] - 0.15)
                dur = (w["end"] - w["start"]) + 0.3
                cmd = [
                    ffmpeg_bin, "-y", "-loglevel", "error",
                    "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                    "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    str(slice_wav),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode == 0 and slice_wav.exists():
                    text = self._transcribe_slice(slice_wav)
                    if text:
                        if segments and segments[-1]["speaker"] == w["speaker"] \
                                and w["start"] - segments[-1]["end"] < 1.5:
                            segments[-1]["text"] += " " + text
                            segments[-1]["end"] = w["end"]
                        else:
                            segments.append({**w, "text": text})
                    try:
                        slice_wav.unlink()
                    except OSError:
                        pass

                if on_progress:
                    pct = int(lo + (i + 1) / max(1, len(windows)) * (hi - lo))
                    on_progress(pct, f"تفريغ المقطع {i + 1} من {len(windows)}...")

        logger.info(f"Speaker-aware ASR: {len(windows)} windows -> {len(segments)} segments.")
        return segments
