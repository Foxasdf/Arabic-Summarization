"""
Audar-ASR Engine Implementation using llama-cpp-python Multimodal (MTMD).
Uses the official model prompt and incorporates dynamic max_tokens and explicit stop tokens.
"""
import os
import re
import logging
try:
    import ctypes
except Exception:
    ctypes = None
from pathlib import Path
from typing import Optional

from .base import BaseASREngine
from ...config import settings

logger = logging.getLogger(__name__)


class AudarASREngine(BaseASREngine):
    """
    Audar-ASR-V1-Turbo Implementation via llama.cpp MTMD.
    Official Prompt: "فرّغ الكلام العربي التالي."
    """

    # Audar ASR runs on CUDA GPU via llama-cpp-python MTMD and scales across available VRAM.
    # Chunking follows AUDAR_* settings (20s native window, decoupled from Cohere 28s).
    allow_secondary_workers: bool = True

    def __init__(self, model_path: Path, mmproj_path: Optional[Path] = None):
        super().__init__(model_path)
        self.mmproj_path = Path(mmproj_path) if mmproj_path else self._find_mmproj_path()
        self.model = None
        self.handler = None
        # Per-instance chunking from settings (base.transcribe reads these attrs first)
        self.chunk_len = float(getattr(settings, "AUDAR_CHUNK_SECONDS", 20.0))
        self.overlap = float(getattr(settings, "AUDAR_OVERLAP_SECONDS", 2.5))

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
            from ...services.hardware import get_hardware_profile

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

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Audar ASR engine completely unloaded from memory.")

    def is_loaded(self) -> bool:
        return self.model is not None and self.handler is not None and getattr(self.handler, "mtmd_ctx", None) is not None

    def _transcribe_slice_internal(self, slice_wav_path: Path, dur_sec: float) -> str:
        """
        Transcribe a single audio chunk (<= 30 seconds) via Audar MTMD.
        Adheres strictly to the official prompt: "فرّغ الكلام العربي التالي."
        Uses dynamic token budget and explicit stop tokens to stop hallucination.
        """
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
            
            # Official default system prompt for Audar-ASR
            sys_prompt = getattr(
                settings,
                "AUDAR_SYSTEM_PROMPT",
                "فرّغ الكلام العربي التالي.",
            )
            prompt_text = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
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

            # Dynamic max_tokens based on slice length
            # Arabic speech rate is ~2.5 - 3.5 words/sec, each Arabic word ~1.5 - 2.5 tokens
            # Safe token budget: dur_sec * 16.0 tokens (max 440, min 64)
            dyn_tokens = int(max(64, min(440, (dur_sec or 10.0) * 16.0)))
            repeat_penalty = float(getattr(settings, "AUDAR_REPEAT_PENALTY", 1.0))
            frequency_penalty = float(getattr(settings, "AUDAR_FREQUENCY_PENALTY", 0.0))

            # Explicit stop tokens to stop generation cleanly
            out = self.model.create_completion(
                prompt=self.model.input_ids[: self.model.n_tokens].tolist(),
                max_tokens=dyn_tokens,
                temperature=0.0,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=0.0,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\n<|im_start|>", "\n\n\n"],
            )
            raw_text = out["choices"][0]["text"].strip()

            # Handle Audar no-speech protocol
            if "language None" in raw_text:
                return ""

            # Strip language tags and special format tokens
            raw_text = re.sub(r"^language\s+[A-Za-z]+(<asr_text>)?\s*", "", raw_text).strip()
            raw_text = raw_text.replace("<asr_text>", "").replace("<|im_end|>", "").strip()
            return raw_text

        finally:
            self.handler._mtmd_cpp.mtmd_bitmap_free(bitmap)

    def _clean_asr_text(self, text: str) -> str:
        """Backward-compatible helper mapping to the guardrail sanitizer."""
        from .guardrails import clean_and_guard_asr_text
        return clean_and_guard_asr_text(text)

    def _transcribe_slice(self, slice_wav_path: Path, dur_sec: Optional[float] = None) -> str:
        """Backward-compatible slice transcription."""
        return self.transcribe_slice(slice_wav_path, dur_sec=dur_sec or 10.0)

