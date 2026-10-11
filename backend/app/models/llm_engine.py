import logging
from pathlib import Path
from typing import Optional

from .base import BaseModelEngine

logger = logging.getLogger(__name__)


class LLMEngine(BaseModelEngine):
    """
    GGUF LLM wrapper (llama-cpp-python) for Arabic summarization.
    Context window, GPU layers, threads and batch size are chosen from the
    current machine's hardware profile at load time, so the same project
    folder runs on a weak laptop or a GPU workstation without edits.
    """

    def __init__(self, model_path: Path, max_context: int = 32768):
        self.model_path = model_path
        self.max_context = max_context
        self.model = None
        self.params: dict = {}

    # ------------------------------------------------------------------ life
    def load(self):
        if self.is_loaded():
            return
        if not self.model_path.exists():
            msg = (
                f"ملف نموذج التلخيص غير موجود في المسار: {self.model_path.name}\n"
                "يرجى تشغيل سكربت التحميل أولاً: python scripts/download_models.py"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        from llama_cpp import Llama
        from ..services.hardware import get_hardware_profile

        size_gb = self.model_path.stat().st_size / 1024**3
        self.params = get_hardware_profile().llm_params(size_gb, self.max_context)
        logger.info(f"Loading LLM {self.model_path.name} with {self.params}")

        # Retry with smaller context if memory allocation fails.
        n_ctx = self.params["n_ctx"]
        last_err = None
        while n_ctx >= 2048:
            try:
                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=n_ctx,
                    n_batch=self.params["n_batch"],
                    n_gpu_layers=self.params["n_gpu_layers"],
                    n_threads=self.params["n_threads"],
                    verbose=False,
                )
                self.params["n_ctx"] = n_ctx
                logger.info(f"LLM loaded (n_ctx={n_ctx}).")
                return
            except Exception as e:  # out-of-memory etc.
                last_err = e
                logger.warning(f"LLM load failed at n_ctx={n_ctx}: {e}; retrying smaller.")
                n_ctx //= 2
                if self.params["n_gpu_layers"] == -1:
                    self.params["n_gpu_layers"] = 0
        raise RuntimeError(f"تعذر تحميل نموذج التلخيص: {last_err}")

    def unload(self):
        if self.model is not None:
            try:
                self.model.close()
            except Exception:
                pass
            self.model = None
            logger.info("LLM unloaded.")

    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------- helpers
    @property
    def n_ctx(self) -> int:
        return self.params.get("n_ctx", 4096)

    @property
    def max_tokens(self) -> int:
        return self.params.get("max_tokens", 1024)

    def count_tokens(self, text: str) -> int:
        if self.model is None:
            return len(text) // 2
        return len(self.model.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def chat(self, messages: list, max_tokens: Optional[int] = None,
             temperature: float = 0.20, top_p: float = 0.88, top_k: int = 40,
             min_p: float = 0.05, repeat_penalty: float = 1.10) -> str:
        if not self.is_loaded():
            self.load()
        max_tokens = max_tokens or self.max_tokens
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "\nUser:", "\nالمستخدم:"]
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop_tokens,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Chat completion failed, trying raw completion: {e}")

        # Format ChatML prompt as fallback
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_tokens,
        )
        return response["choices"][0]["text"].strip()

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.20, top_p: float = 0.88, top_k: int = 40,
                 min_p: float = 0.05, repeat_penalty: float = 1.10) -> str:
        return self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], max_tokens=max_tokens, temperature=temperature, top_p=top_p,
           top_k=top_k, min_p=min_p, repeat_penalty=repeat_penalty)
