import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore
    except ImportError:
        class BaseSettings:  # type: ignore
            """Fallback if pydantic_settings is not installed."""
            def __init__(self, **kwargs):
                for k, v in self.__class__.__dict__.items():
                    if not k.startswith("_") and not callable(v):
                        env_val = os.getenv(f"MEETING_{k}", None)
                        if env_val is not None:
                            try:
                                if isinstance(v, Path):
                                    setattr(self, k, Path(env_val))
                                elif isinstance(v, int):
                                    setattr(self, k, int(env_val))
                                elif isinstance(v, float):
                                    setattr(self, k, float(env_val))
                                else:
                                    setattr(self, k, env_val)
                            except Exception:
                                setattr(self, k, env_val)
                        else:
                            setattr(self, k, v)
                for k, v in kwargs.items():
                    setattr(self, k, v)


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 3011
    MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "uploads"
    DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "meetings.db"
    MAX_FILE_SIZE_MB: int = 500
    
    ASR_MODEL_NAME: str = "cohere-transcribe-arabic"
    ASR_GGUF_FILENAME: str = "Audar-ASR-V1-Turbo-Q8_0.gguf"
    ASR_MMPROJ_FILENAME: str = "mmproj-Audar-ASR-V1-Turbo.gguf"
    COHERE_ASR_FILENAME: str = "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"
    WHISPER_DIALECTAL_DIRNAME: str = "whisper-large-v3-turbo-arabic-dialectal-v2-ct2"
    ASR_NUM_WORKERS: int = 0  # 0 = auto-detect (2 workers if VRAM >= 10GB, else 1), or explicitly 1 or 2

    # Audar ASR Dialect & Generation Parameters (Tuned for Levantine / Syrian speech)
    AUDAR_SYSTEM_PROMPT: str = (
        "فرّغ التسجيل الصوتي بدقة تامة باللهجة الشامية/السورية."
        "مع الالتزام التام بالمفردات المنطوقة."
    )
    AUDAR_REPEAT_PENALTY: float = 1.20
    AUDAR_FREQUENCY_PENALTY: float = 0.10
    
    LLM_MODEL_NAME: str = "Falcon-H1-7B-Instruct"
    LLM_GGUF_FILENAME: str = "Falcon-H1-7B-Instruct-IQ4_XS.gguf"
    LLM_CONTEXT_SIZE: int = 8192
    LLM_GPU_LAYERS: int = -1
    # 0 = automatic (chosen from hardware profile)
    LLM_CONTEXT_SIZE_OVERRIDE: int = 0

    # Summarization & LLM Generation Defaults (optimal tuned settings)
    SUMMARY_DEFAULT_TEMPERATURE: float = 0.20
    SUMMARY_DEFAULT_TOP_P: float = 0.88
    SUMMARY_DEFAULT_TOP_K: int = 40
    SUMMARY_DEFAULT_MIN_P: float = 0.05
    SUMMARY_DEFAULT_REPEAT_PENALTY: float = 1.10

    # Diarization: Pyannote 3.1 is the exclusive engine (no fallback allowed)
    DIARIZATION_BACKEND: str = "pyannote"
    # 0 = auto-detect the number of speakers (with min/max bounds)
    DIARIZATION_NUM_SPEAKERS: int = 0
    DIARIZATION_MIN_SPEAKERS: int = 2
    DIARIZATION_MAX_SPEAKERS: int = 8

    # Timeouts (in seconds)
    AUDIO_CONVERT_TIMEOUT: int = 7200       # 2 hours for ffmpeg conversion of large audio
    KEEP_ALIVE_TIMEOUT: int = 3600          # 1 hour HTTP keep-alive timeout

    # Audio Enhancement & Noise Reduction ("none" default/recommended, "low", "moderate", "high", "dynamic")
    AUDIO_DENOISE_MODE: str = "none"

    # Voice Activity Detection (VAD) & Energy Gate
    VAD_ENABLED: bool = True
    VAD_MIN_RMS_DB: float = -42.0

    # ASR Guardrails & Repetition Penalties
    ASR_REPEAT_PENALTY: float = 1.15
    ASR_GUARDRAILS_ENABLED: bool = True

    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or None

    class Config:
        env_prefix = "MEETING_"


settings = Settings()

try:
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
