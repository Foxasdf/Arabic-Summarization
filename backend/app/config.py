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
    PORT: int = 8000
    MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "uploads"
    DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "meetings.db"
    MAX_FILE_SIZE_MB: int = 500
    
    ASR_MODEL_NAME: str = "Audar-ASR-V1-Turbo"
    ASR_GGUF_FILENAME: str = "Audar-ASR-V1-Turbo-Q8_0.gguf"
    ASR_MMPROJ_FILENAME: str = "mmproj-Audar-ASR-V1-Turbo.gguf"
    
    LLM_MODEL_NAME: str = "Falcon-H1-7B-Instruct"
    LLM_GGUF_FILENAME: str = "Falcon-H1-7B-Instruct-IQ4_XS.gguf"
    LLM_CONTEXT_SIZE: int = 8192
    LLM_GPU_LAYERS: int = -1
    # 0 = automatic (chosen from hardware profile)
    LLM_CONTEXT_SIZE_OVERRIDE: int = 0

    # Diarization: "auto" (pyannote if available, else builtin), "pyannote", "builtin"
    DIARIZATION_BACKEND: str = "auto"
    # 0 = auto-detect the number of speakers
    DIARIZATION_NUM_SPEAKERS: int = 0
    DIARIZATION_MAX_SPEAKERS: int = 8

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
