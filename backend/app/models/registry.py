from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    key: str
    name: str
    model_type: str  # "llm" or "asr"
    gguf_filename: str
    description: str
    context_size: int = 8192

class ModelRegistry:
    """
    Registry for managing AI models. Makes it trivially easy to add new LLMs.
    
    To add a new LLM model:
        1. Place the GGUF file in the models/ directory
        2. Call registry.register(ModelInfo(...))
        3. The model will appear automatically in the UI model selector
    """
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._register_defaults()

    def _register_defaults(self):
        # Primary LLM Model requested by user
        self.register(ModelInfo(
            key="falcon-h1-7b-instruct",
            name="Falcon-H1 7B Instruct (IQ4_XS)",
            model_type="llm",
            gguf_filename="Falcon-H1-7B-Instruct-IQ4_XS.gguf",
            description="النموذج الأساسي لتلخيص الاجتماعات باللغة العربية (Unsloth GGUF)",
            context_size=32768,
        ))
        # Primary ASR Model requested by user
        self.register(ModelInfo(
            key="audar-asr-v1-turbo",
            name="Audar ASR V1 Turbo (Q8_0)",
            model_type="asr",
            gguf_filename="Audar-ASR-V1-Turbo-Q8_0.gguf",
            description="النموذج الأساسي لتحويل الصوت إلى نص عربي (الأعلى دقة وتعرف على اللهجات)",
            context_size=4096,
        ))
        # Keep legacy aliases for seamless compatibility
        self.register(ModelInfo(
            key="falcon-h1-arabic-7b",
            name="Falcon-H1 Arabic 7B",
            model_type="llm",
            gguf_filename="Falcon-H1-7B-Instruct-IQ4_XS.gguf",
            description="نموذج التلخيص العربي الأساسي",
            context_size=32768,
        ))
        self.register(ModelInfo(
            key="audar-asr-v1-flash",
            name="Audar ASR V1",
            model_type="asr",
            gguf_filename="Audar-ASR-V1-Turbo-Q8_0.gguf",
            description="نموذج تحويل الصوت إلى نص",
            context_size=4096,
        ))

    def register(self, model_info: ModelInfo):
        """Register a new model. Overwrites if key already exists."""
        self.models[model_info.key] = model_info
        logger.info(f"Registered model: {model_info.name} [{model_info.model_type}]")

    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """Get all models of a specific type (e.g., 'llm' or 'asr')."""
        return [m for m in self.models.values() if m.model_type == model_type]

    def get_model(self, key: str) -> Optional[ModelInfo]:
        """Get a model by its registry key."""
        return self.models.get(key)

    def list_llm_models(self) -> List[ModelInfo]:
        """Convenience method to list all LLM models."""
        return self.get_models_by_type("llm")

# Singleton instance
registry = ModelRegistry()

