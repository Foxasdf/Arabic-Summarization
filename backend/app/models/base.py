from abc import ABC, abstractmethod

class BaseModelEngine(ABC):
    @abstractmethod
    def load(self): ...
    
    @abstractmethod  
    def unload(self): ...
    
    @abstractmethod
    def is_loaded(self) -> bool: ...
