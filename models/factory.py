from typing import Dict, Type, Optional, List, Any
from .base import ModelInterface, ModelResponse, Message

class ModelFactory:
    """Factory for creating and managing model interfaces."""
    
    _registry: Dict[str, Type[ModelInterface]] = {}
    
    @classmethod
    def register(cls, model_name: str, model_class: Type[ModelInterface]) -> None:
        """Register a model implementation with a name."""
        cls._registry[model_name] = model_class
    
    @classmethod
    def get_model(cls, model_name: str) -> ModelInterface:
        """Get a model instance by name."""
        if model_name not in cls._registry:
            raise ValueError(f"Model '{model_name}' not registered. Available models: {list(cls._registry.keys())}")
        
        return cls._registry[model_name]()
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered models with their capabilities."""
        result = {}
        
        for name, model_class in cls._registry.items():
            result[name] = {
                "type": model_class.get_model_type(),
                "context_window": model_class.get_context_window(),
                "supports_vision": model_class.supports_vision()
            }
        
        return result
    
    @classmethod
    def list_by_type(cls, model_type: str) -> List[str]:
        """List all models of a specific type."""
        return [
            name for name, model_class in cls._registry.items()
            if model_class.get_model_type() == model_type
        ]