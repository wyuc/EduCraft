import json
from typing import Dict, List, Any, Iterator
import requests
from .base import ModelInterface, ModelResponse, Message, MessageRole, MessageContent, ContentType
from .factory import ModelFactory

class Ollama(ModelInterface):
    """Interface for Ollama models."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        super().__init__(base_url, api_key)
   
    def call(self, messages: List[Message], temperature: float = 1.0, 
             max_tokens: int = 4096, **kwargs) -> ModelResponse:
        """Make a synchronous call to Ollama."""
        # Format messages for Ollama API
        formatted_messages = self._format_messages(messages)

        print(formatted_messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['model']}
            }
        }
        
        # Make the API request
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Create standardized response
        return ModelResponse(
            content=result["message"]["content"],
            model_name=model_name,
            metadata={"done": result.get("done", True)}
        )
    
    def stream_call(self, messages: List[Message], temperature: float = 1.0, 
                    max_tokens: int = 4096, **kwargs) -> Iterator[ModelResponse]:
        """Make a streaming call to Ollama."""
        # Format messages for Ollama API
        formatted_messages = self._format_messages(messages)

        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['model']}
            }
        }
        
        # Make the streaming API request
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            # stream=True
        )
        response.raise_for_status()
        
        # Stream the response chunks
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    yield ModelResponse(
                        content=chunk["message"]["content"],
                        model_name=model_name,
                        metadata={"done": chunk.get("done", False)}
                    )
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages from unified format to Ollama API format."""
        ollama_messages = []
        
        for msg in messages:
            role = msg.role
            content = msg.content
            
            # Convert content to Ollama's format
            message_content = ""
            
            # Handle string content directly
            if isinstance(content, str):
                message_content = content
            else:
                # Process MessageContent with entries
                for entry in content.entries:
                    if entry.type == ContentType.TEXT:
                        # Add text content
                        message_content += entry.data.get("text", "")
                    elif entry.type == ContentType.IMAGE:
                        # Add image URL if it's a base64 image
                        image_url = entry.data.get("url", "")
                        if image_url.startswith("data:"):
                            message_content += f"\n[Image: {image_url}]"
            
            # Map standard roles to Ollama roles
            ollama_role = {
                MessageRole.SYSTEM: "system",
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "assistant"
            }.get(role, "user")
            
            ollama_messages.append({
                "role": ollama_role,
                "content": message_content
            })
        
        return ollama_messages
    
    @classmethod
    def supports_vision(cls) -> bool:
        """Check if the model supports vision capabilities."""
        return False  # Most Ollama models don't support vision by default
    
    @classmethod
    def get_context_window(cls) -> int:
        """Get the default context window size."""
        return 4096  # Default for most Ollama models
    
    @classmethod
    def get_model_type(cls) -> str:
        """Return the model type."""
        return "open-source"
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Return the model provider."""
        return "ollama"


# Register model with factory
ModelFactory.register("ollama", Ollama) 