from typing import Dict, List, Optional, Any, Iterator, Union
import requests
from .base import (
    ModelInterface, ModelResponse, Message, MessageRole, MessageContent,
    ContentEntry, ContentType
)
from .factory import ModelFactory
from openai import OpenAI

class GPT(ModelInterface):
    """Base interface for OpenAI-compatible APIs (vLLM, OpenAI, etc.)."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the GPT interface.
        
        Args:
            base_url: The base URL for the API server
            api_key: The API key for authentication
        """
        super().__init__(base_url, api_key)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API format."""
        formatted_messages = []
        
        for msg in messages:
            role = msg.role.value
            
            # Handle string content directly
            if isinstance(msg.content, str):
                formatted_messages.append({
                    "role": role,
                    "content": msg.content
                })
                continue
                
            # Handle MessageContent with entries
            content_entries = msg.content.entries
            
            # If there's only one text entry, use simple content format
            if len(content_entries) == 1 and content_entries[0].type == ContentType.TEXT:
                formatted_messages.append({
                    "role": role,
                    "content": content_entries[0].data.get("text", "")
                })
            else:
                # Multiple entries - use multi-part format for vision API
                formatted_content = []
                
                for entry in content_entries:
                    if entry.type == ContentType.TEXT:
                        formatted_content.append({
                            "type": "text",
                            "text": entry.data.get("text", "")
                        })
                    elif entry.type == ContentType.IMAGE:
                        url = entry.data.get("url", "")
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                
                formatted_messages.append({
                    "role": role,
                    "content": formatted_content
                })
        
        return formatted_messages
    
    def call(self, messages: List[Message], temperature: float = 1.0, 
             max_tokens: int = 4096, **kwargs) -> ModelResponse:
        """Make a synchronous call to the model."""
        # Format messages for API
        formatted_messages = self._format_messages(messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        
        # Prepare the request parameters
        params = {
            "model": model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['model']}
        }
        
        # Make the API request
        response = self.client.chat.completions.create(**params)
        
        # Extract the response
        choice = response.choices[0]
        return ModelResponse(
            content=choice.message.content,
            model_name=model_name,
            metadata={"finish_reason": choice.finish_reason}
        )
    
    def stream_call(self, messages: List[Message], temperature: float = 1.0, 
                    max_tokens: int = 4096, **kwargs) -> Iterator[ModelResponse]:
        """Make a streaming call to the model."""
        # Format messages for API
        formatted_messages = self._format_messages(messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        
        # Prepare the request parameters
        params = {
            "model": model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **{k: v for k, v in kwargs.items() if k not in ['model']}
        }
        
        # Make the streaming API request
        response = self.client.chat.completions.create(**params)
        
        # Stream the response chunks
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ModelResponse(
                    content=chunk.choices[0].delta.content,
                    model_name=model_name,
                    metadata={"finish_reason": chunk.choices[0].finish_reason}
                )

    @classmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for configuration."""
        return "gpt"


ModelFactory.register("gpt", GPT)
    
class VLLM(GPT):
    @classmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for configuration."""
        return "vllm"

# Register the vLLM model with the factory
ModelFactory.register("vllm", VLLM) 

class Gemini_Openai(GPT):
    """interface for gemini api."""
    
    @classmethod
    def get_provider_name(cls) -> str:
        """return the provider name for configuration."""
        return "gemini_openai"


ModelFactory.register("gemini_openai", Gemini_Openai)

class Claude_Openai(GPT):
    """interface for claude api."""
    
    @classmethod
    def get_provider_name(cls) -> str:
        """return the provider name for configuration."""
        return "claude_openai"


ModelFactory.register("claude_openai", Claude_Openai)

class DeepSeek(GPT):
    """interface for deepseek api."""
    
    @classmethod
    def get_provider_name(cls) -> str:
        """return the provider name for configuration."""
        return "deepseek"


ModelFactory.register("deepseek", DeepSeek)