from typing import Dict, List, Any, Iterator
import google.generativeai as genai
from .base import (
    ModelInterface, ModelResponse, Message, MessageRole, MessageContent,
    ContentEntry, ContentType
)
from .factory import ModelFactory

class Gemini(ModelInterface):
    """Interface for Google's Gemini models."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the Gemini interface with API key from configuration."""
        super().__init__(base_url, api_key)
        # Configure the Gemini API
        genai.configure(
            api_key=self.api_key,
            transport="rest",
            client_options={"api_endpoint": self.base_url}
        )
   
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Gemini API format."""
        formatted_messages = []
        
        for msg in messages:
            # Map standard roles to Gemini roles - Gemini only accepts "user" and "model" roles
            role = {
                MessageRole.SYSTEM: "user",  # Map system to user since Gemini doesn't support system role
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "model"
            }.get(msg.role, "user")
            
            # Handle string content directly
            if isinstance(msg.content, str):
                formatted_messages.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
                continue
                
            # Handle MessageContent with entries
            if not hasattr(msg.content, 'entries'):
                # Handle legacy content format or unexpected content
                formatted_messages.append({
                    "role": role,
                    "parts": [{"text": str(msg.content)}]
                })
                continue
            
            content_entries = msg.content.entries
            
            # If there's only one text entry, use simple content format
            if len(content_entries) == 1 and content_entries[0].type == ContentType.TEXT:
                formatted_messages.append({
                    "role": role,
                    "parts": [{"text": content_entries[0].data.get("text", "")}]
                })
            else:
                # Multiple entries - use parts format for Gemini
                parts = []
                
                for entry in content_entries:
                    if entry.type == ContentType.TEXT:
                        parts.append({
                            "text": entry.data.get("text", "")
                        })
                    elif entry.type == ContentType.IMAGE:
                        url = entry.data.get("url", "")
                        if url.startswith("data:"):
                            # Handle base64 images
                            mime_type = url.split(';')[0].split(':')[1]
                            data = url.split(',')[1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": data
                                }
                            })
                        else:
                            # Handle normal URLs
                            parts.append({
                                "file_data": {
                                    "mime_type": "image/jpeg",
                                    "file_uri": url
                                }
                            })
                
                formatted_messages.append({
                    "role": role,
                    "parts": parts
                })
        
        return formatted_messages
    
    def call(self, messages: List[Message], temperature: float = 1.0, 
             max_tokens: int = 32768, **kwargs) -> ModelResponse:
        """Make a synchronous call to Gemini."""
        # Format messages for Gemini API
        formatted_messages = self._format_messages(messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        model = genai.GenerativeModel(model_name=model_name)
        
        try:
            response = model.generate_content(
                formatted_messages,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **{k: v for k, v in kwargs.items() if k not in ['model']}
                }
            )
            
            # Check if response has text content
            if hasattr(response, "text"):
                content = response.text
            else:
                # Handle case when response doesn't have text
                content = "[No content generated]"
                
            # Create standardized response
            return ModelResponse(
                content=content,
                model_name=model_name,
                metadata={"candidate_index": 0}
            )
        except Exception as e:
            # Return the error as a response
            error_message = f"Error from Gemini API: {str(e)}"
            return ModelResponse(
                content=error_message,
                model_name=model_name,
                metadata={"type": "error", "error": str(e)}
            )
    
    def stream_call(self, messages: List[Message], temperature: float = 1.0, 
                    max_tokens: int = 32768, **kwargs) -> Iterator[ModelResponse]:
        """Make a streaming call to Gemini."""
        # Format messages for Gemini API
        formatted_messages = self._format_messages(messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        model = genai.GenerativeModel(model_name=model_name)
        
        try:
            response = model.generate_content(
                formatted_messages,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **{k: v for k, v in kwargs.items() if k not in ['model']}
                },
                stream=True
            )
            
            # Stream the response chunks safely
            for chunk in response:
                # Check if chunk has valid text content
                if hasattr(chunk, "candidates") and chunk.candidates and hasattr(chunk, "text"):
                    text = chunk.text
                    if text:
                        yield ModelResponse(
                            content=text,
                            model_name=model_name,
                            metadata={"type": "chunk"}
                        )
        except Exception as e:
            # Log and yield the error as a response
            error_message = f"Error from Gemini API: {str(e)}"
            yield ModelResponse(
                content=error_message,
                model_name=model_name,
                metadata={"type": "error", "error": str(e)}
            )
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for configuration."""
        return "gemini"

# Register model with factory
ModelFactory.register("gemini", Gemini) 