from typing import List, Iterator
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, AsyncAnthropic
from anthropic.types import MessageParam
from .base import (
    ModelInterface, ModelResponse, Message, MessageRole, MessageContent,
    ContentEntry, ContentType
)
from .factory import ModelFactory

class Claude(ModelInterface):
    """Interface for Anthropic's Claude models."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the Claude interface with API key from configuration."""
        super().__init__(base_url, api_key)
        
        # Initialize Claude client
        self.client = Anthropic(base_url=self.base_url, api_key=self.api_key)

    def _format_messages(self, messages: List[Message]) -> tuple[str, List[MessageParam]]:
        """Format messages for Claude API format.
        
        Returns:
            tuple: (system_message, formatted_messages)
            - system_message: The system message content or empty string if none
            - formatted_messages: List of formatted messages for Claude API
        """
        formatted_messages = []
        system_message = ""
        
        for msg in messages:
            # Extract system messages separately
            if msg.role == MessageRole.SYSTEM:
                # Get system message content
                if isinstance(msg.content, str):
                    system_message = msg.content
                elif hasattr(msg.content, 'entries') and msg.content.entries:
                    # Get text from content entries
                    for entry in msg.content.entries:
                        if entry.type == ContentType.TEXT:
                            system_message += entry.data.get("text", "")
                else:
                    # Fallback for other content types
                    system_message = str(msg.content)
                continue  # Skip adding system message to the messages array
            
            # Map other roles to Claude roles
            role = {
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "assistant"
            }.get(msg.role, "user")
            
            # Handle string content directly
            if isinstance(msg.content, str):
                formatted_messages.append({
                    "role": role,
                    "content": msg.content
                })
                continue
                
            # Handle MessageContent with entries
            if not hasattr(msg.content, 'entries'):
                # Handle legacy content format or unexpected content
                formatted_messages.append({
                    "role": role,
                    "content": str(msg.content)
                })
                continue
            
            content_entries = msg.content.entries
            
            # If there's only one text entry, use simple content format
            if len(content_entries) == 1 and content_entries[0].type == ContentType.TEXT:
                formatted_messages.append({
                    "role": role,
                    "content": content_entries[0].data.get("text", "")
                })
            else:
                # Multiple entries - use content array format for Claude
                content_array = []
                
                for entry in content_entries:
                    if entry.type == ContentType.TEXT:
                        content_array.append({
                            "type": "text",
                            "text": entry.data.get("text", "")
                        })
                    elif entry.type == ContentType.IMAGE:
                        url = entry.data.get("url", "")
                        if url.startswith("data:"):
                            # Handle base64 images
                            mime_type = url.split(';')[0].split(':')[1]
                            data = url.split(',')[1]
                            content_array.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": data
                                }
                            })
                        else:
                            # Handle URLs - Claude might not support direct URLs
                            # Consider adding a warning here
                            content_array.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url
                                }
                            })
                
                formatted_messages.append({
                    "role": role,
                    "content": content_array
                })
        
        return system_message, formatted_messages
    
    def call(self, messages: List[Message], temperature: float = 1.0, 
             max_tokens: int = 4096, **kwargs) -> ModelResponse:
        """Make a synchronous call to Claude."""
        # Format messages for Claude API, extracting system message
        system_message, formatted_messages = self._format_messages(messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        
        try:
            # Prepare parameters for the API call
            api_params = {
                "model": model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['model']}
            }
            
            # Add system parameter if present
            if system_message:
                api_params["system"] = system_message
            
            # Make the API call
            response = self.client.messages.create(**api_params)
            
            # Create standardized response
            return ModelResponse(
                content=response.content[0].text,
                model_name=model_name,
                metadata={"usage": response.usage}
            )
        except Exception as e:
            # Return the error as a response
            error_message = f"Error from Claude API: {str(e)}"
            return ModelResponse(
                content=error_message,
                model_name=model_name,
                metadata={"type": "error", "error": str(e)}
            )
    
    def stream_call(self, messages: List[Message], temperature: float = 1.0, 
                    max_tokens: int = 4096, **kwargs) -> Iterator[ModelResponse]:
        """Make a streaming call to Claude."""
        # Format messages for Claude API, extracting system message
        system_message, formatted_messages = self._format_messages(messages)
        
        # Get model name from kwargs or use default
        model_name = kwargs.get("model", self.default_model)
        
        try:
            # Prepare parameters for the API call
            api_params = {
                "model": model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['model']}
            }
            
            # Add system parameter if present
            if system_message:
                api_params["system"] = system_message
            
            # Make the streaming API call
            with self.client.messages.stream(**api_params) as stream:
                for chunk in stream:
                    # Check for content delta with text content
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and chunk.delta.text:
                        yield ModelResponse(
                            content=chunk.delta.text,
                            model_name=model_name,
                            metadata={"type": "chunk"}
                        )
                    # Skip ThinkingDelta and other non-text delta types
        except Exception as e:
            # Log and yield error as a response
            error_message = f"Error from Claude API: {str(e)}"
            yield ModelResponse(
                content=error_message,
                model_name=model_name,
                metadata={"type": "error", "error": str(e)}
            )
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for configuration."""
        return "claude"

# Register model with factory
ModelFactory.register("claude", Claude) 