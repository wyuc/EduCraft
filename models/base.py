from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Iterator, Literal
from dataclasses import dataclass, field
from enum import Enum
from config import get_model_config

class MessageRole(str, Enum):
    """Standardized message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ContentType(str, Enum):
    """Types of content entries in a message."""
    TEXT = "text"
    IMAGE = "image"

@dataclass
class ContentEntry:
    """A single entry in a message content."""
    type: ContentType
    data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def text(cls, text: str) -> 'ContentEntry':
        """Create a text content entry."""
        return cls(type=ContentType.TEXT, data={"text": text})
    
    @classmethod
    def image(cls, url: str, alt_text: Optional[str] = None) -> 'ContentEntry':
        """Create an image content entry."""
        data = {"url": url}
        if alt_text:
            data["alt_text"] = alt_text
        return cls(type=ContentType.IMAGE, data=data)

@dataclass
class MessageContent:
    """Content of a message, consisting of multiple entries."""
    entries: List[ContentEntry] = field(default_factory=list)
    
    @classmethod
    def from_text(cls, text: str) -> 'MessageContent':
        """Create content from text."""
        return cls(entries=[ContentEntry.text(text)])
    
    @classmethod
    def from_image(cls, url: str, alt_text: Optional[str] = None) -> 'MessageContent':
        """Create content from an image."""
        return cls(entries=[ContentEntry.image(url, alt_text)])
    
    @classmethod
    def from_text_and_images(cls, text: str, image_urls: List[str]) -> 'MessageContent':
        """Create content from text and images."""
        entries = [ContentEntry.text(text)]
        for url in image_urls:
            entries.append(ContentEntry.image(url))
        return cls(entries=entries)

@dataclass
class Message:
    """Class representing a message in a conversation."""
    role: MessageRole
    content: Union[str, MessageContent]
    
    @classmethod
    def system(cls, text: str) -> 'Message':
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=MessageContent.from_text(text))
    
    @classmethod
    def user(cls, text: str) -> 'Message':
        """Create a user message with text."""
        return cls(role=MessageRole.USER, content=MessageContent.from_text(text))
    
    @classmethod
    def assistant(cls, text: str) -> 'Message':
        """Create an assistant message with text."""
        return cls(role=MessageRole.ASSISTANT, content=MessageContent.from_text(text))

class ModelResponse:
    """Class representing a response from a model."""
    def __init__(self, content: str, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.model_name = model_name
        self.metadata = metadata or {}

class ModelInterface(ABC):
    """Base interface that all model implementations must follow."""

    def __init__(self, base_url: str = None, api_key: str = None):
        # Get configuration for this provider
        config = get_model_config(self.get_provider_name())
        
        self.base_url = base_url if base_url else config.get("base_url", "")
        self.base_url = self.base_url.rstrip('/')
        self.api_key = api_key if api_key else config.get("api_key", "")
        self.default_model = config.get("default_model", "")
   
    @abstractmethod
    def call(self, messages: List[Message], temperature: float = 1.0, 
             max_tokens: int = 4096, **kwargs) -> ModelResponse:
        """Synchronous call to the model."""
        pass
    
    @abstractmethod
    def stream_call(self, messages: List[Message], temperature: float = 1.0, 
                    max_tokens: int = 4096, **kwargs) -> Iterator[ModelResponse]:
        """Streaming call to the model, returning an iterator of partial responses."""
        pass
    
    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for configuration."""
        pass
