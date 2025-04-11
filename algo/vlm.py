import os
import re
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Any, Literal
import logging

from utils import preprocess_ppt, extract_and_save_slides
from models.factory import ModelFactory
from algo.get_prompts import write_vlm
from storage import ScriptStorage, TaskStatus

from models import (
    Message, MessageRole, MessageContent, ContentEntry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ppt_with_model(
    ppt_path: str, 
    model_provider: Literal["claude", "gemini", "ollama"] = "claude",
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 32768
) -> Dict[str, str]:
    """
    Process a PowerPoint file using selected model with built-in retry mechanism.
    
    Args:
        ppt_path: Path to the PowerPoint file
        model_provider: The model provider to use ('claude', 'gemini', 'ollama')
        model_name: Specific model name (optional, uses default if not specified)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # Get the prompt
    prompt = write_vlm()
    
    # Initialize storage
    storage = ScriptStorage()
    
    # Create a task in storage with algorithm set to 'vlm'
    task_id = storage.create_task(ppt_path, model_provider, model_name, algo="vlm")
    
    # Use the preprocess function from caption.py to convert PPT to images
    logger.info(f"Preprocessing PowerPoint file: {ppt_path}")
    text_content, image_urls = preprocess_ppt(ppt_path)

    if not image_urls:
        logger.error(f"No valid images generated from {ppt_path}")
        # Set error in storage
        storage.set_error(task_id, "No valid images generated from PowerPoint file")
        return {}
    
    slide_count = len(image_urls)
    # Update task with slide count
    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)
    
    logger.info(f"Processing {slide_count} PPT images with extracted text...")
    
    # Prepare messages for model API
    messages = []
    
    # Add system message with the prompt
    messages.append(Message(
        role=MessageRole.SYSTEM,
        content=MessageContent.from_text(prompt)
    ))
    
    # Create MessageContent with text and images
    message_content = MessageContent()
    
    # Add text instruction
    message_content.entries.append(ContentEntry.text(
        "Please generate lecture notes for each slide. Format the output with ===SLIDE 1=== etc. markers."
    ))
    
    # Add all images and text content
    for i, image_url in enumerate(image_urls):
        # Extract the slide number (1-indexed)
        slide_num = i + 1
        
        # Add text content for this slide
        slide_text = text_content.get(str(i), "")  # Get text content for this slide (0-indexed in text_content)
        slide_info = f"This is slide {slide_num}. Extracted text content from this slide:\n{slide_text}"
        message_content.entries.append(ContentEntry.text(slide_info))
        
        # Add image to the message content
        message_content.entries.append(ContentEntry.image(image_url))

    # Add user message to the conversation
    messages.append(Message(
        role=MessageRole.USER,
        content=message_content
    ))
    
    # Prepare model-specific parameters
    model_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if model_name:
        model_kwargs["model"] = model_name

    try:
        logger.info(f"Calling {model_provider} API...")
        
        model = ModelFactory.get_model(model_provider)
        response = model.stream_call(messages, **model_kwargs)
        
        accumulated_response = ""
        saved_slides = set()  # Track slide numbers we've already saved
        
        for chunk in response:
            chunk_content = chunk.content
            print(chunk_content, end="", flush=True)
            accumulated_response += chunk_content
            
            # Process the accumulated response to extract and save completed slides
            saved_slides, _ = extract_and_save_slides(
                accumulated_response, task_id, storage, saved_slides, is_final=False
            )
        
        # Process the final response to save any remaining slides
        full_response = accumulated_response
        saved_slides, scripts = extract_and_save_slides(
            full_response, task_id, storage, saved_slides, is_final=True
        )

        if len(scripts) != slide_count:
            logger.error(f"Failed to process {slide_count} slides")
            storage.set_error(task_id, f"Failed to process {slide_count} slides")
            return {}
        
        logger.info(f"Successfully extracted content for {len(scripts)} slides")
        
        # Update task status to completed
        storage.update_task_status(task_id, TaskStatus.COMPLETED)
        
        return scripts
        
    except Exception as e:
        logger.error(f"Error processing PowerPoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Set error in storage
        storage.set_error(task_id, str(e))
        
        return {}