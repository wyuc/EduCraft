"""
Slide extraction and management utilities for handling slide content.
"""

import re
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Literal
from utils.paths import get_temp_dir
from models.factory import ModelFactory
from storage import ScriptStorage

from models import (
    Message, MessageRole, MessageContent, ContentEntry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_captions_for_slides(
    image_urls: List[str],
    text_content: Optional[Dict[str, str]] = None,
    input_path: Optional[str] = None,
    caption_model_provider: Literal["gpt", "gemini", "vllm", "claude", "qwen"] = "gpt",
    caption_model_name: Optional[str] = None,
    prompt: Optional[str] = None,
    save_captions: bool = True
) -> Dict[str, str]:
    """
    Generate captions for a list of slides.
    
    Args:
        image_urls: List of image URLs for each slide
        text_content: Dictionary with slide indices as keys and extracted text content as values
        input_path: Path to the input file (for saving captions to disk)
        caption_model_provider: The model provider to use for caption generation
        caption_model_name: Specific model name for caption generation
        prompt: The prompt to use for caption generation
        save_captions: Whether to save captions to disk
        
    Returns:
        Dictionary with slide numbers as keys and captions as values
    """
    # Import here to avoid circular imports
    from algo.get_prompts import caption as default_caption_prompt
    
    if prompt is None:
        prompt = default_caption_prompt()
    
    # Check for existing captions file
    captions = {}
    output_dir = get_temp_dir(input_path)
    captions_file = output_dir / "captions.json"
    if input_path and save_captions:
        # Check if captions.json already exists
        if captions_file.exists():
            logger.info(f"Found existing captions file at {captions_file}, loading instead of regenerating")
            try:
                with open(captions_file, "r", encoding="utf-8") as f:
                    captions = json.load(f)
                logger.info(f"Successfully loaded captions for {len(captions)} slides")
                return captions
            except Exception as e:
                logger.error(f"Error loading existing captions file: {str(e)}")
                logger.info("Will regenerate captions")
                captions = {}
    
    # Get model for caption generation
    caption_model = ModelFactory.get_model(caption_model_provider)
    
    slide_count = len(image_urls)
    logger.info(f"Generating captions for {slide_count} slides using {caption_model_provider}...")
    
    for i, image_url in enumerate(image_urls):
        slide_num = i + 1  # Convert to 1-indexed
        
        if text_content:
            # Extract text for this slide
            slide_text = text_content.get(str(i), "")
        else:
            slide_text = ""
        
        # Create system message for caption generation
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=MessageContent.from_text(prompt)
        )
        
        # Create user message with image and extracted text
        message_content = MessageContent()
        if slide_text:
            message_content.entries.append(ContentEntry.text(
                f"Extracted text content:\n{slide_text}"
            ))
        else:
            message_content.entries.append(ContentEntry.text(
                f"No text content available for this slide"
            ))
        message_content.entries.append(ContentEntry.image(image_url))
        
        user_message = Message(
            role=MessageRole.USER,
            content=message_content
        )
        
        # Call the model to generate caption
        logger.info(f"Generating caption for slide {slide_num}/{slide_count}...")
        try:
            caption_kwargs = {
                "temperature": 0.3
            }
            if caption_model_name:
                caption_kwargs["model"] = caption_model_name
            response = caption_model.call(
                messages=[system_message, user_message],
                **caption_kwargs
            )
            caption_text = response.content
            captions[str(slide_num)] = caption_text
            logger.info(f"Caption generated for slide {slide_num}")
        except Exception as e:
            logger.error(f"Error generating caption for slide {slide_num}: {str(e)}")
            captions[str(slide_num)] = f"Error generating caption: {str(e)}"
            continue
    
    # Save all captions to a JSON file for reference
    if input_path and save_captions:
        with open(captions_file, "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved captions to {captions_file}")
            
    return captions

def extract_and_save_slides(accumulated_response, task_id, storage, saved_slides=None, is_final=False):
    """
    Extract slide content from accumulated response and save to storage.
    
    Args:
        accumulated_response: The text response containing slide content with markers
        task_id: The unique task identifier for storage
        storage: An instance of ScriptStorage to save slides
        saved_slides: Set of slide numbers that have already been saved (or None to initialize)
        is_final: Whether this is the final processing (True) or streaming (False)
        
    Returns:
        Tuple of (saved_slides, scripts) where:
            - saved_slides is a set of slide numbers that have been saved
            - scripts is a dictionary of all extracted scripts with slide numbers as keys
    """
    # Initialize saved_slides if not provided
    if saved_slides is None:
        saved_slides = set()
    
    # Pattern to extract slide content between markers
    slide_pattern = r'===SLIDE\s+(\d+)===\s*(.*?)(?=\s*===SLIDE\s+\d+===|\s*$)'
    
    # Extract all slides from the response
    matches = re.findall(slide_pattern, accumulated_response, re.DOTALL)
    if not matches:
        return saved_slides, {}
    
    # Dictionary to store all scripts (for return value)
    scripts = {}
    
    # Determine which slides to process:
    # - During streaming (is_final=False): Process all except the last one (which might be incomplete)
    # - During final processing (is_final=True): Process all slides
    slides_to_process = matches if is_final else matches[:-1]
    
    # Add all slides to the scripts dictionary
    for slide_num, content in matches:
        slide_num_str = slide_num.strip()
        content = content.strip()
        scripts[slide_num_str] = content
    
    # Save slides that should be processed and haven't been saved yet
    for slide_num, content in slides_to_process:
        slide_num_int = int(slide_num.strip())
        content = content.strip()
        
        if slide_num_int not in saved_slides:
            storage.save_slide(task_id, slide_num_int, content)
            saved_slides.add(slide_num_int)
            logger.debug(f"Saved slide {slide_num_int} to storage")
    
    return saved_slides, scripts
