import os
import re
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Any, Literal
import logging
import json

from utils import preprocess_ppt, extract_and_save_slides, generate_captions_for_slides, get_temp_dir
from models.factory import ModelFactory
# Removed: from algo.get_prompts import write_vlm
from storage import ScriptStorage, TaskStatus
# Removed: from utils.rag import search_knowledge_base, DEFAULT_EMBEDDING_MODEL

from models import (
    Message, MessageRole, MessageContent, ContentEntry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ppt_with_direct_prompt( # Renamed function
    input_path: str,
    text_content: Optional[dict],
    image_urls: List[str],
    model_provider: Literal["claude", "gemini", "ollama", "qwen", "gpt", "gemini_openai"] = "claude", # Added more common providers
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 32768,
    # Removed RAG parameters
    caption_model_provider: Literal["gpt", "gemini", "vllm", "claude", "qwen"] = "gpt",
    caption_model_name: Optional[str] = None,
    prompt_variant: str = "full"
) -> Dict[str, str]:
    """
    Process a input file using selected model with a minimal direct prompt.

    Args:
        input_path: Path to the input file
        text_content: Dictionary with slide numbers as keys and extracted text content as values
        image_urls: List of image URLs for each slide
        model_provider: The model provider to use
        model_name: Specific model name (optional, uses default if not specified)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        caption_model_provider: The model provider to use for caption generation (if needed)
        caption_model_name: Specific model name for caption generation (if needed)

    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # Minimal system prompt
    minimal_system_prompt = "Generate a lecture script for the following slides based on the images and text provided. Use ===SLIDE X=== markers between slides."

    # Initialize storage
    storage = ScriptStorage()

    if isinstance(input_path, str):
        input_path = Path(input_path)

    file_name = input_path.stem

    # Log all important parameters
    logger.info(f"Processing presentation {file_name} with direct prompt and model provider: {model_provider}")
    if model_name:
        logger.info(f"Using specific model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max tokens: {max_tokens}")

    # Create a task in storage with algorithm set to 'direct_prompt'
    algo_name = "direct_prompt"
    task_id = storage.create_task(file_name, model_provider, model_name, algo=algo_name)
    logger.info(f"Preprocessing input file: {file_name}")

    slide_count = len(image_urls)
    # Update task with slide count
    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)

    logger.info(f"Processing {slide_count} images...")

    # Generate captions for slides (still might be needed if text_content is None)
    captions = generate_captions_for_slides(
        image_urls=image_urls,
        text_content=text_content,
        input_path=input_path,
        caption_model_provider=caption_model_provider,
        caption_model_name=caption_model_name,
        save_captions=True
    )
    logger.info(f"Generated captions for {len(captions)} slides (used if text_content missing)")

    # Removed RAG logic block

    # Prepare messages for model API
    messages = []

    # Add system message with the minimal prompt
    system_prompt = minimal_system_prompt
    messages.append(Message(
        role=MessageRole.SYSTEM,
        content=MessageContent.from_text(system_prompt)
    ))

    # Create MessageContent with text and images
    message_content = MessageContent()

    # Removed RAG context adding

    # Add text instruction (basic)
    message_content.entries.append(ContentEntry.text(
        "Please generate lecture notes for each slide. Format the output with ===SLIDE 1=== etc. markers."
    ))

    # Add all images and text content
    for i, image_url in enumerate(image_urls):
        # Extract the slide number (1-indexed)
        slide_num = i + 1

        # Add text content for this slide
        if text_content:
            slide_text = text_content.get(str(i), "")  # Get text content for this slide (0-indexed in text_content)
            slide_info = f"This is slide {slide_num}. Extracted text content from this slide:\n{slide_text}"
            message_content.entries.append(ContentEntry.text(slide_info))
        else:
            # Use caption if text content is missing for this slide
            slide_caption = captions.get(str(i), "")
            if slide_caption:
                 slide_info = f"This is slide {slide_num}. Image caption: {slide_caption}"
            else:
                 slide_info = f"This is slide {slide_num}. No text content or caption available."
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

    logger.info(f"Calling {model_provider} API with direct prompt...")

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
        logger.error(f"Failed to process all slides. Expected {slide_count}, got {len(scripts)}")
        storage.set_error(task_id, f"Failed to process all slides. Expected {slide_count}, got {len(scripts)}")
        return scripts  # Return what we have instead of empty dict

    logger.info(f"Successfully extracted content for {len(scripts)} slides")

    # Update task status to completed
    storage.update_task_status(task_id, TaskStatus.COMPLETED)

    return scripts 