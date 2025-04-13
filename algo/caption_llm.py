from utils import get_temp_dir, extract_and_save_slides
import json
import logging
from typing import Dict, List, Optional, Any, Literal
from models.factory import ModelFactory
from storage import ScriptStorage, TaskStatus
from algo.get_prompts import caption, write_caption_llm

from models import (
    Message, MessageRole, MessageContent, ContentEntry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_ppt_with_caption_llm(
    file_name: str,
    text_content: Dict[str, str],
    image_urls: List[str],
    model_provider: Literal["claude", "gemini", "gpt", "deepseek"] = "gpt",
    model_name: Optional[str] = "gpt-4o-2024-11-20",
    caption_model_provider: Literal["gpt", "gemini"] = "gpt",
    caption_model_name: Optional[str] = "gpt-4o-2024-11-20",
    temperature: float = 0.6,
    max_tokens: int = 32768
) -> Dict[str, str]:
    """
    Process a PowerPoint file using caption-then-LLM approach:
    1. Generate captions for each slide
    2. Feed the captions to an LLM to generate the final lecture script
    
    Args:
        file_name: Name of the PowerPoint file
        text_content: Dictionary with slide numbers as keys and extracted text content as values
        image_urls: List of image URLs for each slide
        model_provider: The model provider to use for the final script generation
        model_name: Specific model name for script generation (optional, uses default if not specified)
        caption_model_provider: The model provider to use for caption generation (default: gpt)
        caption_model_name: Specific model name for caption generation (default: gpt-4o)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # Get the prompts
    caption_prompt = caption()  # Caption prompt from caption.txt
    writing_prompt = write_caption_llm()  # Prompt for converting captions to lecture script
    
    # Initialize storage
    storage = ScriptStorage()
    
    # Create a task in storage with algorithm set to 'caption_llm'
    task_id = storage.create_task(file_name, model_provider, model_name, algo="caption_llm")
   
    slide_count = len(image_urls)
    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)
    
    # Step 1: Generate captions for each slide
    logger.info(f"Generating captions for {slide_count} slides using {caption_model_provider}...")
    
    # Get model for caption generation
    caption_model = ModelFactory.get_model(caption_model_provider)
    captions = {}
    
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
            content=MessageContent.from_text(caption_prompt)
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
            storage.set_error(task_id, f"Error generating caption for slide {slide_num}: {str(e)}")
            return {}
    
    # Save all captions to a JSON file for reference
    try:
        output_dir = get_temp_dir(file_name)
        with open(output_dir / "captions.json", "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved captions to {output_dir / 'captions.json'}")
    except Exception as e:
        logger.warning(f"Failed to save captions JSON: {str(e)}")
    
    # Step 2: Generate lecture script from captions
    logger.info(f"Generating lecture script from captions using {model_provider}...")
    
    # Create system message with the writing prompt
    system_message = Message(
        role=MessageRole.SYSTEM,
        content=MessageContent.from_text(writing_prompt)
    )
    
    # Create user message with all captions
    captions_text = json.dumps(captions, ensure_ascii=False, indent=4)
    user_message = Message(
        role=MessageRole.USER,
        content=MessageContent.from_text(captions_text)
    )
    
    # Get the model for script generation
    script_model = ModelFactory.get_model(model_provider)
    
    # Prepare model-specific parameters
    model_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if model_name:
        model_kwargs["model"] = model_name
    
    try:
        logger.info(f"Calling {model_provider} API for script generation...")
        
        # Stream the response to show progress
        response = script_model.stream_call(
            messages=[system_message, user_message], 
            **model_kwargs
        )
        
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
            logger.warning(f"Only processed {len(scripts)} out of {slide_count} slides")
        
        logger.info(f"Successfully extracted content for {len(scripts)} slides")
        
        # Update task status to completed
        storage.update_task_status(task_id, TaskStatus.COMPLETED)
        
        return scripts
        
    except Exception as e:
        logger.error(f"Error generating lecture script: {str(e)}")
        import traceback
        traceback.print_exc()
        
        storage.set_error(task_id, str(e))
        return {}