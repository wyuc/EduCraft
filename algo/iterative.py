import os
import re
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Any, Literal
import logging
from tqdm import tqdm

from models.factory import ModelFactory
from storage import ScriptStorage, TaskStatus
from models.base import Message, MessageRole, MessageContent, ContentEntry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ppt_with_iterative(
    file_name: str,
    text_content: Dict[str, str],
    image_urls: List[str],
    model_provider: str,
    model_name: str,
    temperature: float = 0.6,
) -> Dict[str, str]:
    """
    Process a input file using iterative approach.
    
    Args:
        file_name: Name of the input file
        text_content: Dictionary with slide numbers as keys and extracted text content as values
        image_urls: List of image URLs for each slide
        model_provider: The model provider to use (default: 'gpt')
        model_name: Specific model name (default: 'gpt-4o-2024-05-13')
        temperature: Temperature parameter for generation
        
    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # System prompt for the iterative approach
    system_prompt = """This agent speaks Chinese. Lecture Script Writer's primary function is to analyze PowerPoint (PPT) slides based on user inputs and the texts extracted from those slides. It then generates a script for teachers to teach students about the content illustrated on the page, assuming the role of the teacher who also made the slides. The script is intended for the teacher to read out loud, directly engaging with the audience without referring to itself as an external entity. It focuses on educational content, suitable for classroom settings or self-study. It emphasizes clarity, accuracy, and engagement in explanations, avoiding overly technical jargon unless necessary. The agent is not allowed to ask the user any questions even if the provided information is insufficient or unclear, ensuring the responses have to be a script. The script for each slide is limited to no more than two sentences, leaving most of the details to be discussed when interacting with the student's questions. The scripts for each slide has to be consistant to the previouse slide and it is important to make sure the agent's generated return can be directly joined as a fluent and continued script without any further adjustment. The agent should also never assume what is one the next slide before processing it. It adopts a friendly and supportive tone, encouraging learning and curiosity."""
    
    # Initialize storage
    storage = ScriptStorage()
    
    # Create a task in storage with algorithm set to 'iterative'
    task_id = storage.create_task(file_name, model_provider, model_name, algo="iterative")
    
    slide_count = len(image_urls)

    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)
    
    logger.info(f"Processing {slide_count} slides with iterative approach...")
    
    # Context management for iterative approach
    ctx_len = 3
    messages = []
    # Start with system message
    messages.append(Message(
        role=MessageRole.SYSTEM,
        content=MessageContent.from_text(system_prompt)
    ))
    
    scripts = {}
    
    try:
        # Process each slide iteratively
        for slide_number, image_url in enumerate(image_urls):
            if text_content:
                # Extract text for this slide (0-indexed in image_urls)
                slide_text = text_content.get(str(slide_number), "")
            else:
                slide_text = ""
            
            # Create message content with text and image
            message_content = MessageContent()
            if slide_text:
                message_content.entries.append(ContentEntry.text(slide_text))
            else:
                message_content.entries.append(ContentEntry.text("No text content available for this slide"))
            message_content.entries.append(ContentEntry.image(image_url))
            
            # Create user message
            user_message = Message(
                role=MessageRole.USER,
                content=message_content
            )
            
            # Keep only system message and recent conversation history
            if len(messages) > 1 + (ctx_len * 2):  # +1 for system message
                # Keep system message and most recent messages
                messages = [messages[0]] + messages[-(ctx_len * 2):]
                
            # Add current message
            messages.append(user_message)
            
            # Call the model
            model = ModelFactory.get_model(model_provider)
            model_kwargs = {
                "temperature": temperature
            }
            if model_name:
                model_kwargs["model"] = model_name
            response = model.call(
                messages=messages,
                **model_kwargs
            )
            
            # Get the response content
            response_text = response.content
            
            # Save the script for this slide
            slide_num = slide_number + 1  # Convert to 1-indexed for storage
            storage.save_slide(task_id, slide_num, response_text)
            scripts[str(slide_num)] = response_text
            
            logger.info(f"Generated script for slide {slide_num}/{slide_count}")
            
            # Add response to context
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=MessageContent.from_text(response_text)
            )
            messages.append(assistant_message)
        
        # All slides processed successfully
        if len(scripts) == slide_count:
            storage.update_task_status(task_id, TaskStatus.COMPLETED)
            logger.info(f"Successfully processed all {slide_count} slides")
        else:
            error_msg = f"Only processed {len(scripts)} out of {slide_count} slides"
            logger.error(error_msg)
            storage.set_error(task_id, error_msg)
        
        return scripts
        
    except Exception as e:
        logger.error(f"Error processing PowerPoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        storage.set_error(task_id, str(e))
        return {}
        