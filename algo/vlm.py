import os
import re
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Any, Literal
import logging
import json

from utils import preprocess_ppt, extract_and_save_slides, generate_captions_for_slides, get_temp_dir
from models.factory import ModelFactory
from algo.get_prompts import write_vlm
from storage import ScriptStorage, TaskStatus
from utils.rag import search_knowledge_base, DEFAULT_EMBEDDING_MODEL

from models import (
    Message, MessageRole, MessageContent, ContentEntry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ppt_with_model(
    input_path: str,
    text_content: Optional[dict],
    image_urls: List[str],
    model_provider: Literal["claude", "gemini", "ollama", "qwen"] = "claude",
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 32768,
    use_rag: bool = False,
    kb_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    top_k: int = 5,
    caption_model_provider: Literal["gpt", "gemini", "vllm", "claude", "qwen"] = "gpt",
    caption_model_name: Optional[str] = None,
    prompt_variant: str = 'full'
) -> Dict[str, str]:
    """
    Process a input file using selected model with built-in retry mechanism.
    
    Args:
        input_path: Path to the input file
        text_content: Dictionary with slide numbers as keys and extracted text content as values
        image_urls: List of image URLs for each slide
        model_provider: The model provider to use ('claude', 'gemini', 'ollama')
        model_name: Specific model name (optional, uses default if not specified)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        use_rag: Whether to use RAG for enhanced context
        kb_path: Path to the knowledge base directory
        embedding_model: Embedding model to use for vector search
        top_k: Number of top results to return from knowledge base
        caption_model_provider: The model provider to use for caption generation (for PDF files)
        caption_model_name: Specific model name for caption generation (for PDF files)
        prompt_variant: The variant of the prompt to use (e.g., 'full', 'no_narrative')
        
    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # Get the prompt based on the variant
    logger.info(f"Loading prompt variant: {prompt_variant}")
    prompt = write_vlm(variant=prompt_variant)
    
    # Initialize storage
    storage = ScriptStorage()

    if isinstance(input_path, str):
        input_path = Path(input_path)

    file_name = input_path.stem
    
    # Log all important parameters
    logger.info(f"Processing presentation {file_name} with model provider: {model_provider}")
    if model_name:
        logger.info(f"Using specific model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max tokens: {max_tokens}")
    
    # Create a task in storage with algorithm set to 'vlm' + variant name
    base_algo_name = "vlm"
    if use_rag:
        base_algo_name += "_rag" # e.g., vlm_rag

    # Append variant if not 'full'
    if prompt_variant != 'full':
         algo_name = f"{base_algo_name}_{prompt_variant}" # e.g., vlm_no_narrative or vlm_rag_no_narrative
    else:
         algo_name = base_algo_name # e.g., vlm or vlm_rag
    
    logger.info(f"Storing task under effective algorithm name: {algo_name}")
    task_id = storage.create_task(file_name, model_provider, model_name, algo=algo_name)
    logger.info(f"Preprocessing input file: {file_name}")
   
    slide_count = len(image_urls)
    # Update task with slide count
    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)
    
    logger.info(f"Processing {slide_count} images...")

    # Generate captions for slides
    captions = generate_captions_for_slides(
        image_urls=image_urls,
        text_content=text_content,
        input_path=input_path,
        caption_model_provider=caption_model_provider,
        caption_model_name=caption_model_name,
        save_captions=True
    )
    logger.info(f"Generated captions for {len(captions)} slides")

    # Retrieve context from knowledge base if RAG is enabled
    rag_context = ""
    if use_rag:
        if not kb_path:
            logger.error("RAG is enabled but no knowledge base path provided")
            storage.set_error(task_id, "RAG is enabled but no knowledge base path provided")
            return {}
            
        # Check if knowledge base exists
        if not os.path.exists(kb_path):
            logger.error(f"Knowledge base path does not exist: {kb_path}")
            storage.set_error(task_id, f"Knowledge base path does not exist: {kb_path}")
            return {}
            
        # Check for index.json in knowledge base
        index_path = os.path.join(kb_path, "index.json")
        if not os.path.exists(index_path):
            logger.error(f"Knowledge base index file not found: {index_path}")
            storage.set_error(task_id, f"Knowledge base index file not found: {index_path}")
            return {}
            
        logger.info(f"Using RAG with knowledge base at {kb_path}")
        
        # Create a query from the captions
        # Use captions directly as the query
        caption_values = list(captions.values())
        query = " ".join(caption_values)
        if len(query) > 2000:  # Limit query length
            query = query[:2000]
        
        logger.info(f"Generated RAG query: {query[:100]}...")
        
        # Search knowledge base
        logger.info(f"Searching knowledge base with {embedding_model or DEFAULT_EMBEDDING_MODEL} embedding model, top_k={top_k}")
        search_results = search_knowledge_base(
            kb_path, query, top_k=top_k, embedding_model=embedding_model
        )
        
        if search_results:
            # Format retrieved content into context
            rag_context = "### RETRIEVED KNOWLEDGE ###\n\n"
            for i, result in enumerate(search_results):
                source = result.get("metadata", {}).get("title", "Unknown")
                content = result.get("content", "").strip()
                score = result.get("score", 0)
                
                # Add each piece of retrieved content
                rag_context += f"[Source {i+1}: {source} (relevance: {score:.2f})]\n"
                rag_context += f"{content}\n\n"
            
            logger.info(f"Retrieved {len(search_results)} relevant passages from knowledge base")
        else:
            logger.warning("No relevant information found in knowledge base")

    
    # Prepare messages for model API
    messages = []
    
    # Add system message with the prompt
    system_prompt = prompt
    if use_rag and rag_context:
        # Enhance the prompt with RAG instructions
        system_prompt += "\n\n" + """
IMPORTANT: I've provided you with additional knowledge relevant to this presentation.
Use this information to enhance your lecture notes with accurate facts, definitions, 
and context that might be missing from the slides alone. Ensure you:

1. Integrate relevant knowledge naturally into your lecture notes
2. Maintain accurate information that aligns with the slide content
3. Add depth to explanations where appropriate
4. Do not add irrelevant information that distracts from the main topic
5. Make sure your explanations stay focused on the slides' content
"""
    
    messages.append(Message(
        role=MessageRole.SYSTEM,
        content=MessageContent.from_text(system_prompt)
    ))
    
    # Create MessageContent with text and images
    message_content = MessageContent()
    
    # Add RAG context if available
    if use_rag and rag_context:
        message_content.entries.append(ContentEntry.text(
            f"Here is relevant knowledge to enhance your lecture notes:\n\n{rag_context}"
        ))
    
    # Add text instruction
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
            slide_info = f"This is slide {slide_num}. No text content available."
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
        logger.error(f"Failed to process all slides. Expected {slide_count}, got {len(scripts)}")
        storage.set_error(task_id, f"Failed to process all slides. Expected {slide_count}, got {len(scripts)}")
        return scripts  # Return what we have instead of empty dict
    
    logger.info(f"Successfully extracted content for {len(scripts)} slides")
    
    # Update task status to completed
    storage.update_task_status(task_id, TaskStatus.COMPLETED)
    
    return scripts