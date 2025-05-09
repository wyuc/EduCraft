from utils import extract_and_save_slides, generate_captions_for_slides
import json
import logging
import os
from typing import Dict, List, Optional, Any, Literal
from models.factory import ModelFactory
from storage import ScriptStorage, TaskStatus
from algo.get_prompts import caption, write_caption_llm
from pathlib import Path
from utils.rag import search_knowledge_base, DEFAULT_EMBEDDING_MODEL

from models import (
    Message, MessageRole, MessageContent, ContentEntry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_ppt_with_caption_llm(
    input_path: str,
    text_content: Optional[Dict[str, str]],
    image_urls: List[str],
    model_provider: Literal["claude", "gemini", "gpt", "deepseek", "qwen"] = "gpt",
    model_name: Optional[str] = "gpt-4o-2024-11-20",
    caption_model_provider: Literal["gpt", "gemini", "vllm", "claude", "qwen"] = "gpt",
    caption_model_name: Optional[str] = "gpt-4o-2024-11-20",
    temperature: float = 0.6,
    max_tokens: int = 32768,
    use_rag: bool = False,
    kb_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, str]:
    """
    Process a presentation file using caption-then-LLM approach:
    1. Generate captions for each slide
    2. Feed the captions to an LLM to generate the final lecture script
    
    Args:
        input_path: Path to the input file
        text_content: Dictionary with slide numbers as keys and extracted text content as values
        image_urls: List of image URLs for each slide
        model_provider: The model provider to use for the final script generation
        model_name: Specific model name for script generation (optional, uses default if not specified)
        caption_model_provider: The model provider to use for caption generation (default: gpt)
        caption_model_name: Specific model name for caption generation (default: gpt-4o)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        use_rag: Whether to use RAG for enhanced context
        kb_path: Path to the knowledge base directory
        embedding_model: Embedding model to use for vector search
        top_k: Number of top results to return from knowledge base
        
    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # Get the prompts
    caption_prompt = caption()  # Caption prompt from caption.txt
    writing_prompt = write_caption_llm()  # Prompt for converting captions to lecture script
    
    # Initialize storage
    storage = ScriptStorage()
    
    if isinstance(input_path, str):
        input_path = Path(input_path)

    file_name = input_path.stem
    
    # Create a task in storage with algorithm set to 'caption_llm'
    algo_name = "caption_llm_rag" if use_rag else "caption_llm"
    task_id = storage.create_task(file_name, model_provider, model_name, algo=algo_name)
   
    slide_count = len(image_urls)
    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)
    
    # Step 1: Generate captions for each slide
    logger.info(f"Generating captions for {slide_count} slides using {caption_model_provider}...")
    
    captions = generate_captions_for_slides(
        image_urls=image_urls,
        text_content=text_content,
        input_path=input_path,
        caption_model_provider=caption_model_provider,
        caption_model_name=caption_model_name,
        prompt=caption_prompt,
        save_captions=True
    )
    
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
    
    # Step 2: Generate lecture script from captions
    logger.info(f"Generating lecture script from captions using {model_provider}...")
    
    # Create system message with the writing prompt
    system_prompt = writing_prompt
    if use_rag and rag_context:
        # Enhance the prompt with RAG instructions
        system_prompt += "\n\n" + """
IMPORTANT: I've provided you with additional knowledge relevant to this presentation.
Use this information to enhance your lecture notes with accurate facts, definitions, 
and context that might be missing from the captions alone. Ensure you:

1. Integrate relevant knowledge naturally into your lecture notes
2. Maintain accurate information that aligns with the slide content
3. Add depth to explanations where appropriate
4. Do not add irrelevant information that distracts from the main topic
5. Make sure your explanations stay focused on the slides' content
"""
    
    system_message = Message(
        role=MessageRole.SYSTEM,
        content=MessageContent.from_text(system_prompt)
    )
    
    # Create user message with all captions
    captions_text = json.dumps(captions, ensure_ascii=False, indent=4)
    
    # Add RAG context if available
    user_text = captions_text
    if use_rag and rag_context:
        user_text = f"Here is relevant knowledge to enhance your lecture notes:\n\n{rag_context}\n\nCaption data:\n{captions_text}"
    
    user_message = Message(
        role=MessageRole.USER,
        content=MessageContent.from_text(user_text)
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
            logger.error(f"Failed to process {slide_count} slides")
            storage.set_error(task_id, f"Failed to process {slide_count} slides")
            return {}
        
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