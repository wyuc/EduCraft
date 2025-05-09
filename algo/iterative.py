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
from utils import generate_captions_for_slides
from utils.rag import search_knowledge_base, DEFAULT_EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ppt_with_iterative(
    input_path: str,
    text_content: Optional[Dict[str, str]],
    image_urls: List[str],
    model_provider: str,
    model_name: str,
    temperature: float = 0.6,
    caption_model_provider: Literal["gpt", "gemini", "vllm", "claude", "qwen"] = "gpt",
    caption_model_name: Optional[str] = None,
    use_rag: bool = False,
    kb_path: Optional[str] = None, 
    embedding_model: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, str]:
    """
    Process a input file using iterative approach.
    
    Args:
        input_path: Path to the input file
        text_content: Dictionary with slide numbers as keys and extracted text content as values
        image_urls: List of image URLs for each slide
        model_provider: The model provider to use (default: 'gpt')
        model_name: Specific model name (default: 'gpt-4o-2024-05-13')
        temperature: Temperature parameter for generation
        caption_model_provider: The model provider to use for caption generation (for PDF files)
        caption_model_name: Specific model name for caption generation (for PDF files)
        use_rag: Whether to use RAG for enhanced context
        kb_path: Path to the knowledge base directory
        embedding_model: Embedding model to use for vector search
        top_k: Number of top results to return from knowledge base
        
    Returns:
        Dictionary with slide numbers as keys and lecture notes as values
    """
    # System prompt for the iterative approach
    system_prompt = """This agent speaks Chinese. Lecture Script Writer's primary function is to analyze PowerPoint (PPT) slides based on user inputs and the texts extracted from those slides. It then generates a script for teachers to teach students about the content illustrated on the page, assuming the role of the teacher who also made the slides. The script is intended for the teacher to read out loud, directly engaging with the audience without referring to itself as an external entity. It focuses on educational content, suitable for classroom settings or self-study. It emphasizes clarity, accuracy, and engagement in explanations, avoiding overly technical jargon unless necessary. The agent is not allowed to ask the user any questions even if the provided information is insufficient or unclear, ensuring the responses have to be a script. The script for each slide is limited to no more than two sentences, leaving most of the details to be discussed when interacting with the student's questions. The scripts for each slide has to be consistant to the previouse slide and it is important to make sure the agent's generated return can be directly joined as a fluent and continued script without any further adjustment. The agent should also never assume what is one the next slide before processing it. It adopts a friendly and supportive tone, encouraging learning and curiosity."""
    
    # Add RAG instructions if enabled
    if use_rag:
        system_prompt += "\n\n" + """
IMPORTANT: You may be provided with additional knowledge relevant to this presentation.
Use this information to enhance your lecture notes with accurate facts, definitions, 
and context that might be missing from the slides alone. Ensure you:

1. Integrate relevant knowledge naturally into your lecture notes
2. Maintain accurate information that aligns with the slide content
3. Add depth to explanations where appropriate
4. Do not add irrelevant information that distracts from the main topic
5. Make sure your explanations stay focused on the slides' content
"""
    
    # Initialize storage
    storage = ScriptStorage()

    if isinstance(input_path, str):
        input_path = Path(input_path)

    file_name = input_path.stem
    
    # Create a task in storage with algorithm set to 'iterative'
    algo_name = "iterative_rag" if use_rag else "iterative"
    task_id = storage.create_task(file_name, model_provider, model_name, algo=algo_name)
    
    slide_count = len(image_urls)

    storage.update_task_status(task_id, TaskStatus.PROCESSING, slide_count)
    
    logger.info(f"Processing {slide_count} slides with iterative approach...")
    
    captions = generate_captions_for_slides(
        image_urls=image_urls, 
        text_content=None,
        input_path=input_path,
        caption_model_provider=caption_model_provider,
        caption_model_name=caption_model_name,
        save_captions=True
    )
    
    # Convert captions to the text_content format
    for slide_num, caption in captions.items():
        # Convert to 0-indexed for text_content
        slide_idx = str(int(slide_num) - 1)
        text_content[slide_idx] = caption
    
    logger.info(f"Generated captions for {len(captions)} slides")
    
    # Replace the initial RAG search with per-slide RAG context retrieval
    rag_enabled = use_rag and kb_path and os.path.exists(kb_path)
    if rag_enabled:
        logger.info(f"RAG is enabled with knowledge base at {kb_path}")
        
        # Validate knowledge base
        index_path = os.path.join(kb_path, "index.json")
        if not os.path.exists(index_path):
            logger.error(f"Knowledge base index file not found: {index_path}")
            storage.set_error(task_id, f"Knowledge base index file not found: {index_path}")
            return {}
            
        # Set default embedding model if not provided
        if embedding_model is None:
            embedding_model = DEFAULT_EMBEDDING_MODEL
            logger.info(f"Using default embedding model: {embedding_model}")
    
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
            
            # If RAG is enabled, retrieve context for this specific slide
            if rag_enabled:
                slide_num = slide_number + 1  # 1-indexed for captions
                caption = captions.get(str(slide_num), "")
                
                if caption:
                    # Use this slide's caption as the query
                    logger.info(f"Retrieving knowledge for slide {slide_num} using its caption")
                    query = caption[:2000]  # Limit query length
                    
                    # Search knowledge base for this specific slide
                    search_results = search_knowledge_base(
                        kb_path, query, top_k=top_k, embedding_model=embedding_model
                    )
                    
                    if search_results:
                        # Format retrieved content into context
                        rag_context = f"### RETRIEVED KNOWLEDGE FOR SLIDE {slide_num} ###\n\n"
                        for i, result in enumerate(search_results):
                            source = result.get("metadata", {}).get("title", "Unknown")
                            content = result.get("content", "").strip()
                            score = result.get("score", 0)
                            
                            # Add each piece of retrieved content
                            rag_context += f"[Source {i+1}: {source} (relevance: {score:.2f})]\n"
                            rag_context += f"{content}\n\n"
                        
                        logger.info(f"Retrieved {len(search_results)} relevant passages for slide {slide_num}")
                        
                        # Add the RAG context to the message
                        message_content.entries.append(ContentEntry.text(
                            f"Relevant knowledge for this slide:\n\n{rag_context}\n\n"
                        ))
                    else:
                        logger.info(f"No relevant information found for slide {slide_num}")
            
            # Add slide text
            if slide_text:
                message_content.entries.append(ContentEntry.text(slide_text))
            else:
                message_content.entries.append(ContentEntry.text("No text content available for this slide"))
            
            # Add image
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
        