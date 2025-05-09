#!/usr/bin/env python
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from algo.vlm import process_ppt_with_model
from algo.caption_llm import process_ppt_with_caption_llm
from algo.iterative import process_ppt_with_iterative
from algo.direct_prompt import process_ppt_with_direct_prompt
from storage import ScriptStorage
from scripts.scripts_to_excel import scripts_to_excel
from utils.ppt import preprocess_ppt
from utils import get_images_dir, convert_pdf_to_png, encode_images_to_base64
from utils.rag import DEFAULT_EMBEDDING_MODEL
from config import BASE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_ppt(
    input_path: Union[str, Path],
    algorithm: str = 'vlm',
    temperature: float = 0.7,
    output_dir: Optional[str] = None,
    export_excel: bool = False,
    model_params: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Unified interface for processing PowerPoint files with various algorithms.
    
    Args:
        input_path: Path to the input file (.pptx or .pdf)
        algorithm: Algorithm to use ('vlm', 'caption_llm', 'iterative', or 'direct_prompt')
        temperature: Temperature parameter for text generation
        output_dir: Directory to save output files (default: same as input file directory)
        export_excel: Whether to export scripts to Excel
        model_params: Dictionary of algorithm-specific parameters
            - For 'vlm': model_provider, model_name, max_tokens, [use_rag, kb_path, embedding_model, top_k], [prompt_variant]
            - For 'caption_llm': model_provider, model_name, caption_model_provider, caption_model_name, max_tokens, [use_rag, kb_path, embedding_model, top_k]
            - For 'iterative': model_provider, model_name, [use_rag, kb_path, embedding_model, top_k]
            - For 'direct_prompt': model_provider, model_name, max_tokens, caption_model_provider, caption_model_name
    
    Returns:
        List of dictionaries containing slide information and generated scripts
        
    Raises:
        ValueError: If algorithm is invalid or input_path doesn't exist or is unsupported type
        Exception: For any algorithm-specific errors
    """
    # Convert to Path object if string
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input_path}")

    # --- Unified Preprocessing --- 
    # Call the single preprocess_ppt function which handles both types
    try:
        logger.info(f"Preprocessing input file: {input_path.name}")
        text_content, image_urls = preprocess_ppt(input_path)
        
        # Check if preprocessing was successful (at least got images)
        if not image_urls:
             logger.error(f"Preprocessing failed to generate images for {input_path.name}. Aborting.")
             raise RuntimeError(f"Preprocessing failed for {input_path.name}")
             
    except (ValueError, FileNotFoundError, RuntimeError) as e: # Catch specific expected errors
        raise e # Re-raise to be caught by the main handler
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing of {input_path.name}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected preprocessing error for {input_path.name}") from e
    
    logger.info(f"Running algorithm: {algorithm}")
    
    # Extract caption model parameters (used if text_content is None)
    caption_model_provider = model_params.get('caption_model_provider', 'gpt')
    caption_model_name = model_params.get('caption_model_name')
    
    # Run the selected algorithm
    result = None
    
    try:
        if algorithm == 'vlm':
            # Check if RAG is enabled in model_params
            use_rag = model_params.get('use_rag', False)
            kb_path = model_params.get('kb_path', None)
            embedding_model = model_params.get('embedding_model')
            top_k = model_params.get('top_k', 5)
            
            if use_rag:
                logger.info(f"Using RAG with knowledge base at {kb_path}")
                
                # Set default embedding model if none provided
                if embedding_model is None:
                    embedding_model = DEFAULT_EMBEDDING_MODEL
                    logger.info(f"Using default embedding model: {embedding_model}")
            
            # Get prompt variant if specified
            prompt_variant = model_params.get('prompt_variant', 'full')

            # Run VLM algorithm
            result = process_ppt_with_model(
                input_path=input_path,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                temperature=temperature,
                max_tokens=model_params['max_tokens'],
                use_rag=use_rag,
                kb_path=kb_path,
                embedding_model=embedding_model,
                top_k=top_k,
                caption_model_provider=caption_model_provider,
                caption_model_name=caption_model_name,
                prompt_variant=prompt_variant
            )
        
        elif algorithm == 'caption_llm':
            # Run Caption LLM algorithm
            result = process_ppt_with_caption_llm(
                input_path=input_path,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                caption_model_provider=model_params['caption_model_provider'],
                caption_model_name=model_params['caption_model_name'],
                temperature=temperature,
                max_tokens=model_params['max_tokens'],
                use_rag=model_params.get('use_rag', False),
                kb_path=model_params.get('kb_path'),
                embedding_model=model_params.get('embedding_model'),
                top_k=model_params.get('top_k', 5),
                prompt_variant=model_params.get('prompt_variant', 'full')
            )
        
        elif algorithm == 'iterative':
            # Run Iterative algorithm
            result = process_ppt_with_iterative(
                input_path=input_path,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                temperature=temperature,
                caption_model_provider=caption_model_provider,
                caption_model_name=caption_model_name,
                use_rag=model_params.get('use_rag', False),
                kb_path=model_params.get('kb_path'),
                embedding_model=model_params.get('embedding_model'),
                top_k=model_params.get('top_k', 5),
                prompt_variant=model_params.get('prompt_variant', 'full')
            )
        
        elif algorithm == 'direct_prompt':
            result = process_ppt_with_direct_prompt(
                input_path=input_path,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                temperature=temperature,
                max_tokens=model_params['max_tokens'],
                caption_model_provider=caption_model_provider,
                caption_model_name=caption_model_name,
                prompt_variant=model_params.get('prompt_variant', 'full')
            )
        
        # Check the result
        if not result:
            logger.error("Algorithm returned no results")
            raise Exception("Algorithm returned no results")
        
        # If export to Excel is requested
        if export_excel:
            _output_dir = output_dir if output_dir else os.path.dirname(input_path)
            logger.info(f"Exporting scripts to Excel in {_output_dir}")
            scripts_to_excel(
                input_path=str(input_path),
                algo=algorithm,
                model_provider=model_params['model_provider'],
                output_dir=_output_dir
            )
        
        logger.info(f"Successfully processed {len(result)} slides")
        return result
        
    except Exception as e:
        logger.error(f"Error processing PowerPoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    # Set up the main argument parser
    parser = argparse.ArgumentParser(
        description='Generate lecture scripts from input files using various AI algorithms.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_path', help='Path to the input file')
    
    # Shared optional arguments
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                        help='Temperature parameter for text generation (higher values = more creative)')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files (default: same as input file directory)')
    parser.add_argument('--export-excel', '-e', action='store_true', 
                        help='Export scripts to Excel after generation')
    
    # Create subparsers for algorithms
    subparsers = parser.add_subparsers(dest='algorithm', help='Algorithm to use for script generation')
    subparsers.required = True  # Make algorithm selection required
    
    # VLM algorithm subparser
    vlm_parser = subparsers.add_parser('vlm', help='Vision Language Model algorithm')
    vlm_parser.add_argument('--model-provider', '-mp', choices=['gemini_openai', 'gemini', 'gpt', 'vllm', 'ollama', 'claude'], default='gemini_openai',
                         help='Model provider to use')
    vlm_parser.add_argument('--model-name', '-m',
                         help='Specific model name')
    vlm_parser.add_argument('--max-tokens', '-mt', type=int, default=32768,
                         help='Maximum tokens to generate')
    # Add RAG options to VLM
    vlm_parser.add_argument('--use-rag', action='store_true',
                         help='Enable Retrieval-Augmented Generation')
    vlm_parser.add_argument('--kb-path', type=str, 
                         help='Path to the knowledge base directory')
    vlm_parser.add_argument('--embedding-model', type=str,
                         help='Embedding model to use for vector search')
    vlm_parser.add_argument('--top-k', type=int, default=5,
                         help='Number of top results to return from knowledge base')
    # Add prompt variant option to VLM
    vlm_parser.add_argument('--prompt-variant', '-pv', 
                         choices=['full', 'no_narrative', 'no_mapping', 'no_grounding'], 
                         default='full',
                         help='Prompt variant to use for ablation study')
    
    # Caption LLM algorithm subparser
    caption_parser = subparsers.add_parser('caption_llm', help='Caption then LLM algorithm')
    caption_parser.add_argument('--model-provider', '-mp', choices=['deepseek', 'vllm', 'ollama'], default='deepseek',
                             help='Model provider to use for script generation')
    caption_parser.add_argument('--model-name', '-m',
                             help='Specific model name for script generation')
    caption_parser.add_argument('--caption-model-provider', '-cmp', choices=['gpt', 'gemini', 'vllm', 'claude', 'ollama'], default='gpt',
                             help='Model provider to use for caption generation')
    caption_parser.add_argument('--caption-model-name', '-cm',
                             help='Specific model name for caption generation')
    caption_parser.add_argument('--max-tokens', '-mt', type=int, default=8192,
                             help='Maximum tokens to generate')
    # Add RAG options to Caption LLM
    caption_parser.add_argument('--use-rag', action='store_true',
                             help='Enable Retrieval-Augmented Generation')
    caption_parser.add_argument('--kb-path', type=str, 
                             help='Path to the knowledge base directory')
    caption_parser.add_argument('--embedding-model', type=str,
                             help='Embedding model to use for vector search')
    caption_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of top results to return from knowledge base')
    caption_parser.add_argument('--prompt-variant', '-pv', 
                             choices=['full', 'no_narrative', 'no_mapping', 'no_grounding'], 
                             default='full',
                             help='Prompt variant to use for ablation study')
    
    # Iterative algorithm subparser
    iterative_parser = subparsers.add_parser('iterative', help='Iterative algorithm')
    iterative_parser.add_argument('--model-provider', '-mp', choices=['gpt', 'vllm', 'ollama'], default='gpt',
                              help='Model provider to use')
    iterative_parser.add_argument('--model-name', '-m',
                              help='Specific model name')
    # Add RAG options to Iterative
    iterative_parser.add_argument('--use-rag', action='store_true',
                              help='Enable Retrieval-Augmented Generation')
    iterative_parser.add_argument('--kb-path', type=str, 
                              help='Path to the knowledge base directory')
    iterative_parser.add_argument('--embedding-model', type=str,
                              help='Embedding model to use for vector search')
    iterative_parser.add_argument('--top-k', type=int, default=5,
                              help='Number of top results to return from knowledge base')
    iterative_parser.add_argument('--prompt-variant', '-pv', 
                              choices=['full', 'no_narrative', 'no_mapping', 'no_grounding'], 
                              default='full',
                              help='Prompt variant to use for ablation study')
    
    # Direct Prompt algorithm subparser
    direct_parser = subparsers.add_parser('direct_prompt', help='Direct VLM prompting with minimal instructions')
    direct_parser.add_argument('--model-provider', '-mp', choices=['gemini_openai', 'gemini', 'gpt', 'vllm', 'ollama', 'claude'], default='gemini_openai',
                            help='Model provider to use')
    direct_parser.add_argument('--model-name', '-m',
                            help='Specific model name')
    direct_parser.add_argument('--max-tokens', '-mt', type=int, default=32768,
                            help='Maximum tokens to generate')
    # Add caption model params needed if text_content is None during preprocessing
    direct_parser.add_argument('--caption-model-provider', '-cmp', choices=['gpt', 'gemini', 'vllm', 'claude', 'ollama'], default='gpt',
                         help='Model provider to use for caption generation (if needed)')
    direct_parser.add_argument('--caption-model-name', '-cm',
                         help='Specific model name for caption generation (if needed)')
    direct_parser.add_argument('--prompt-variant', '-pv', 
                         choices=['full', 'no_narrative', 'no_mapping', 'no_grounding'], 
                         default='full',
                         help='Prompt variant to use for ablation study')
    
    # Parse the arguments
    args = parser.parse_args()
    
    try:
        # Map CLI args to process_ppt API
        model_params = {}
        
        if args.algorithm == 'vlm':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name,
                'max_tokens': args.max_tokens,
                'prompt_variant': args.prompt_variant
            }
            
            # Add RAG parameters if enabled
            if hasattr(args, 'use_rag') and args.use_rag:
                kb_path = args.kb_path or BASE_DIR / "data" / "wiki_knowledge_base"
                model_params.update({
                    'use_rag': True,
                    'kb_path': kb_path,
                    'embedding_model': args.embedding_model,
                    'top_k': args.top_k
                })
                
        elif args.algorithm == 'caption_llm':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name,
                'caption_model_provider': args.caption_model_provider,
                'caption_model_name': args.caption_model_name,
                'max_tokens': args.max_tokens,
                'prompt_variant': args.prompt_variant
            }
            
            # Add RAG parameters if enabled
            if hasattr(args, 'use_rag') and args.use_rag:
                kb_path = args.kb_path or BASE_DIR / "data" / "wiki_knowledge_base"
                model_params.update({
                    'use_rag': True,
                    'kb_path': kb_path,
                    'embedding_model': args.embedding_model,
                    'top_k': args.top_k
                })
                
        elif args.algorithm == 'iterative':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name,
                'prompt_variant': args.prompt_variant
            }
            
            # Add RAG parameters if enabled
            if hasattr(args, 'use_rag') and args.use_rag:
                kb_path = args.kb_path or BASE_DIR / "data" / "wiki_knowledge_base"
                model_params.update({
                    'use_rag': True,
                    'kb_path': kb_path,
                    'embedding_model': args.embedding_model,
                    'top_k': args.top_k
                })
        
        elif args.algorithm == 'direct_prompt':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name,
                'max_tokens': args.max_tokens,
                'caption_model_provider': args.caption_model_provider,
                'caption_model_name': args.caption_model_name,
                'prompt_variant': args.prompt_variant
            }
        
        # Call the unified processing function
        process_ppt(
            input_path=args.input_path,
            algorithm=args.algorithm,
            temperature=args.temperature,
            output_dir=args.output_dir,
            export_excel=args.export_excel,
            model_params=model_params
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 