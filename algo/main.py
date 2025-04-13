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
from storage import ScriptStorage
from algo.dump2excel import scripts_to_excel
from utils import preprocess_ppt, get_images_dir, convert_pdf_to_png, encode_images_to_base64
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
        ppt_path: Path to the PowerPoint file
        algorithm: Algorithm to use ('vlm', 'caption_llm', or 'iterative')
        temperature: Temperature parameter for text generation
        output_dir: Directory to save output files (default: same as PPT directory)
        export_excel: Whether to export scripts to Excel
        model_params: Dictionary of algorithm-specific parameters
            - For 'vlm': model_provider, model_name, max_tokens
            - For 'caption_llm': model_provider, model_name, caption_model_provider, caption_model_name, max_tokens
            - For 'iterative': model_provider, model_name
    
    Returns:
        List of dictionaries containing slide information and generated scripts
        
    Raises:
        ValueError: If algorithm is invalid or ppt_path doesn't exist
        Exception: For any algorithm-specific errors
    """
    # Convert to Path object if string
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input_path}")

    if input_path.suffix == '.pptx':
        text_content, image_urls = preprocess_ppt(input_path)
    elif input_path.suffix == '.pdf':
        text_content = None
        image_path = get_images_dir(input_path)
        convert_pdf_to_png(str(input_path), str(image_path))
        image_urls = encode_images_to_base64(image_path)
    else:
        raise ValueError(f"Invalid input file: {input_path}")
    
    logger.info(f"Running algorithm: {algorithm}")
    
    # Run the selected algorithm
    result = None
    
    try:
        if algorithm == 'vlm':
            # Run VLM algorithm
            result = process_ppt_with_model(
                file_name=input_path.stem,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                temperature=temperature,
                max_tokens=model_params['max_tokens']
            )
        
        elif algorithm == 'caption_llm':
            # Run Caption LLM algorithm
            result = process_ppt_with_caption_llm(
                file_name=input_path.stem,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                caption_model_provider=model_params['caption_model_provider'],
                caption_model_name=model_params['caption_model_name'],
                temperature=temperature,
                max_tokens=model_params['max_tokens']
            )
        
        elif algorithm == 'iterative':
            # Run Iterative algorithm
            result = process_ppt_with_iterative(
                file_name=input_path.stem,
                text_content=text_content,
                image_urls=image_urls,
                model_provider=model_params['model_provider'],
                model_name=model_params['model_name'],
                temperature=temperature
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
    vlm_parser.add_argument('--model-provider', '-mp', choices=['gemini_openai', 'gemini', 'vllm', 'ollama', 'claude'], default='gemini_openai',
                         help='Model provider to use')
    vlm_parser.add_argument('--model-name', '-m',
                         help='Specific model name')
    vlm_parser.add_argument('--max-tokens', '-mt', type=int, default=32768,
                         help='Maximum tokens to generate')
    
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
    
    # Iterative algorithm subparser
    iterative_parser = subparsers.add_parser('iterative', help='Iterative algorithm')
    iterative_parser.add_argument('--model-provider', '-mp', choices=['gpt', 'vllm', 'ollama'], default='gpt',
                              help='Model provider to use')
    iterative_parser.add_argument('--model-name', '-m',
                              help='Specific model name')
    
    # Parse the arguments
    args = parser.parse_args()
    
    try:
        # Map CLI args to process_ppt API
        model_params = {}
        
        if args.algorithm == 'vlm':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name,
                'max_tokens': args.max_tokens
            }
        elif args.algorithm == 'caption_llm':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name,
                'caption_model_provider': args.caption_model_provider,
                'caption_model_name': args.caption_model_name,
                'max_tokens': args.max_tokens
            }
        elif args.algorithm == 'iterative':
            model_params = {
                'model_provider': args.model_provider,
                'model_name': args.model_name
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