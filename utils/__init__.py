"""
Utilities module for lecgen.

This module provides various utility functions for file handling, presentation processing,
image manipulation, text extraction, and more.
"""

# Re-export all functions from the modules
from utils.paths import get_temp_dir, get_images_dir
from utils.ppt import preprocess_ppt, extract_text_from_ppt, pptx_to_pdf
from utils.image import (
    encode_image_to_base64, 
    encode_images_to_base64, 
    convert_pdf_to_png
)
from utils.text import extract_json
from utils.slides import generate_captions_for_slides, extract_and_save_slides
from utils.rag import (
    RAGUtils,
    DEFAULT_EMBEDDING_MODEL,
    build_vector_index,
    search_knowledge_base
)

# Make all these functions available at the top level
__all__ = [
    # Path utilities
    'get_temp_dir',
    'get_images_dir',
    
    # PowerPoint utilities
    'preprocess_ppt',
    'extract_text_from_ppt',
    'pptx_to_pdf',
    
    # Image utilities
    'encode_image_to_base64',
    'encode_images_to_base64',
    'convert_pdf_to_png',
    
    # Text utilities
    'extract_json',
    
    # Slide utilities
    'extract_and_save_slides',
    'generate_captions_for_slides',
    
    # RAG utilities
    'RAGUtils',
    'DEFAULT_EMBEDDING_MODEL',
    'build_vector_index',
    'search_knowledge_base',
] 