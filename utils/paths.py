"""
Path utilities for handling temporary directories and file locations.
"""

from pathlib import Path
from config import BASE_DIR

def get_temp_dir(input_path):
    """
    Get the temporary directory path for a given input file.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        Path: Path to the temporary directory
    """
    return BASE_DIR / 'buffer' / Path(input_path).stem

def get_images_dir(input_path):
    """
    Get the images directory path for a given input file.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        Path: Path to the images directory
    """
    return get_temp_dir(input_path) / 'images' 