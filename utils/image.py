"""
Image processing utilities for handling base64 encoding and PDF to image conversion.
"""

import os
import base64
import re
import fitz
import logging

# Configure logging
logger = logging.getLogger(__name__)

def encode_image_to_base64(img_path):
    """
    Encode an image file to base64 data URL.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        str: Base64 encoded data URL for the image
    """
    try:
        # Get file extension and corresponding MIME type
        extension = os.path.splitext(img_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(extension, 'image/jpeg')  # default to jpeg if unknown
        
        with open(img_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
            
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        return None

def encode_images_to_base64(img_dir):
    """
    Encode all images in a directory to base64 data URLs.
    
    Args:
        img_dir: Directory containing image files
        
    Returns:
        list: List of base64 encoded data URLs for images
    """
    image_files = os.listdir(img_dir)
    # Filter for files that match the pattern of "1.png", "2.png", etc.
    image_files = [f for f in image_files if re.match(r'^\d+\.png$', f)]
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    image_urls = []
    
    for filename in image_files:
        img_path = os.path.join(img_dir, filename)
        image_url = encode_image_to_base64(img_path)
        if image_url:
            image_urls.append(image_url)
    
    return image_urls

def convert_pdf_to_png(input_file, output_dir):
    """
    Convert each page of a PDF file to PNG images.
    
    Args:
        input_file: Path to the PDF file
        output_dir: Directory to save the PNG images
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    doc = fitz.open(input_file)

    # Iterate through each page of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the current page
        pix = page.get_pixmap()  # Render page to an image
        output_path = os.path.join(output_dir, f'{page_num + 1}.png')  # Define output path for the image
        pix.save(output_path)  # Save the image

    logger.info(f"Conversion completed. Images are saved in '{output_dir}'.") 