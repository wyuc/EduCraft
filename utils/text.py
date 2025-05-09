"""
Text processing utilities for handling JSON extraction and formatting.
"""

import re
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

def extract_json(text):
    """
    Extracts and parses JSON from a text string that may contain other content.
    
    This function attempts multiple strategies to extract valid JSON:
    1. First tries to find complete JSON structures using regex patterns
    2. Falls back to finding outermost { } or [ ] if regex fails
    3. Cleans the text to fix common JSON formatting issues
    4. Attempts multiple parsing methods with increasing aggressiveness
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Parsed JSON as a dictionary or list
        
    Raises:
        ValueError: If JSON cannot be parsed after all attempts
    """
    # Clean up the text first to handle common issues
    # Remove markdown code block markers if present
    text = re.sub(r'```(?:json)?|```', '', text)
    
    # Strategy 1: Use regex to find JSON structures
    # Try to find a complete JSON object with balanced braces
    json_pattern = r'(\{(?:[^{}]|(?R))*\})'
    array_pattern = r'(\[(?:[^\[\]]|(?R))*\])'
    
    # Since Python's re doesn't support recursion (?R), we'll use a simpler approach
    # Try to match the outermost JSON object or array
    obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    
    json_candidates = []
    if obj_match:
        json_candidates.append(obj_match.group(0))
    if arr_match:
        json_candidates.append(arr_match.group(0))
    
    # If we found potential JSON structures via regex
    for json_str in json_candidates:
        # Try to parse it directly first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If direct parsing fails, we'll continue to more aggressive methods
            pass
    
    # Strategy 2: Fall back to the original method if regex didn't work
    # Find leftmost { or [
    left_curly = text.find('{')
    left_bracket = text.find('[')
    
    # Determine which comes first (if both exist)
    if left_curly != -1 and left_bracket != -1:
        left_pos = min(left_curly, left_bracket)
    elif left_curly != -1:
        left_pos = left_curly
    elif left_bracket != -1:
        left_pos = left_bracket
    else:
        raise ValueError("No JSON structure found in text")
    
    # Find rightmost } or ]
    right_curly = text.rfind('}')
    right_bracket = text.rfind(']')
    
    # Determine which comes last (if both exist)
    if right_curly != -1 and right_bracket != -1:
        right_pos = max(right_curly, right_bracket)
    elif right_curly != -1:
        right_pos = right_curly
    elif right_bracket != -1:
        right_pos = right_bracket
    else:
        raise ValueError("No JSON structure found in text")
    
    # Extract the JSON substring
    json_str = text[left_pos:right_pos+1]
    
    # Strategy 3: Clean up common issues before parsing
    # Fix line breaks after commas which are common in LLM outputs
    json_str = re.sub(r',\s*\n\s*', ', ', json_str)
    
    # Fix missing quotes around keys
    json_str = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_str)
    
    # Remove any trailing commas before closing brackets or braces
    json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
    
    # Strategy 4: Try multiple parsing methods with increasing aggressiveness
    # First attempt: standard json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Standard JSON parsing failed: {e}")
        
        # Second attempt: Try with ast.literal_eval for more forgiving parsing
        try:
            import ast
            return ast.literal_eval(json_str)
        except (SyntaxError, ValueError) as e:
            logger.error(f"AST literal_eval failed: {e}")
            raise ValueError(f"Failed to parse JSON after multiple attempts. Original text: {text[:100]}...") 