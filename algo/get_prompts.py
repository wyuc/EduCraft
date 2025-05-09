import os
from pathlib import Path
from config import BASE_DIR

PROMPTS_DIR = Path(__file__).parent / 'prompts'

def _get_prompt(file_name: str) -> str:
    prompt_file = PROMPTS_DIR / file_name
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
        return prompt
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file}")
        # Return a minimal fallback prompt
        return "Generate lecture notes for the slides. Use ===SLIDE X=== format."

def write_vlm(variant: str = 'full'):
    """Loads the specified variant of the VLM prompt."""
    if variant == 'full':
        return _get_prompt("write_vlm.txt")
    elif variant == 'no_narrative':
        # Assumes you will create this file
        return _get_prompt("write_vlm_no_narrative.txt")
    elif variant == 'no_mapping':
        # Assumes you will create this file
        return _get_prompt("write_vlm_no_mapping.txt")
    elif variant == 'no_grounding':
        # Assumes you will create this file
        return _get_prompt("write_vlm_no_grounding.txt")
    else:
        print(f"Warning: Unknown prompt variant '{variant}'. Loading full prompt.")
        return _get_prompt("write_vlm.txt")

def write_caption_llm():
    return _get_prompt("write_caption_llm.txt")

def caption() -> str:
    return _get_prompt("caption.txt")
