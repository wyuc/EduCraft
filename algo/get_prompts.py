import os
from pathlib import Path
from config import BASE_DIR

PROMPTS_DIR = Path(__file__).parent / 'prompts'

def _get_prompt(file_name: str) -> str:
    prompt_file = PROMPTS_DIR / file_name
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    return prompt

def write_vlm():
    return _get_prompt("write_vlm.txt")

def write_caption_llm():
    return _get_prompt("write_caption_llm.txt")

def caption() -> str:
    return _get_prompt("caption.txt")
