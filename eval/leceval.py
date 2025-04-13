from config import get_model_config, LEC_EVAL_MODEL
from models.factory import ModelFactory
from models.base import Message, MessageRole, MessageContent, ContentEntry, ContentType
import re
import numpy as np
from storage import ScriptStorage
from utils import get_images_dir, encode_images_to_base64
import os

MODEL_PROVIDER = "vllm"

EVAL_PROMPT = """Instructions:

You are provided with a segment of a lecture slide and its corresponding transcript. Evaluate the transcript based on the following criteria:

1. Faithfulness to the Slide: How accurately does the transcript represent the information on the provided PowerPoint slide?
2. Clarity of Language: How clear and understandable is the language used in the transcript?
3. Structure and Organization: How well is the transcript structured and organized?
4. Inspirational Value: How inspiring and engaging is the transcript?

Rate each criterion on a scale from 1 to 5, where 1 is the lowest and 5 is the highest. Provide your ratings in the format: Faithfulness, Clarity, Structure, Inspirational (e.g., 3, 4, 2, 5).

Transcript:
{script}

Output:
"""

def call_leceval(script: str, image: str):
    """
    Evaluate a single slide's script using the LecEval model.
    
    Args:
        script: The transcript/script for the slide
        image: The slide image
        
    Returns:
        The raw model response (string containing scores)
    """
    model = ModelFactory.get_model(MODEL_PROVIDER)
    
    # Create a message with both image and text content
    content = MessageContent(entries=[
        ContentEntry(type=ContentType.IMAGE, data={"url": image}),
        ContentEntry(type=ContentType.TEXT, data={"text": EVAL_PROMPT.format(script=script)})
    ])
    
    msgs = [Message(role=MessageRole.USER, content=content)]
    response = model.call(messages=msgs, model=LEC_EVAL_MODEL)
    return response.content

def parse_leceval_scores(response: str):
    """
    Parse the scores from the LecEval model response.
    
    Args:
        response: The raw response string from the model
        
    Returns:
        A list of integers representing the scores [faithfulness, clarity, structure, inspirational]
    """
    # Look for patterns like "3, 4, 2, 5" in the response
    score_pattern = r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)'
    match = re.search(score_pattern, response)
    
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    else:
        # Fallback: look for individual numbers
        scores = []
        for i in range(1, 5):
            pattern = rf'(\d+)\s*/\s*5|(\d+)/5|{i}\.\s*\w+.*?:\s*(\d+)'
            score_match = re.search(pattern, response)
            if score_match:
                # Use the first non-None group
                for g in score_match.groups():
                    if g is not None:
                        scores.append(int(g))
                        break
            else:
                # Default score if not found
                scores.append(3)
        
        return scores if len(scores) == 4 else [3, 3, 3, 3]

def evaluate_ppt(ppt_path: str, algorithm: str, model_provider: str = None):
    """
    Evaluate all slides in a PowerPoint presentation using the specified algorithm.
    
    Args:
        ppt_path: Path to the PowerPoint file
        algorithm: Algorithm used to generate scripts (e.g., 'vlm', 'caption_llm')
        model_provider: Provider of the model (e.g., 'gemini', 'gpt', 'claude')
        
    Returns:
        A dictionary containing:
        - average_scores: Average scores across all slides [faithfulness, clarity, structure, inspirational]
        - per_slide_scores: List of scores for each slide
        - overall_score: Average of all dimensions across all slides
    """
    # Initialize storage and get the latest scripts
    storage = ScriptStorage()

    images_dir = get_images_dir(ppt_path)

    if not images_dir.exists():
        raise ValueError(f"Images directory not found for {ppt_path}")

    # Extract just the filename without directory path and extension
    file_name = os.path.splitext(os.path.basename(ppt_path))[0]
    
    latest_task = storage.get_latest_completed_task(file_name, algorithm, model_provider)
    if not latest_task:
        raise ValueError(f"No completed task found for {ppt_path} using algorithm {algorithm}, model provider {model_provider}")
    
    # Get the scripts for this presentation
    all_slides = storage.get_all_slides(latest_task['task_id'])
    
    if not all_slides:
        raise ValueError(f"No scripts found for {ppt_path} using algorithm {algorithm}, model provider {model_provider}")

    scripts = [slide['content'] for slide in all_slides]
    
    # Initialize lists to store scores
    all_scores = []
    slide_scores = []
    
    images = encode_images_to_base64(images_dir)
    # Process each slide
    for slide_idx, script in enumerate(scripts):
        # Skip empty scripts
        if not script:
            continue
            
        try:
            image = images[slide_idx]
            # Evaluate the script
            response = call_leceval(script, image)
            scores = parse_leceval_scores(response)
            
            print(f"Slide {slide_idx+1} scores: {scores}")
            all_scores.append(scores)
            slide_scores.append({"slide": slide_idx+1, "scores": scores})
        except Exception as e:
            print(f"Error evaluating slide {slide_idx+1}: {str(e)}")
    
    # Calculate average scores
    if all_scores:
        all_scores_array = np.array(all_scores)
        average_scores = all_scores_array.mean(axis=0).tolist()
        overall_score = all_scores_array.mean().item()
        
        # Round to 2 decimal places
        average_scores = [round(score, 2) for score in average_scores]
        overall_score = round(overall_score, 2)
        
        criteria = ["Faithfulness", "Clarity", "Structure", "Inspirational"]
        
        print("\n===== Evaluation Results =====")
        print(f"PPT: {ppt_path}")
        print(f"Algorithm: {algorithm}")
        print(f"Model provider: {model_provider}")
        print(f"Slides evaluated: {len(all_scores)}")
        print("\nAverage scores:")
        for i, criterion in enumerate(criteria):
            print(f"{criterion}: {average_scores[i]}/5")
        print(f"\nOverall score: {overall_score}/5")
        
        return {
            "average_scores": average_scores,
            "per_slide_scores": slide_scores,
            "overall_score": overall_score
        }
    else:
        return {
            "average_scores": [0, 0, 0, 0],
            "per_slide_scores": [],
            "overall_score": 0
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate lecture scripts for PowerPoint slides")
    parser.add_argument("ppt_path", help="Path to the PowerPoint file")
    parser.add_argument("--algorithm", "-a", choices=["vlm", "caption_llm", "iterative"], default="vlm",
                        help="Algorithm used to generate scripts")
    parser.add_argument("--model-provider", "-mp", default=None,
                        help="Provider of the model (e.g., 'gemini', 'gpt', 'claude')")
    
    args = parser.parse_args()
    
    # Run the evaluation
    evaluate_ppt(args.ppt_path, args.algorithm, args.model_provider)
