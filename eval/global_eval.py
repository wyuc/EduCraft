from config import get_model_config
from models.factory import ModelFactory
from models.base import Message, MessageRole, MessageContent, ContentEntry, ContentType
import re
import numpy as np
from storage import ScriptStorage
from utils import get_images_dir, encode_images_to_base64
import json
from typing import List, Dict, Any, Tuple
import os

# Use Claude for global evaluations since it's better at holistic assessments
MODEL_PROVIDER = "claude"

# Combined evaluation prompt
EVAL_PROMPT = """You are an expert educational content evaluator specializing in lecture quality assessment.
Your task is to evaluate a complete lecture based on its PowerPoint slides and corresponding scripts.

Focus ONLY on global qualities that emerge from the lecture as a whole. Use these dimensions:

1. Narrative Flow: How well does the lecture flow as a single coherent story? Does it have a clear beginning, middle, and end? Do ideas connect naturally across slides?

2. Information Hierarchy: How well is information prioritized and structured across the lecture? Is there a clear understanding of what's most important vs supporting details?

3. Conceptual Integration: How well does the lecture integrate concepts across different slides? Do earlier concepts support later ones? Is knowledge built progressively?

4. Thematic Consistency: How consistently are themes and messaging maintained? Does the lecture maintain a unified voice and perspective?

5. Cross-referencing: How effectively does the lecture reference and connect earlier material with later material? Are connections made between different parts?

After analyzing the entire lecture, provide:
1. A holistic analysis of the lecture script (150-200 words)
2. Score each dimension from 1-5 (where 1 is lowest and 5 is highest)
3. Brief justification for each dimension score

Format your response like this:
```
HOLISTIC ANALYSIS:
[Your analysis of the entire lecture]

DIMENSION SCORES:
1. Narrative Flow: [score]/5
[Brief justification]

2. Information Hierarchy: [score]/5
[Brief justification]

3. Conceptual Integration: [score]/5
[Brief justification]

4. Thematic Consistency: [score]/5
[Brief justification]

5. Cross-referencing: [score]/5
[Brief justification]
```

Focus exclusively on qualities that emerge from the lecture as a whole, not slide-by-slide qualities.
I will now present all slides with their corresponding scripts.
"""

def call_global_eval(scripts: List[str], images: List[str]) -> str:
    """
    Evaluate the global quality of a complete lecture script using Claude.
    
    Args:
        scripts: List of script content for each slide
        images: List of base64-encoded images for each slide
        
    Returns:
        The evaluation response as a string
    """
    model = ModelFactory.get_model(MODEL_PROVIDER)
    
    # Create user message with interleaved slides and scripts
    user_entries = []
    
    # Add interleaved slide images and scripts
    for i, (image, script) in enumerate(zip(images, scripts)):
        # Add the slide image
        user_entries.append(ContentEntry(type=ContentType.IMAGE, data={"url": image}))
        
        # Add the slide script with slide number
        slide_text = f"Slide {i+1} Script:\n{script}\n\n"
        user_entries.append(ContentEntry(type=ContentType.TEXT, data={"text": slide_text}))
    
    # Create the message content
    user_content = MessageContent(entries=user_entries)
    user_message = Message(role=MessageRole.USER, content=user_content)
    system_message = Message(role=MessageRole.SYSTEM, content=EVAL_PROMPT)
    # Make the API call with a higher max_tokens limit for comprehensive evaluation
    response = model.call(messages=[system_message, user_message], max_tokens=8192, temperature=0.2)
    return response.content

def parse_global_eval_scores(response: str) -> Dict[str, Any]:
    """
    Parse the global evaluation scores from the Claude response.
    
    Args:
        response: The raw response string from Claude
        
    Returns:
        Dictionary with parsed scores and analysis
    """
    result = {
        "holistic_analysis": "",
        "dimension_scores": {
            "narrative_flow": 0,
            "information_hierarchy": 0,
            "conceptual_integration": 0,
            "thematic_consistency": 0,
            "cross_referencing": 0
        },
        "raw_response": response
    }
    
    # Extract holistic analysis
    holistic_match = re.search(r'HOLISTIC ANALYSIS:\s*(.*?)(?=DIMENSION SCORES:|$)', response, re.DOTALL)
    if holistic_match:
        result["holistic_analysis"] = holistic_match.group(1).strip()
    
    # Extract dimension scores
    dimensions = [
        ("narrative_flow", r'1\.\s*Narrative Flow:\s*(\d+)/5'),
        ("information_hierarchy", r'2\.\s*Information Hierarchy:\s*(\d+)/5'),
        ("conceptual_integration", r'3\.\s*Conceptual Integration:\s*(\d+)/5'),
        ("thematic_consistency", r'4\.\s*Thematic Consistency:\s*(\d+)/5'),
        ("cross_referencing", r'5\.\s*Cross-referencing:\s*(\d+)/5')
    ]
    
    for dim_key, pattern in dimensions:
        match = re.search(pattern, response)
        if match:
            result["dimension_scores"][dim_key] = int(match.group(1))
    
    # Calculate overall score as average of dimension scores
    if any(result["dimension_scores"].values()):
        result["overall_score"] = np.mean(list(result["dimension_scores"].values()))
    else:
        result["overall_score"] = 0
    
    return result

def evaluate_global_quality(ppt_path: str, algorithm: str, model_provider: str = None):
    """
    Evaluate the global quality of a lecture script generated for a PowerPoint presentation.
    
    Args:
        ppt_path: Path to the PowerPoint file
        algorithm: Algorithm used to generate scripts (e.g., 'vlm', 'caption_llm')
        model_provider: Provider of the model (e.g., 'gemini', 'gpt', 'claude')
        
    Returns:
        Dictionary containing evaluation results
    """
    # Initialize storage
    storage = ScriptStorage()
    
    # Get images directory
    images_dir = get_images_dir(ppt_path)
    if not images_dir.exists():
        raise ValueError(f"Images directory not found for {ppt_path}")
    
    # Extract just the filename without directory path and extension
    file_name = os.path.splitext(os.path.basename(ppt_path))[0]
    
    # Get the latest completed task
    latest_task = storage.get_latest_completed_task(file_name, algorithm, model_provider)
    if not latest_task:
        raise ValueError(f"No completed task found for {ppt_path} using algorithm {algorithm}, model provider {model_provider}")
    
    # Get all slides
    all_slides = storage.get_all_slides(latest_task['task_id'])
    if not all_slides:
        raise ValueError(f"No scripts found for {ppt_path} using algorithm {algorithm}, model provider {model_provider}")
    
    scripts = [slide['content'] for slide in all_slides]
    
    # Encode images to base64
    images = encode_images_to_base64(images_dir)

    if len(scripts) != len(images):
        raise ValueError(f"Number of scripts and images do not match for {ppt_path} using algorithm {algorithm}, model provider {model_provider}")
   
    print(f"Evaluating global quality of {len(scripts)} slides...")
    
    # Call global evaluation
    response = call_global_eval(scripts, images)
    
    # Parse scores
    evaluation_results = parse_global_eval_scores(response)
    
    # Add metadata
    evaluation_results["metadata"] = {
        "ppt_path": ppt_path,
        "algorithm": algorithm,
        "model_provider": model_provider if model_provider else MODEL_PROVIDER,
        "slides_count": len(scripts),
        "task_id": latest_task['task_id']
    }
    
    # Print results
    dim_scores = evaluation_results["dimension_scores"]
    
    print("\n===== Global Quality Evaluation Results =====")
    print(f"PPT: {ppt_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Model provider: {model_provider if model_provider else MODEL_PROVIDER}")
    print(f"Slides evaluated: {len(scripts)}")
    
    print("\nDimension Scores:")
    print(f"1. Narrative Flow: {dim_scores['narrative_flow']}/5")
    print(f"2. Information Hierarchy: {dim_scores['information_hierarchy']}/5")
    print(f"3. Conceptual Integration: {dim_scores['conceptual_integration']}/5")
    print(f"4. Thematic Consistency: {dim_scores['thematic_consistency']}/5")
    print(f"5. Cross-referencing: {dim_scores['cross_referencing']}/5")
    
    # Calculate average dimension score (overall score)
    overall_score = round(evaluation_results["overall_score"], 2)
    print(f"\nOverall Global Quality Score: {overall_score}/5")
    print(f"(Calculated as average of dimension scores)")
    
    # Store rounded overall score
    evaluation_results["overall_score"] = overall_score
    
    return evaluation_results

def save_evaluation_results(results: Dict[str, Any], output_path: str = None):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results (optional)
    """
    if not output_path:
        # Generate default filename based on metadata
        metadata = results.get("metadata", {})
        ppt_name = metadata.get("ppt_path", "unknown").split("/")[-1].split(".")[0]
        algo = metadata.get("algorithm", "unknown")
        provider = metadata.get("model_provider", "unknown")
        task_id = metadata.get("task_id", "unknown")
        
        output_path = f"global_eval_{ppt_name}_{algo}_{provider}_{task_id}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate global quality of lecture scripts")
    parser.add_argument("ppt_path", help="Path to the PowerPoint file")
    parser.add_argument("--algorithm", "-a", choices=["vlm", "caption_llm", "iterative"], default="vlm",
                      help="Algorithm used to generate scripts")
    parser.add_argument("--model-provider", "-mp", default=None,
                      help="Provider of the model (e.g., 'gemini', 'gpt', 'claude')")
    parser.add_argument("--output", "-o", default=None,
                      help="Path to save evaluation results (JSON format)")
    
    args = parser.parse_args()
    
    # Run the evaluation
    results = evaluate_global_quality(args.ppt_path, args.algorithm, args.model_provider)
    
    # Save results if requested
    save_evaluation_results(results, args.output) 