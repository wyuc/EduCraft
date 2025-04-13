from algo.main import process_ppt
import os
import argparse
import json
from storage import ScriptStorage
from pathlib import Path
from config import TEST_SETTING
from eval.leceval import evaluate_ppt as evaluate_slide_quality
from eval.global_eval import evaluate_global_quality
from tqdm import tqdm


def generate(input_path, force=False):
    """
    Generate lecture scripts using the specified algorithm and model.
    
    Args:
        input_path: Path to the input file or directory
        force: Force regeneration even if results already exist
    """
    storage = ScriptStorage()
    
    # Determine if we're processing a single file or a directory
    if os.path.isdir(input_path):
        file_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                     if f.endswith(('.pptx', '.pdf'))]
    else:
        file_paths = [input_path]
    
    if not file_paths:
        print(f"No valid presentation files found in {input_path}")
        return
    
    # Determine which algorithms and model providers to use
    algos_to_use = TEST_SETTING.keys()
    
    for algo in algos_to_use:
        settings = TEST_SETTING[algo]
        
        for setting in settings:
            provider = setting['model_provider']
            if "model_name" in setting:
                model_name = setting['model_name']
            else:
                model_name = None
            if provider == 'vllm':
                input(f"Please deploy the appropriate VLLM model for {algo} algorithm and press Enter to continue...")
            
            for file_path in tqdm(file_paths, desc=f"Processing files with {algo}/{provider}"):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                print(f"Processing file: {file_name}")
                
                # Check if we already have results for this combination
                latest_task = storage.get_latest_completed_task(file_name, algo, provider)
                
                if latest_task and not force:
                    print(f"Skipping {file_name} with {algo} and {provider} because it already exists")
                    print(f"  (Use --force to regenerate)")
                    continue
                
                print(f"Generating script for {file_name} with {algo} and {provider}...")
                
                try:
                    # Create model parameters based on algorithm
                    if algo == 'vlm':
                        model_params = {
                            "model_provider": provider,
                            "model_name": model_name,
                            "max_tokens": 8192
                        }
                    elif algo == 'caption_llm':
                        model_params = {
                            "model_provider": provider,
                            "model_name": model_name,
                            "caption_model_provider": "gpt",
                            "caption_model_name": None,
                            "max_tokens": 8192
                        }
                    elif algo == 'iterative':
                        model_params = {
                            "model_provider": provider,
                            "model_name": model_name,
                        }
                    
                    # Process the presentation
                    process_ppt(
                        input_path=file_path,
                        algorithm=algo,
                        model_params=model_params
                    )
                    
                    print(f"✓ Completed {file_name} with {algo} and {provider}")
                    
                except Exception as e:
                    print(f"✗ Error processing {file_name} with {algo} and {provider}: {str(e)}")
                    raise e


def evaluate_all(input_path):
    """
    Evaluate all generated scripts for the input path across all algorithms and model providers.
    
    Args:
        input_path: Path to the input file or directory
    """
    storage = ScriptStorage()
    
    # Create output directory and prepare output path
    output_dir = Path("data/eval_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = Path(input_path).stem + "_" + Path().cwd().name
    output_path = output_dir / f"evaluation_results_{timestamp}.json"
    
    # Try to load existing results if available
    eval_results = {}
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            print(f"Loaded existing evaluation results from {output_path}")
        except json.JSONDecodeError:
            print(f"Error loading existing results, starting fresh")
    
    # Determine if we're processing a single file or a directory
    if os.path.isdir(input_path):
        file_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                     if f.endswith(('.pptx', '.pdf'))]
    else:
        file_paths = [input_path]
    
    if not file_paths:
        print(f"No valid presentation files found in {input_path}")
        return
    
    # Determine which algorithms and model providers to use
    algos_to_use = TEST_SETTING.keys()
    
    print("\n===== Starting Evaluation =====")
    
    for algo in algos_to_use:
        if algo not in eval_results:
            eval_results[algo] = {}
        
        settings = TEST_SETTING[algo]
        
        for setting in settings:
            provider = setting['model_provider']
            
            if provider not in eval_results[algo]:
                eval_results[algo][provider] = {}
            
            for file_path in tqdm(file_paths, desc=f"Evaluating files with {algo}/{provider}"):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Skip if already evaluated successfully
                if file_name in eval_results[algo][provider] and "error" not in eval_results[algo][provider][file_name]:
                    print(f"Skipping evaluation for {file_name} with {algo} and {provider} (already evaluated)")
                    continue
                
                # Check if we have results for this combination
                latest_task = storage.get_latest_completed_task(file_name, algo, provider)
                
                if not latest_task:
                    print(f"No completed task found for {file_name} with {algo} and {provider}, skipping evaluation")
                    eval_results[algo][provider][file_name] = {"error": "No completed task found"}
                    
                    # Save incremental results
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(eval_results, f, ensure_ascii=False, indent=2)
                    print(f"Progress saved to: {output_path}")
                    
                    continue
                
                print(f"Evaluating {file_name} with {algo} and {provider}...")
                
                try:
                    # Evaluate slide quality
                    print(f"Evaluating slide quality for {file_name} with {algo} and {provider}...")
                    slide_eval = evaluate_slide_quality(file_path, algo, provider)
                    
                    # Evaluate global quality
                    print(f"Evaluating global quality for {file_name} with {algo} and {provider}...")
                    global_eval = evaluate_global_quality(file_path, algo, provider)
                    
                    # Store results
                    eval_results[algo][provider][file_name] = {
                        "slide_evaluation": slide_eval,
                        "global_evaluation": global_eval,
                        "combined_score": round((slide_eval["overall_score"] + global_eval["overall_score"]) / 2, 2)
                    }
                    
                    print(f"✓ Evaluation completed for {file_name} with {algo} and {provider}")
                    print(f"  Slide score: {slide_eval['overall_score']}/5")
                    print(f"  Global score: {global_eval['overall_score']}/5")
                    print(f"  Combined score: {eval_results[algo][provider][file_name]['combined_score']}/5")
                    
                except Exception as e:
                    print(f"✗ Error evaluating {file_name} with {algo} and {provider}: {str(e)}")
                    eval_results[algo][provider][file_name] = {"error": str(e)}
                
                # Save incremental results after each file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(eval_results, f, ensure_ascii=False, indent=2)
                print(f"Progress saved to: {output_path}")
    
    print(f"\nAll evaluation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate lecture scripts from presentations')
    
    # Input path - file or directory
    parser.add_argument('input', help='Path to a presentation file or directory containing presentations')
    # Force regeneration
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force regeneration even if results already exist')
    # Skip evaluation
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation after generation')
    # Only evaluate (no generation)
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing scripts without generation')
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_all(args.input)
    else:
        generate(args.input, args.force)
        if not args.no_eval:
            evaluate_all(args.input)


if __name__ == "__main__":
    main()
