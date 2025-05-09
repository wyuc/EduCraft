from algo.main import process_ppt
import os
import argparse
import json
from storage import ScriptStorage
from pathlib import Path
from config import TEST_SETTING, BASE_DIR
from eval.leceval import evaluate_ppt as evaluate_slide_quality
from eval.global_eval import evaluate_global_quality
from tqdm import tqdm
import logging # Import logging

# Get logger for this module
logger = logging.getLogger(__name__)

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
    
    # Expand TEST_SETTING to include effective algorithm names for VLM variants
    expanded_configs = []
    for base_algo, settings_list in TEST_SETTING.items():
        if base_algo == 'vlm':
            for setting in settings_list:
                prompt_variant = setting.get('prompt_variant', 'full')
                is_rag = setting.get('use_rag', False)

                effective_algo_name = "vlm"
                if is_rag:
                    effective_algo_name += "_rag"
                if prompt_variant != 'full':
                    effective_algo_name += f"_{prompt_variant}"

                expanded_configs.append({
                    "effective_algo": effective_algo_name,
                    "setting": setting,
                    "base_algo": base_algo # Keep track of original algo type
                })
        else:
            # For other algorithms, the effective name is the base name
            for setting in settings_list:
                expanded_configs.append({
                    "effective_algo": base_algo,
                    "setting": setting,
                    "base_algo": base_algo
                })

    # Process each expanded configuration
    for config in expanded_configs:
        effective_algo = config["effective_algo"]
        setting = config["setting"]
        base_algo = config["base_algo"]
        provider = setting['model_provider']

        if "model_name" in setting:
            model_name = setting['model_name']
        else:
            model_name = None

        if provider == 'vllm':
             # Ensure VLLM model is ready if needed
             input(f"Please deploy the appropriate VLLM model for {effective_algo} ({model_name or 'default'}) and press Enter to continue...")

        for file_path in tqdm(file_paths, desc=f"Processing files with {effective_algo}/{provider}"):
            file_name = Path(file_path).stem
            print(f"\nProcessing file: {file_name} ({effective_algo}/{provider})")

            # Check storage using the effective algorithm name
            algo_storage_name = effective_algo
            latest_task = storage.get_latest_completed_task(file_name, algo_storage_name, provider)

            if latest_task and not force:
                print(f"Skipping {file_name} with {effective_algo} and {provider} because it already exists (Task ID: {latest_task['task_id']})")
                print(f"  (Use --force to regenerate)")
                continue

            print(f"Generating script for {file_name} with {effective_algo} and {provider}...")

            try:
                # Create model parameters based on the original base algorithm type
                model_params = {}

                if base_algo == 'vlm':
                    model_params = {
                        "model_provider": provider,
                        "model_name": model_name,
                        "max_tokens": setting.get('max_tokens', 8192),
                        "prompt_variant": setting.get('prompt_variant', 'full')
                    }
                    if setting.get('use_rag', False):
                        kb_path = setting.get('kb_path', str(BASE_DIR / "data" / "wiki_knowledge_base"))
                        model_params.update({
                            "use_rag": True,
                            "kb_path": kb_path,
                            "embedding_model": setting.get('embedding_model'),
                            "top_k": setting.get('top_k', 5)
                        })
                        print(f"  Using RAG with knowledge base at {kb_path}")

                elif base_algo == 'caption_llm':
                    model_params = {
                        "model_provider": provider,
                        "model_name": model_name,
                        "caption_model_provider": setting.get("caption_model_provider", "gpt"),
                        "caption_model_name": setting.get("caption_model_name"),
                        "max_tokens": setting.get('max_tokens', 8192)
                    }
                    if setting.get('use_rag', False):
                        kb_path = setting.get('kb_path', str(BASE_DIR / "data" / "wiki_knowledge_base"))
                        model_params.update({
                            "use_rag": True,
                            "kb_path": kb_path,
                            "embedding_model": setting.get('embedding_model'),
                            "top_k": setting.get('top_k', 5)
                        })

                elif base_algo == 'iterative':
                    model_params = {
                        "model_provider": provider,
                        "model_name": model_name,
                        # Assuming iterative might need caption models too
                        "caption_model_provider": setting.get("caption_model_provider", "gpt"),
                        "caption_model_name": setting.get("caption_model_name"),
                    }
                    if setting.get('use_rag', False):
                        kb_path = setting.get('kb_path', str(BASE_DIR / "data" / "wiki_knowledge_base"))
                        model_params.update({
                            "use_rag": True,
                            "kb_path": kb_path,
                            "embedding_model": setting.get('embedding_model'),
                            "top_k": setting.get('top_k', 5)
                        })

                elif base_algo == 'direct_prompt':
                    model_params = {
                        "model_provider": provider,
                        "model_name": model_name,
                        "max_tokens": setting.get('max_tokens', 8192),
                        "caption_model_provider": setting.get("caption_model_provider", "gpt"),
                        "caption_model_name": setting.get("caption_model_name")
                    }
                else:
                    logger.warning(f"Unknown base algorithm type '{base_algo}' when creating model params for effective algo '{effective_algo}'. Using basic params.")
                    model_params = {
                        "model_provider": provider,
                        "model_name": model_name
                    }

                # Call process_ppt with the *base* algorithm name for correct function dispatch
                # The task will be stored under the *effective* name (handled in algo/vlm.py)
                process_ppt(
                    input_path=file_path,
                    algorithm=base_algo, # Use base name for function dispatch
                    model_params=model_params,
                    # temperature=setting.get('temperature', 0.7) # Pass temp if needed
                )

                print(f"✓ Completed {file_name} with {effective_algo} and {provider}")

            except Exception as e:
                print(f"✗ Error processing {file_name} with {effective_algo} and {provider}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue to next file/config


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
    
    # Determine which effective algorithms and model providers to use
    # Regenerate the expanded list of specific algorithm names + settings
    expanded_configs_for_eval = []
    for base_algo, settings_list in TEST_SETTING.items():
        if base_algo == 'vlm':
            for setting in settings_list:
                prompt_variant = setting.get('prompt_variant', 'full')
                is_rag = setting.get('use_rag', False)
                effective_algo_name = "vlm"
                if is_rag:
                    effective_algo_name += "_rag"
                if prompt_variant != 'full':
                    effective_algo_name += f"_{prompt_variant}"
                expanded_configs_for_eval.append({"effective_algo": effective_algo_name, "setting": setting})
        else:
            for setting in settings_list:
                expanded_configs_for_eval.append({"effective_algo": base_algo, "setting": setting})

    print("\n===== Starting Evaluation =====")

    # Iterate through the specific algorithm configurations
    for config in expanded_configs_for_eval:
        effective_algo = config["effective_algo"]
        setting = config["setting"]
        provider = setting['model_provider']

        # Ensure keys exist in eval_results
        if effective_algo not in eval_results:
            eval_results[effective_algo] = {}
        if provider not in eval_results[effective_algo]:
            eval_results[effective_algo][provider] = {}

        for file_path in tqdm(file_paths, desc=f"Evaluating files with {effective_algo}/{provider}"):
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            # Skip if already evaluated successfully
            if file_name in eval_results[effective_algo][provider] and "error" not in eval_results[effective_algo][provider][file_name]:
                print(f"Skipping evaluation for {file_name} with {effective_algo} and {provider} (already evaluated)")
                continue

            # Get the latest completed task using the effective algorithm name
            latest_task = storage.get_latest_completed_task(file_name, effective_algo, provider)

            if not latest_task:
                print(f"No completed task found for {file_name} with {effective_algo} and {provider}, skipping evaluation")
                eval_results[effective_algo][provider][file_name] = {"error": "No completed task found"}

                # Save incremental results
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(eval_results, f, ensure_ascii=False, indent=2)
                print(f"Progress saved to: {output_path}")

                continue

            print(f"\nEvaluating {file_name} with {effective_algo} and {provider} (Task ID: {latest_task['task_id']})...")

            try:
                # Evaluate slide quality using effective_algo name
                print(f"Evaluating slide quality for {file_name} with {effective_algo} and {provider}...")
                slide_eval = evaluate_slide_quality(file_path, effective_algo, provider)

                # Evaluate global quality using effective_algo name
                print(f"Evaluating global quality for {file_name} with {effective_algo} and {provider}...")
                global_eval = evaluate_global_quality(file_path, effective_algo, provider)

                # Store results under effective_algo name
                eval_results[effective_algo][provider][file_name] = {
                    "slide_evaluation": slide_eval,
                    "global_evaluation": global_eval,
                    "combined_score": round((slide_eval["overall_score"] + global_eval["overall_score"]) / 2, 2)
                }

                print(f"✓ Evaluation completed for {file_name} with {effective_algo} and {provider}")
                print(f"  Slide score: {slide_eval['overall_score']}/5")
                print(f"  Global score: {global_eval['overall_score']}/5")
                print(f"  Combined score: {eval_results[effective_algo][provider][file_name]['combined_score']}/5")

            except Exception as e:
                print(f"✗ Error evaluating {file_name} with {effective_algo} and {provider}: {str(e)}")
                import traceback
                traceback.print_exc()
                eval_results[effective_algo][provider][file_name] = {"error": str(e)}

            # Save incremental results after each file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
            print(f"Progress saved to: {output_path}")

    print(f"\nAll evaluation results saved to: {output_path}")
    return output_path


def generate_scores_report(eval_results_path):
    """
    Generate a report of model scores for each algorithm based on evaluation results.
    
    Args:
        eval_results_path: Path to the evaluation results JSON file
    """
    try:
        # Load evaluation results
        with open(eval_results_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        
        # Output directory
        output_dir = Path("data/eval_results")
        output_dir.mkdir(exist_ok=True)

        # Determine timestamp from input path
        timestamp = Path(eval_results_path).stem.replace("evaluation_results_", "")
        
        # Prepare scores report structure
        scores_report = {}
        all_individual_scores = {} # New dictionary to hold all individual scores
        
        # Process results (iterating through effective_algo names now)
        for effective_algo in eval_results:
            scores_report[effective_algo] = {}
            all_individual_scores[effective_algo] = {} # Initialize algo key
            
            for provider in eval_results[effective_algo]:
                scores_report[effective_algo][provider] = {}
                all_individual_scores[effective_algo][provider] = {} # Initialize provider key
                
                # Initialize stats
                file_count = 0
                
                # Initialize sums for detailed scores based on confirmed structure
                faithfulness_scores_sum = 0 # CR
                clarity_scores_sum = 0      # EC
                structure_scores_sum = 0    # LS
                inspirational_scores_sum = 0# AE
                
                # Global dimensions sums
                narrative_flow_sum = 0
                info_hierarchy_sum = 0
                concept_integration_sum = 0
                thematic_consistency_sum = 0
                cross_referencing_sum = 0
                
                # Overall Global Score sum (average of the above 5)
                global_overall_score_sum = 0 # Was llm_judge_scores_sum
                
                individual_file_scores = {} # Keep collecting individual scores locally

                # Process each file
                for file_name, results in eval_results[effective_algo][provider].items():
                    if "error" not in results:
                        # Check if required keys exist based on confirmed structure
                        if "slide_evaluation" in results and "global_evaluation" in results and \
                           "average_scores" in results["slide_evaluation"] and len(results["slide_evaluation"]["average_scores"]) == 4 and \
                           "overall_score" in results["global_evaluation"] and \
                           "dimension_scores" in results["global_evaluation"]:
                           
                            # Check global dimension keys exist
                            global_dims = results["global_evaluation"]["dimension_scores"]
                            if all(k in global_dims for k in ["narrative_flow", "information_hierarchy", "conceptual_integration", "thematic_consistency", "cross_referencing"]):

                                file_count += 1
                                
                                # Extract slide scores using correct keys and mapping
                                slide_avg_scores = results["slide_evaluation"]["average_scores"]
                                faithfulness_score = slide_avg_scores[0] # CR
                                clarity_score = slide_avg_scores[1]      # EC
                                structure_score = slide_avg_scores[2]    # LS
                                inspirational_score = slide_avg_scores[3]# AE
                                
                                # Extract overall global score
                                global_overall_score = results["global_evaluation"]["overall_score"] 
                                
                                # Extract individual global dimension scores
                                nf_score = global_dims["narrative_flow"]
                                ih_score = global_dims["information_hierarchy"]
                                ci_score = global_dims["conceptual_integration"]
                                tc_score = global_dims["thematic_consistency"]
                                cr_score = global_dims["cross_referencing"]

                                # Add slide scores to sums
                                faithfulness_scores_sum += faithfulness_score
                                clarity_scores_sum += clarity_score
                                structure_scores_sum += structure_score
                                inspirational_scores_sum += inspirational_score
                                
                                # Add global scores to sums
                                narrative_flow_sum += nf_score
                                info_hierarchy_sum += ih_score
                                concept_integration_sum += ci_score
                                thematic_consistency_sum += tc_score
                                cross_referencing_sum += cr_score
                                global_overall_score_sum += global_overall_score

                                # Store individual file detailed scores 
                                individual_file_scores[file_name] = {
                                    # Slide dims
                                    "content_relevance": faithfulness_score,
                                    "expressive_clarity": clarity_score,
                                    "logical_structure": structure_score,
                                    "audience_engagement": inspirational_score,
                                    # Global dims
                                    "narrative_flow": nf_score,
                                    "information_hierarchy": ih_score,
                                    "conceptual_integration": ci_score,
                                    "thematic_consistency": tc_score,
                                    "cross_referencing": cr_score,
                                    # Overall scores
                                    "global_score": global_overall_score, # Renamed from llm_judge_score
                                    "slide_overall_score": results["slide_evaluation"].get("overall_score"), # Renamed from raw_slide_overall_score
                                    "combined_score": results.get("combined_score") # Renamed from raw_combined_score
                                    # Optionally keep raw slide avg scores if needed
                                    # "raw_slide_avg_scores": slide_avg_scores,
                                }
                            else:
                                print(f"Warning: Missing expected global dimension score keys in results for {file_name} under {effective_algo}/{provider}. Skipping this file for averaging.")
                                if file_name not in individual_file_scores:
                                    individual_file_scores[file_name] = {"error": "Missing global dimension score keys"}
                        else:
                            print(f"Warning: Missing expected score keys (slide_evaluation, global_evaluation, etc.) in results for {file_name} under {effective_algo}/{provider}. Skipping this file for averaging.")
                            if file_name not in individual_file_scores:
                                individual_file_scores[file_name] = {"error": "Missing evaluation keys"}

                # Store the collected individual scores for this provider
                all_individual_scores[effective_algo][provider] = individual_file_scores

                # Calculate averages
                if file_count > 0:
                    # Calculate average slide overall score
                    avg_slide_overall = sum(f.get("slide_overall_score", 0) for f in individual_file_scores.values() if isinstance(f, dict) and "slide_overall_score" in f) / file_count
                    # Calculate combined score
                    avg_global = global_overall_score_sum / file_count
                    avg_combined = (avg_slide_overall + avg_global) / 2
                    
                    scores_report[effective_algo][provider] = {
                        # Slide dimensions averages
                        "avg_content_relevance": round(faithfulness_scores_sum / file_count, 2),
                        "avg_expressive_clarity": round(clarity_scores_sum / file_count, 2),
                        "avg_logical_structure": round(structure_scores_sum / file_count, 2),
                        "avg_audience_engagement": round(inspirational_scores_sum / file_count, 2),
                        # Global dimensions averages
                        "avg_narrative_flow": round(narrative_flow_sum / file_count, 2),
                        "avg_information_hierarchy": round(info_hierarchy_sum / file_count, 2),
                        "avg_conceptual_integration": round(concept_integration_sum / file_count, 2),
                        "avg_thematic_consistency": round(thematic_consistency_sum / file_count, 2),
                        "avg_cross_referencing": round(cross_referencing_sum / file_count, 2),
                        # Overall scores averages
                        "avg_slide_score": round(avg_slide_overall, 2), # Renamed from avg_slide_overall_score
                        "avg_global_score": round(avg_global, 2), # Renamed from avg_llm_judge_score
                        "avg_combined_score": round(avg_combined, 2),
                        # Meta
                        "file_count": file_count,
                    }
                    print(f"Report summary for {effective_algo}/{provider}: Successfully processed {file_count} files.")
                else:
                    scores_report[effective_algo][provider] = {
                        "error": "No valid evaluations found"
                    }
                    print(f"Report summary for {effective_algo}/{provider}: No valid evaluations found (0 files).")
        
        # Save the detailed individual scores to a separate file
        individual_scores_path = output_dir / f"individual_scores_{timestamp}.json"
        with open(individual_scores_path, 'w', encoding='utf-8') as f:
            json.dump(all_individual_scores, f, ensure_ascii=False, indent=2)
        print(f"\nIndividual file scores saved to: {individual_scores_path}")

        # Save summary scores report (without individual files)
        scores_report_path = output_dir / f"scores_report_{timestamp}.json"
        
        with open(scores_report_path, 'w', encoding='utf-8') as f:
            json.dump(scores_report, f, ensure_ascii=False, indent=2)
        
        print(f"\nScores report saved to: {scores_report_path}")
        return scores_report_path
        
    except Exception as e:
        print(f"Error generating scores report: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate lecture scripts from presentations')
    
    # Input path - file or directory
    parser.add_argument('input', nargs='?', default=None, help='Path to a presentation file or directory containing presentations (required unless --report-only is used)')
    # Force regeneration
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force regeneration even if results already exist')
    # Skip evaluation
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation after generation')
    # Only evaluate (no generation)
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing scripts without generation (requires input path)')
    # Generate scores report from existing evaluation results (DEPRECATED but kept for compatibility)
    parser.add_argument('--report', metavar='EVAL_RESULTS_PATH', 
                        help='DEPRECATED: Use --report-only instead. Generate report from results.')
    # Generate ONLY the scores report from existing evaluation results
    parser.add_argument('--report-only', metavar='EVAL_RESULTS_PATH', 
                        help='Generate ONLY the scores reports (summary and individual) from an existing evaluation results JSON file.')
    
    args = parser.parse_args()
    
    if args.report_only:
        print(f"Generating reports only from: {args.report_only}")
        generate_scores_report(args.report_only)
    elif args.report: # Handle deprecated argument
        print(f"Generating reports from (using deprecated --report): {args.report}")
        print("Warning: --report is deprecated. Please use --report-only in the future.")
        generate_scores_report(args.report)
    elif args.eval_only:
        if not args.input:
            parser.error("--eval-only requires the input path argument.")
        print(f"Starting evaluation only for: {args.input}")
        eval_results_path = evaluate_all(args.input)
        if eval_results_path:
            print(f"\nGenerating scores report from evaluation results: {eval_results_path}")
            generate_scores_report(eval_results_path)
    else:
        if not args.input:
             parser.error("Input path argument is required unless --report-only is used.")
        print(f"Starting generation process for: {args.input}")
        generate(args.input, args.force)
        if not args.no_eval:
            print(f"\nStarting evaluation process for: {args.input}")
            eval_results_path = evaluate_all(args.input)
            if eval_results_path:
                print(f"\nGenerating scores report from evaluation results: {eval_results_path}")
                generate_scores_report(eval_results_path)
        else:
            print("Skipping evaluation as requested.")


if __name__ == "__main__":
    main()
