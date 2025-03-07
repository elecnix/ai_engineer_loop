#!/usr/bin/env python3
"""
AI Model Evaluator

This program evaluates multiple Ollama models on their ability to implement
a given specification. Each model gets multiple runs with multiple iterations
per run to test its performance.
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from typing import Dict, List, Any
from pathlib import Path

# Import the AI engineer module
import ai_engineer
from ai_engineer import run_tests, parse_arguments, load_conversation, save_conversation, extract_code_from_response


# Constants
MAX_RUNS_PER_MODEL = 3
MAX_ITERATIONS_PER_RUN = 5
IMPLEMENTATION_FILE = "implementation.py"
CONVERSATION_FILE = "conversation.json"

# These will be set in main() based on command line arguments
BASE_DIR = "model_evaluations"
RESULTS_FILE = "model_evaluation_results.json"

def get_available_models() -> List[str]:
    """Get a list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        # Skip the header line
        models = []
        for line in lines[1:]:
            if line.strip():
                # Extract model name from the first column
                model_name = line.split()[0]
                models.append(model_name)
        
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error getting models: {e}")
        return []

def load_results() -> Dict[str, Any]:
    """Load existing evaluation results."""
    results_path = Path(BASE_DIR) / RESULTS_FILE
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {
        "models_evaluated": {},
        "current_model": None,
        "current_run": 0
    }

def save_results(results: Dict[str, Any]) -> None:
    """Save evaluation results to file."""
    results_path = Path(BASE_DIR) / RESULTS_FILE
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

def setup_directories(model: str, run: int) -> Dict[str, str]:
    """Set up directories for the model and run."""
    # Create model directory if it doesn't exist
    model_dir = Path(BASE_DIR) / model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run directory
    run_dir = model_dir / f"{run:03d}"
    run_dir.mkdir(exist_ok=True)
    
    # Return paths
    return {
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "implementation_path": str(run_dir / IMPLEMENTATION_FILE),
        "conversation_path": str(run_dir / CONVERSATION_FILE),
        "usage_path": str(run_dir / "usage_data.json")
    }

def run_model_evaluation(model: str, run: int, prompt_path: str) -> Dict[str, Any]:
    """Run a single evaluation for a model and run number."""
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model} (Run {run}/{MAX_RUNS_PER_MODEL})")
    print(f"{'='*80}")
    
    # Setup directories
    paths = setup_directories(model, run)
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = model
    env["IMPLEMENTATION_FILE"] = paths["implementation_path"]
    env["CONVERSATION_FILE"] = paths["conversation_path"]
    env["MAX_ITERATIONS"] = str(MAX_ITERATIONS_PER_RUN)
    
    # Create empty conversation file
    with open(paths["conversation_path"], 'w') as f:
        json.dump([], f, indent=2)
    
    # Run the AI engineer process
    try:
        result = subprocess.run(
            ["python3", "ai_engineer.py", prompt_path, "--usage-file", paths["usage_path"]],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        
        output = result.stdout + result.stderr
        
        # Save the output to a log file
        with open(os.path.join(paths["run_dir"], "output.log"), 'w') as f:
            f.write(output)
        
        # Check results.json for test status and iteration count
        results_path = os.path.join(paths["run_dir"], "results.json")
        passed = False
        iterations = 0
        has_tests = False
        
        try:
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    if results and isinstance(results, list):
                        # Get the last result (final iteration)
                        last_result = results[-1]
                        passed = last_result.get("tests_passed", False)
                        has_tests = last_result.get("has_tests", False)
                        libraries_installed = last_result.get("libraries_installed", False)
                        iterations = len(results)
            else:
                print(f"Warning: results.json not found at {results_path}")
                # Fallback to checking output
                passed = "All tests passed!" in output
                
                # Fallback to getting iterations from conversation
                try:
                    with open(paths["conversation_path"], 'r') as f:
                        conversation = json.load(f)
                        # Each iteration has 2 messages (assistant + user)
                        iterations = len(conversation) // 2
                except Exception as e:
                    print(f"Error reading conversation file: {e}")
        except Exception as e:
            print(f"Error reading conversation file: {e}")
        
        # Get usage data if available
        usage_data = None
        try:
            if os.path.exists(paths["usage_path"]):
                with open(paths["usage_path"], 'r') as f:
                    usage_data = json.load(f)
        except Exception as e:
            print(f"Error reading usage data file: {e}")
        
        return {
            "passed": passed,
            "has_tests": has_tests,
            "exit_code": result.returncode,
            "output_path": os.path.join(paths["run_dir"], "output.log"),
            "iterations": iterations,
            "usage_data": usage_data
        }
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return {
            "passed": False,
            "has_tests": False,
            "exit_code": -1,
            "error": str(e),
            "usage_data": None
        }

def main():
    """Main function to run the model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate multiple Ollama models on a coding task")
    parser.add_argument("prompt_path", help="Path to the prompt file")
    parser.add_argument("--models", nargs="+", help="List of models to evaluate (space-separated)")
    parser.add_argument("--model", help="Single specific model to evaluate (overrides --models)")
    parser.add_argument("--resume", action="store_true", help="Resume from last evaluation")
    parser.add_argument("--base-dir", default="model_evaluations", help="Base directory for evaluation results")
    parser.add_argument("--results-file", default="model_evaluation_results.json", help="Name of the results file")
    args = parser.parse_args()
    
    # Set global constants based on command line arguments
    global BASE_DIR, RESULTS_FILE
    BASE_DIR = args.base_dir
    RESULTS_FILE = args.results_file
    
    # Create base directory if it doesn't exist
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    results = load_results()
    
    # Get available models
    all_models = get_available_models()
    if not all_models:
        print("No models found. Make sure Ollama is installed and running.")
        sys.exit(1)
    
    # Determine which models to evaluate
    if args.model:  # Single model takes precedence
        if args.model in all_models:
            models_to_evaluate = [args.model]
        else:
            print(f"Model {args.model} not found. Available models: {', '.join(all_models[:5])}...")
            sys.exit(1)
    elif args.models:  # List of models provided via command line
        # Filter out models that aren't available
        models_to_evaluate = [model for model in args.models if model in all_models]
        if not models_to_evaluate:
            print("None of the specified models are available.")
            print(f"Available models: {', '.join(all_models[:5])}...")
            sys.exit(1)
    else:  # Default to all models if no specific models are provided
        models_to_evaluate = all_models
        print("No models specified, evaluating all available models.")
        print("This may take a long time. Consider specifying models with --models.")
    
    print(f"Evaluating {len(models_to_evaluate)} models: {', '.join(models_to_evaluate)}")
    
    # Resume from last evaluation if requested
    start_model_idx = 0
    start_run = 1
    
    if args.resume and results["current_model"]:
        # Check if the current model from results is in the list of models to evaluate
        if results["current_model"] in models_to_evaluate:
            # Get the index of the current model
            current_model_idx = models_to_evaluate.index(results["current_model"])
            current_run = results["current_run"]
            
            # If we're in the middle of runs for the current model
            if current_run < MAX_RUNS_PER_MODEL:
                # Continue with the current model, next run
                start_model_idx = current_model_idx
                start_run = current_run  # We'll continue with the current run (the loop will increment)
                print(f"Resuming with model {results['current_model']} run {current_run}")
            else:
                # We've completed all runs for this model, move to the next model
                start_model_idx = current_model_idx + 1
                start_run = 1
                
                if start_model_idx < len(models_to_evaluate):
                    next_model = models_to_evaluate[start_model_idx]
                    print(f"All runs completed for {results['current_model']}, moving to {next_model}")
                else:
                    print(f"All runs completed for {results['current_model']}, no more models to evaluate")
    
    # Evaluate models
    try:
        for i, model in enumerate(models_to_evaluate[start_model_idx:], start=start_model_idx):
            # Initialize model in results if not present
            if model not in results["models_evaluated"]:
                results["models_evaluated"][model] = {
                    "runs": {},
                    "pass_rate": 0.0,
                    "tests_included_rate": 0.0,
                    "libraries_installed_rate": 0.0,
                    "usage_stats": {
                        "total_tokens": 0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_duration_seconds": 0,
                        "total_api_calls": 0
                    }
                }
            
            # Set current model
            results["current_model"] = model
            
            # Reset start_run after the first model
            if i > start_model_idx:
                start_run = 1
                
            # Run evaluations
            for run in range(start_run, MAX_RUNS_PER_MODEL + 1):
                # Store the current run before execution
                # This way, if interrupted, we'll resume from this exact run
                results["current_run"] = run
                save_results(results)
                
                # Run evaluation
                run_result = run_model_evaluation(model, run, args.prompt_path)
                
                # Save run result
                results["models_evaluated"][model]["runs"][str(run)] = run_result
                
                # Update pass rate, tests included rate, and libraries installed rate
                passed_runs = sum(1 for r in results["models_evaluated"][model]["runs"].values() if r.get("passed", False))
                tests_included_runs = sum(1 for r in results["models_evaluated"][model]["runs"].values() if r.get("has_tests", False))
                libraries_installed_runs = sum(1 for r in results["models_evaluated"][model]["runs"].values() if r.get("libraries_installed", False))
                total_runs = len(results["models_evaluated"][model]["runs"])
                results["models_evaluated"][model]["pass_rate"] = passed_runs / total_runs if total_runs > 0 else 0.0
                results["models_evaluated"][model]["tests_included_rate"] = tests_included_runs / total_runs if total_runs > 0 else 0.0
                results["models_evaluated"][model]["libraries_installed_rate"] = libraries_installed_runs / total_runs if total_runs > 0 else 0.0
                
                # Update usage statistics if available
                if run_result.get("usage_data") and isinstance(run_result["usage_data"], dict) and "totals" in run_result["usage_data"]:
                    totals = run_result["usage_data"]["totals"]
                    
                    # Update model usage stats
                    results["models_evaluated"][model]["usage_stats"]["total_tokens"] += totals.get("total_tokens", 0)
                    results["models_evaluated"][model]["usage_stats"]["total_prompt_tokens"] += totals.get("total_prompt_tokens", 0)
                    results["models_evaluated"][model]["usage_stats"]["total_completion_tokens"] += totals.get("total_completion_tokens", 0)
                    results["models_evaluated"][model]["usage_stats"]["total_duration_seconds"] += totals.get("total_duration_seconds", 0)
                    results["models_evaluated"][model]["usage_stats"]["total_api_calls"] += totals.get("total_api_calls", 0)
          
                # Save results after each run
                save_results(results)
                
                # Print progress
                print(f"\nModel: {model} - Run {run}/{MAX_RUNS_PER_MODEL} - {'PASSED' if run_result['passed'] else 'FAILED'}")
                print(f"Tests included: {'YES' if run_result.get('has_tests', False) else 'NO'}")
                print(f"Libraries installed: {'YES' if run_result.get('libraries_installed', False) else 'NO'}")
                print(f"Iterations: {run_result.get('iterations', 'unknown')}")
                print(f"Current pass rate: {results['models_evaluated'][model]['pass_rate']:.2%}")
                print(f"Tests included rate: {results['models_evaluated'][model]['tests_included_rate']:.2%}")
                print(f"Libraries installed rate: {results['models_evaluated'][model]['libraries_installed_rate']:.2%}")
            
            # Reset start run for next model
            start_run = 1
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted. Progress has been saved.")
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Sort models by pass rate
    sorted_models = sorted(
        results["models_evaluated"].items(),
        key=lambda x: x[1]["pass_rate"],
        reverse=True
    )
    
    # Print header
    print(f"{'Model':<20} {'Pass Rate':<15} {'Tests Rate':<15} {'Libs Rate':<15} {'Total Tokens':<15} {'Prompt Tokens':<15} {'Completion':<15} {'Duration (s)':<15}")
    print("-" * 125)
    
    for model, data in sorted_models:
        usage = data.get("usage_stats", {})
        passes = sum(1 for r in data['runs'].values() if r.get('passed', False))
        tests_included = sum(1 for r in data['runs'].values() if r.get('has_tests', False))
        libraries_installed = sum(1 for r in data['runs'].values() if r.get('libraries_installed', False))
        total_runs = len(data['runs'])
        
        print(f"{model:<20} {data['pass_rate']:.2%} ({passes}/{total_runs}) {data['tests_included_rate']:.2%} ({tests_included}/{total_runs}) {data.get('libraries_installed_rate', 0):.2%} ({libraries_installed}/{total_runs}) {usage.get('total_tokens', 0):<15} {usage.get('total_prompt_tokens', 0):<15} {usage.get('total_completion_tokens', 0):<15} {usage.get('total_duration_seconds', 0):<15.2f}")
    
    results_path = Path(BASE_DIR) / RESULTS_FILE
    print("\nEvaluation complete. Results saved to", results_path)

if __name__ == "__main__":
    main()
