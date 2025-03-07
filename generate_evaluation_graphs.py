#!/usr/bin/env python3
import json
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

def load_evaluation_results(base_dir="generated/model_evaluations"):
    """Load evaluation results from multiple challenge directories."""
    all_results = {}
    
    # Find all challenge directories
    challenge_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for challenge in challenge_dirs:
        challenge_path = os.path.join(base_dir, challenge)
        result_file = os.path.join(challenge_path, "model_evaluation_results.json")
        
        if os.path.exists(result_file):
            with open(result_file, "r") as file:
                try:
                    data = json.load(file)
                    all_results[challenge] = data
                    print(f"Loaded results for challenge: {challenge}")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON file for challenge {challenge}")
    
    return all_results

def extract_metrics(all_results):
    """Extract key metrics from the evaluation results."""
    metrics = {
        "pass_rate": {},
        "tests_included_rate": {},
        "iterations": {},
        "tokens": {},
        "duration": {}
    }
    
    # Track models that appear in any challenge
    all_models = set()
    
    for challenge, data in all_results.items():
        models_evaluated = data.get("models_evaluated", {})
        
        for model, details in models_evaluated.items():
            all_models.add(model)
            
            # Initialize model data if not already present
            if model not in metrics["pass_rate"]:
                metrics["pass_rate"][model] = {}
                metrics["tests_included_rate"][model] = {}
                metrics["iterations"][model] = {}
                metrics["tokens"][model] = {}
                metrics["duration"][model] = {}
            
            # Store metrics for this model and challenge
            try:
                metrics["pass_rate"][model][challenge] = float(details.get("pass_rate", 0))
                metrics["tests_included_rate"][model][challenge] = float(details.get("tests_included_rate", 0))
            except (ValueError, TypeError):
                metrics["pass_rate"][model][challenge] = 0.0
                metrics["tests_included_rate"][model][challenge] = 0.0
            
            # Extract data from successful runs
            successful_runs = []
            for run_id, run in details.get("runs", {}).items():
                if run.get("passed", False):
                    try:
                        iterations = float(run.get("iterations", 0))
                        tokens = float(run.get("usage_data", {}).get("totals", {}).get("total_tokens", 0))
                        duration = float(run.get("usage_data", {}).get("totals", {}).get("total_duration_seconds", 0))
                        
                        successful_runs.append({
                            "iterations": iterations,
                            "tokens": tokens,
                            "duration": duration
                        })
                    except (ValueError, TypeError):
                        # Skip runs with invalid data
                        print(f"Warning: Skipping run with invalid data for {model} in {challenge}")
            
            # Calculate average metrics for successful runs
            if successful_runs:
                metrics["iterations"][model][challenge] = sum(run["iterations"] for run in successful_runs) / len(successful_runs)
                metrics["tokens"][model][challenge] = sum(run["tokens"] for run in successful_runs) / len(successful_runs)
                metrics["duration"][model][challenge] = sum(run["duration"] for run in successful_runs) / len(successful_runs)
            else:
                # If no successful runs, use NaN
                metrics["iterations"][model][challenge] = float('nan')
                metrics["tokens"][model][challenge] = float('nan')
                metrics["duration"][model][challenge] = float('nan')
    
    return metrics, list(all_models)

def create_dataframes(metrics, all_models, all_challenges):
    """Create pandas DataFrames for visualization."""
    dataframes = {}
    
    for metric_name, model_data in metrics.items():
        # Create a DataFrame with rows for each model and columns for each challenge
        df = pd.DataFrame(index=all_models, columns=all_challenges)
        
        for model in all_models:
            for challenge in all_challenges:
                if model in model_data and challenge in model_data[model]:
                    df.at[model, challenge] = model_data[model][challenge]
                else:
                    df.at[model, challenge] = float('nan')
        
        # Convert all values to float to ensure compatibility with heatmap
        df = df.astype(float)
        dataframes[metric_name] = df
    
    return dataframes

def plot_heatmaps(dataframes, output_dir="generated/evaluation_graphs"):
    """Generate heatmap visualizations for each metric."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot style
    sns.set(style="whitegrid")
    
    for metric_name, df in dataframes.items():
        plt.figure(figsize=(12, 10))
        
        # Create a mask for NaN values
        mask = df.isna()
        
        # Skip if all values are NaN
        if mask.all().all():
            print(f"Skipping {metric_name} heatmap - no data available")
            plt.close()
            continue
            
        # Choose an appropriate colormap based on the metric
        if metric_name == "pass_rate" or metric_name == "tests_included_rate":
            cmap = "YlGnBu"
            vmin = 0
            vmax = 1
        else:
            cmap = "viridis"
            vmin = None
            vmax = None
        
        # Create the heatmap
        try:
            ax = sns.heatmap(df, annot=True, cmap=cmap, mask=mask, 
                        linewidths=.5, fmt=".2f", vmin=vmin, vmax=vmax)
            
            # Set title and labels
            metric_title = {
                "pass_rate": "Pass Rate",
                "tests_included_rate": "Tests Included Rate",
                "iterations": "Average Iterations for Successful Runs",
                "tokens": "Average Tokens for Successful Runs",
                "duration": "Average Duration (seconds) for Successful Runs"
            }.get(metric_name, metric_name.capitalize())
            
            plt.title(f"{metric_title} by Model and Challenge", fontsize=16)
            plt.ylabel("Model", fontsize=12)
            plt.xlabel("Challenge", fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")
            
            # Save the figure
            output_file = os.path.join(output_dir, f"{metric_name}_heatmap.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Saved {metric_name} heatmap to {output_file}")
        except Exception as e:
            print(f"Error creating heatmap for {metric_name}: {e}")
            plt.close()

def plot_bar_charts(dataframes, output_dir="generated/evaluation_graphs"):
    """Generate bar charts for each metric, aggregated across challenges."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, df in dataframes.items():
        try:
            # Calculate the mean across challenges, ignoring NaN values
            mean_values = df.mean(axis=1, skipna=True).sort_values(ascending=False)
            
            # Filter out models with all NaN values
            mean_values = mean_values.dropna()
            
            if mean_values.empty:
                print(f"No data available for {metric_name} bar chart")
                continue
            
            plt.figure(figsize=(12, 8))
            
            # Choose an appropriate color based on the metric
            color = {
                "pass_rate": "blue",
                "tests_included_rate": "green",
                "iterations": "orange",
                "tokens": "purple",
                "duration": "red"
            }.get(metric_name, "gray")
            
            # Create the bar chart
            ax = mean_values.plot(kind='bar', color=color)
            
            # Set title and labels
            metric_title = {
                "pass_rate": "Average Pass Rate",
                "tests_included_rate": "Average Tests Included Rate",
                "iterations": "Average Iterations for Successful Runs",
                "tokens": "Average Tokens for Successful Runs",
                "duration": "Average Duration (seconds) for Successful Runs"
            }.get(metric_name, metric_name.capitalize())
            
            plt.title(f"{metric_title} by Model (Across All Challenges)", fontsize=16)
            plt.ylabel(metric_title, fontsize=12)
            plt.xlabel("Model", fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")
            
            # Add value labels on top of bars
            for i, v in enumerate(mean_values):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
            
            # Save the figure
            output_file = os.path.join(output_dir, f"{metric_name}_bar.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Saved {metric_name} bar chart to {output_file}")
        except Exception as e:
            print(f"Error creating bar chart for {metric_name}: {e}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation graphs from model evaluation results")
    parser.add_argument("--base-dir", default="generated/model_evaluations", 
                        help="Base directory containing challenge subdirectories with model_evaluation_results.json files")
    parser.add_argument("--output-dir", default="generated/evaluation_graphs", 
                        help="Directory to save the generated graphs")
    args = parser.parse_args()
    
    # Load evaluation results from all challenge directories
    all_results = load_evaluation_results(args.base_dir)
    
    if not all_results:
        print("No evaluation results found. Please check the base directory.")
        return
    
    # Extract metrics from the results
    metrics, all_models = extract_metrics(all_results)
    all_challenges = list(all_results.keys())
    
    print(f"Found {len(all_models)} models and {len(all_challenges)} challenges")
    
    # Create DataFrames for visualization
    dataframes = create_dataframes(metrics, all_models, all_challenges)
    
    # Generate visualizations
    plot_heatmaps(dataframes, args.output_dir)
    plot_bar_charts(dataframes, args.output_dir)
    
    print(f"All graphs have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()
