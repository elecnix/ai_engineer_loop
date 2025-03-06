#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies if needed
if ! pip list | grep -q "pytest"; then
  uv add pytest
fi

# List of prompts to evaluate
PROMPTS=(
  "sample_prompts/1_distributed_rate_limiter.txt"
  "sample_prompts/2_inmemory_database.txt"
  "sample_prompts/3_neural_network_framework.txt"
  "sample_prompts/4_distributed_task_scheduler.txt"
  "sample_prompts/5_compiler.txt"
)

# Challenge names
CHALLENGE_NAMES=(
  "distributed_rate_limiter"
  "inmemory_database"
  "neural_network_framework"
  "distributed_task_scheduler"
  "compiler"
)

# Models to evaluate (all available models)
MODELS="starcoder2:7b
qwen2.5:0.5b
reader-lm:0.5b
llama3:8b
tinyllama:1.1b
qwen2.5-coder:1.5b
llama3-chatqa:8b
granite3-dense:2b
exaone3.5:2.4b
codegeex4:9b
qwen2.5-coder:0.5b-instruct-q4_K_S
starcoder2:3b
phi3.5:latest
dolphin-mistral:7b
qwen2:latest
nemotron-mini:latest
qwen2.5-coder:7b
granite3.2:8b
deepseek-v2:latest
deepseek-coder-v2:lite"

# Run evaluations for each prompt
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  CHALLENGE_NAME="${CHALLENGE_NAMES[$i]}"
  OUTPUT_DIR="model_evaluations/${CHALLENGE_NAME}"
  
  echo "==============================================================="
  echo "Evaluating prompt: $PROMPT"
  echo "Challenge: $CHALLENGE_NAME"
  echo "Output directory: $OUTPUT_DIR"
  echo "==============================================================="
  
  # Create output directory if it doesn't exist
  mkdir -p "$OUTPUT_DIR"
  
  # Run the evaluation with the specified base directory
  python model_evaluator.py "$PROMPT" --models $MODELS --base-dir "$OUTPUT_DIR" --resume > "$OUTPUT_DIR/evaluation.log" 2>&1
  
  echo "Evaluation complete for $CHALLENGE_NAME. Results saved to $OUTPUT_DIR/"
  echo ""
done

echo "All evaluations complete!"
