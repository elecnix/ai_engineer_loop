#!/bin/bash

# Ensure pytest is installed using uv
if ! uv pip list | grep -q "pytest"; then
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

# Models to evaluate (filtered models)
MODELS="deepseek-coder:1.3b
deepseek-coder:6.7b-instruct
granite3.2:8b
deepseek-coder:latest
exaone3.5:7.8b
exaone3.5:2.4b
qwen2.5-coder:0.5b-instruct-q4_K_S
qwen2.5-coder:14b-instruct-q2_K
qwen2.5-coder:0.5b
qwen2.5-coder:1.5b
qwen2.5-coder:3b
qwen2.5-coder:7b
qwen2.5-coder:latest
codegeex4:9b
starcoder2:7b
granite-code:8b
granite-code:3b
dolphin-mistral:7b
deepseek-coder:6.7b
deepseek-coder-v2:lite
granite3-dense:2b
granite3-dense:8b
deepseek-v2:latest
starcoder2:3b"

# Run evaluations for each prompt
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  CHALLENGE_NAME="${CHALLENGE_NAMES[$i]}"
  OUTPUT_DIR="generated/model_evaluations/${CHALLENGE_NAME}"
  
  echo "==============================================================="
  echo "Evaluating prompt: $PROMPT"
  echo "Challenge: $CHALLENGE_NAME"
  echo "Output directory: $OUTPUT_DIR"
  echo "==============================================================="
  
  # Create output directory if it doesn't exist
  mkdir -p "$OUTPUT_DIR"
  
  # Run the evaluation with the specified base directory using uv run
  uv run model_evaluator.py "$PROMPT" --models $MODELS --base-dir "$OUTPUT_DIR" --resume > "$OUTPUT_DIR/evaluation.log" 2>&1
  
  echo "Evaluation complete for $CHALLENGE_NAME. Results saved to $OUTPUT_DIR/"
  echo ""
done

echo "All evaluations complete!"
