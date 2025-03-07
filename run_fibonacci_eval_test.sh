#!/bin/bash

# Test with just one quick model to verify our fixes
MODELS="starcoder2:3b"

# Run evaluation for each model
for MODEL in $MODELS; do
  echo "Evaluating model: $MODEL"
  
  # Create model-specific directory with sanitized name (replace colons with hyphens)
  SANITIZED_MODEL=$(echo "$MODEL" | tr ':' '-')
  MODEL_DIR="generated/fibonacci_test_results/${SANITIZED_MODEL}"
  mkdir -p "$MODEL_DIR"
  
  # Run the AI Engineer with the model
  python ai_engineer.py \
    --model "$MODEL" \
    --implementation-file "${MODEL_DIR}/implementation.py" \
    --conversation-file "${MODEL_DIR}/conversation.json" \
    --usage-file "${MODEL_DIR}/usage.json" \
    --max-iterations 2 \
    sample_prompts/fibonacci.txt
  
  echo "Completed evaluation for $MODEL"
  echo "----------------------------------------"
done

echo "Evaluation complete! Results saved to fibonacci_test_results directory"
