#!/bin/bash

# Get the list of top-performing models
MODELS=$(ollama list | grep 'exaone\|codegeex4\|granite\|qwen2.5-coder\|dolphin-mistral\|deepseek-coder\|deepseek-v2\|starcoder2' | awk '{print $1}')

# Create output directory
mkdir -p flask_eval_results

# Run evaluation for each model
for MODEL in $MODELS; do
  echo "Evaluating model: $MODEL"
  
  # Create model-specific directory
  MODEL_DIR="flask_eval_results/${MODEL}"
  mkdir -p "$MODEL_DIR"
  
  # Run the AI Engineer with the model
  python ai_engineer.py \
    --model "$MODEL" \
    --implementation-file "${MODEL_DIR}/implementation.py" \
    --conversation-file "${MODEL_DIR}/conversation.json" \
    --usage-file "${MODEL_DIR}/usage.json" \
    --max-iterations 2 \
    prompts/flask_random_numbers.txt
    
  echo "Completed evaluation for $MODEL"
  echo "----------------------------------------"
done

# Run the model evaluator to generate a summary
python model_evaluator.py \
  --base-dir "flask_eval_results" \
  --results-file "flask_eval_results.json" \
  prompts/flask_random_numbers.txt

echo "Evaluation complete! Results saved to flask_eval_results.json"
