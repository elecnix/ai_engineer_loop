# AI Software Engineer Loop

This program implements an AI software engineering loop that:

1. Reads a software specification prompt
2. Generates a Python implementation that satisfies the spec
3. Creates and runs tests for the implementation
4. Analyzes test results
5. Iteratively improves the implementation until all tests pass

The program uses Ollama with configurable models to generate code and maintains a memory file to track learnings across iterations. It includes a model evaluator that can test multiple Ollama models to compare their performance.

## Requirements

- Python 3.7+
- Ollama installed and running locally
- Langfuse running locally

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install openai langfuse python-dotenv
```

3. Make sure Ollama is running with the `deepseek-coder` model:

```bash
ollama pull deepseek-coder
ollama run deepseek-coder
```

4. Set up Langfuse environment variables in a `.env` file:

```
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
```

## Usage

You can provide the software specification prompt in three ways:

1. As a command-line argument:

```bash
python ai_engineer.py "Create a function that calculates the factorial of a number"
```

2. As a file containing the prompt:

```bash
python ai_engineer.py prompt.txt
```

3. Via standard input:

```bash
python ai_engineer.py
# Then type or paste your prompt and press Ctrl+D when finished
```

## How It Works

### AI Engineer Loop

1. The program reads the software specification prompt.
2. It uses Ollama with the configured model to generate an initial implementation with tests.
3. It runs the tests and checks if they pass.
4. If tests fail, it updates a memory file with learnings from the current iteration.
5. It generates an improved implementation based on the original prompt, test results, and memory.
6. Steps 3-5 repeat until all tests pass or the maximum number of iterations is reached.

### Model Evaluator

1. The evaluator identifies all available Ollama models.
2. For each model, it runs multiple evaluations (default: 5 runs per model).
3. Each run allows multiple iterations (default: 3 iterations per run).
4. Results are organized in a directory structure:
   ```
   model_evaluations/
   ├── model_name_1/
   │   ├── 001/
   │   │   ├── implementation.py
   │   │   ├── memory.json
   │   │   └── output.log
   │   ├── 002/
   │   └── ...
   ├── model_name_2/
   └── ...
   ```
5. A global results file (`model_evaluation_results.json`) tracks the performance of each model.
6. The evaluator can be interrupted and resumed at any point.

## Output Files

- `implementation.py`: The generated Python implementation
- `memory.json`: A JSON file containing learnings from each iteration

## Limitations

- The program is limited to a maximum of 10 iterations to prevent infinite loops.
- The implementation is limited to a single Python file.
- Test execution has a timeout of 30 seconds to prevent hanging.

## Customization

You can modify the following constants in the script:

- `MODEL`: The Ollama model to use (default: "deepseek-coder")
- `MEMORY_FILE`: The file to store memory (default: "memory.json")
- `IMPLEMENTATION_FILE`: The file to store the implementation (default: "implementation.py")
- `MAX_ITERATIONS`: Maximum number of iterations (default: 10)
