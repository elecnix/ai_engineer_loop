#!/usr/bin/env python3
"""
AI Software Engineer Loop

This program reads a software spec prompt, generates a Python implementation
that satisfies the spec along with tests, runs the tests, and iteratively
improves the implementation until all tests pass.
"""

import os
import sys
import json
import argparse
import subprocess
import re
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse

# Load environment variables
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="http://localhost:3000"
)

# Initialize OpenAI client for Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

# Constants
MODEL = "deepseek-coder"
MEMORY_FILE = "memory.json"
IMPLEMENTATION_FILE = "implementation.py"
MAX_ITERATIONS = 3  # Maximum number of iterations to prevent infinite loops


def parse_arguments() -> str:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Software Engineer Loop")
    parser.add_argument("prompt", nargs="?", help="Software spec prompt or file containing the prompt")
    args = parser.parse_args()

    # If prompt is provided as an argument
    if args.prompt:
        # Check if it's a file
        if os.path.isfile(args.prompt):
            with open(args.prompt, 'r') as file:
                return file.read()
        # Otherwise, treat it as a direct prompt
        return args.prompt
    
    # If no prompt is provided, read from stdin
    print("Enter software spec prompt (Ctrl+D to finish):")
    return sys.stdin.read()


def extract_code_from_response(response: str) -> str:
    """Extract code from the response that is enclosed in ```python and ``` markers.
    If no markers are found, clean the response as best as possible.
    """
    # First, try to extract code from markdown code blocks
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if not matches:
        # Try without language specifier
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        # Join all code blocks if multiple are found
        if len(matches) > 1:
            print(f"Warning: Multiple code blocks found ({len(matches)}). Joining them.")
        return '\n\n'.join(matches)
    
    # If no code blocks found, clean the response
    # Remove common prefixes that might indicate explanations
    cleaned_response = response
    prefixes_to_remove = [
        r"^Here's the implementation:.*?\n",
        r"^Here is the implementation:.*?\n",
        r"^The implementation is:.*?\n",
        r"^Here's the code:.*?\n",
        r"^Here is the code:.*?\n",
        r"^Implementation:.*?\n",
        r"^Code:.*?\n"
    ]
    
    for prefix in prefixes_to_remove:
        cleaned_response = re.sub(prefix, "", cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove trailing explanations
    suffixes_to_remove = [
        r"\nThis implementation.*$",
        r"\nThe code above.*$",
        r"\nIn this implementation.*$"
    ]
    
    for suffix in suffixes_to_remove:
        cleaned_response = re.sub(suffix, "", cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    
    print("No code blocks found. Returning cleaned response.")
    return cleaned_response


def save_implementation(code: str, filename: str = IMPLEMENTATION_FILE) -> None:
    """Save the implementation code to a file."""
    with open(filename, 'w') as file:
        file.write(code)
    print(f"Implementation saved to {filename}")


def run_tests(implementation_file: str = IMPLEMENTATION_FILE, prompt: str = None) -> Tuple[bool, str]:
    """Run the tests in the implementation file and return results.
    
    Uses the Ollama model to determine if tests passed by analyzing the prompt,
    code, and test output.
    """
    try:
        # Read the implementation code
        with open(implementation_file, 'r') as file:
            implementation_code = file.read()
            
        # Run the tests
        try:
            result = subprocess.run(
                ["python3", implementation_file],
                capture_output=True,
                text=True,
                timeout=30  # Set a timeout to prevent hanging
            )
        except subprocess.TimeoutExpired:
            return False, "Execution timed out after 30 seconds."
        except Exception as e:
            return False, f"Error executing tests: {str(e)}"
        
        # Combine stdout and stderr for complete output
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
            
        # If there was an error running the code, it definitely failed
        if result.returncode != 0:
            return False, output
            
        # Use Ollama to determine if tests passed
        trace = langfuse.trace(
            name="evaluate_tests",
            metadata={"returncode": result.returncode}
        )
        
        test_evaluation_prompt = f"""
        You are an expert Python developer evaluating test results.
        
        Original specification:
        {prompt}
        
        Implementation code:
        ```python
        {implementation_code}
        ```
        
        Test output:
        ```
        {output}
        ```
        
        Based on the specification, implementation, and test output, determine if ALL tests have passed.
        
        Respond with ONLY 'PASSED' or 'FAILED' followed by a one-sentence explanation.
        """
        
        generation = trace.generation(
            name="test_evaluation",
            model=MODEL,
            input={
                "prompt": test_evaluation_prompt,
                "test_output": output, 
                "implementation": implementation_code
            }
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Python developer evaluating test results. Your task is to determine if all tests have passed based on the implementation, specification, and test output. Respond with ONLY 'PASSED' or 'FAILED' followed by a brief explanation."},
                {"role": "user", "content": test_evaluation_prompt}
            ]
        )
        
        evaluation = response.choices[0].message.content
        
        generation.end(
            output=evaluation,
            metadata={"returncode": result.returncode}
        )
        
        # Determine if tests passed based on the model's evaluation
        all_tests_passed = evaluation.strip().startswith("PASSED")
        
        # Add the evaluation to the output
        output += "\n\nModel Evaluation: " + evaluation
            
        return all_tests_passed, output
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Error running tests: {str(e)}"


def load_memory() -> Dict[str, Any]:
    """Load the memory from file if it exists."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error reading memory file. Starting with empty memory.")
    return {"learnings": [], "iterations": 0}


def save_memory(memory: Dict[str, Any]) -> None:
    """Save the memory to a file."""
    with open(MEMORY_FILE, 'w') as file:
        json.dump(memory, file, indent=2)
    print(f"Memory saved to {MEMORY_FILE}")


def update_memory(implementation: str, test_output: str, memory: Dict[str, Any]) -> Dict[str, Any]:
    """Update the memory with learnings from the current iteration."""
    trace = langfuse.trace(
        name="update_memory",
        metadata={"iteration": memory["iterations"] + 1}
    )
    
    memory_prompt = f"""
    You are an AI software engineer analyzing test results and implementation code.
    Based on the implementation and test output, identify key learnings that would be useful for future iterations.
    
    Implementation:
    ```python
    {implementation}
    ```
    
    Test Output:
    ```
    {test_output}
    ```
    
    Previous Learnings:
    {memory.get("learnings", [])}
    
    Provide ONLY 2-3 one-line bullet points that describe:
    1. A brief description of the current solution approach
    2. The main problems with the current implementation
    
    Each bullet point MUST be one line only. Be extremely concise.
    DO NOT provide general programming advice. Focus ONLY on specific issues with THIS implementation.
    """
    
    generation = trace.generation(
        name="memory_update",
        model=MODEL,
        input={
            "prompt": memory_prompt,
            "implementation": implementation, 
            "test_output": test_output
        }
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert software engineer analyzing test results and implementation code. Provide extremely concise one-line bullet points about the current solution approach and main problems. No explanations or general advice."},
            {"role": "user", "content": memory_prompt}
        ]
    )
    
    new_learnings = response.choices[0].message.content
    
    generation.end(
        output=new_learnings,
        metadata={"iteration": memory["iterations"] + 1}
    )
    
    # Update memory with new learnings
    memory["learnings"].append({
        "iteration": memory["iterations"] + 1,
        "learning": new_learnings
    })
    memory["iterations"] += 1
    
    return memory


def generate_implementation(prompt: str, memory: Dict[str, Any], test_output: Optional[str] = None) -> str:
    """Generate implementation based on the prompt and memory."""
    trace = langfuse.trace(
        name="generate_implementation",
        metadata={"iteration": memory["iterations"] + 1}
    )
    
    # Construct the prompt for the implementation
    if memory["iterations"] == 0:
        # First iteration
        implementation_prompt = f"""
        You are an expert Python developer. Write a complete Python implementation that satisfies this specification:
        
        {prompt}
        
        Your implementation should:
        1. Be self-contained in a single file
        2. Include comprehensive tests using unittest or pytest
        3. Use minimal comments - only add comments for complex logic that needs explanation
        4. Follow best practices for Python code with clear variable names that are self-documenting
        5. Handle edge cases appropriately
        
        IMPORTANT: Your response must contain ONLY the Python code implementation, with no explanations, comments outside the code, or markdown formatting. Do not include any text like 'Here's the implementation' or 'This code does X'. Just provide the raw Python code that would be saved directly to a .py file.
        
        The implementation should be complete and runnable as-is, with no placeholders or TODO comments.
        """
    else:
        # Subsequent iterations with memory and test output
        learnings = "\n".join([f"{i+1}. {item['learning']}" for i, item in enumerate(memory["learnings"])])
        
        implementation_prompt = f"""
        You are an expert Python developer. Improve the implementation based on test results and learnings.
        
        Original specification:
        {prompt}
        
        Test output from previous iteration:
        ```
        {test_output}
        ```
        
        Learnings from previous iterations:
        {learnings}
        
        Your implementation should:
        1. Be self-contained in a single file
        2. Include comprehensive tests using unittest or pytest
        3. Use minimal comments - only add comments for complex logic that needs explanation
        4. Follow best practices for Python code with clear variable names that are self-documenting
        5. Handle edge cases appropriately
        6. Fix all issues identified in the test output
        
        IMPORTANT: Your response must contain ONLY the Python code implementation, with no explanations, comments outside the code, or markdown formatting. Do not include any text like 'Here's the implementation' or 'This code does X'. Just provide the raw Python code that would be saved directly to a .py file.
        
        The implementation should be complete and runnable as-is, with no placeholders or TODO comments.
        """
    
    generation = trace.generation(
        name="code_generation",
        model=MODEL,
        input={
            "prompt": implementation_prompt,
            "spec": prompt, 
            "test_output": test_output, 
            "memory": memory
        }
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Python developer. Your task is to write Python code that implements the given specification. Return ONLY the Python code with no explanations, markdown formatting, or text outside of the code itself. The code should be complete, runnable, and include tests."},
            {"role": "user", "content": implementation_prompt}
        ]
    )
    
    implementation = response.choices[0].message.content
    code = extract_code_from_response(implementation)
    
    generation.end(
        output=code,
        metadata={"iteration": memory["iterations"] + 1}
    )
    
    return code


def main():
    """Main function to run the AI software engineering loop."""
    # Parse arguments to get the prompt
    prompt = parse_arguments()
    
    # Load memory
    memory = load_memory()
    
    print(f"Starting AI Software Engineer Loop with {MODEL} model")
    print(f"Current iteration: {memory['iterations'] + 1}")
    
    # Initialize variables
    all_tests_passed = False
    test_output = None
    iteration = 0
    
    # Main loop
    while not all_tests_passed and iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        
        # Generate implementation
        print("Generating implementation...")
        implementation = generate_implementation(prompt, memory, test_output)
        
        # Save implementation to file
        save_implementation(implementation)
        
        # Run tests
        print("Running tests...")
        all_tests_passed, test_output = run_tests(prompt=prompt)
        print(f"Tests {'PASSED' if all_tests_passed else 'FAILED'}")
        print("Test output:")
        print("-" * 40)
        print(test_output)
        print("-" * 40)
        
        # Update memory with learnings
        print("Updating memory with learnings...")
        memory = update_memory(implementation, test_output, memory)
        save_memory(memory)
        
        if all_tests_passed:
            print("\n🎉 All tests passed! Final implementation is ready.")
            break
        
        if iteration >= MAX_ITERATIONS:
            print(f"\n⚠️ Reached maximum iterations ({MAX_ITERATIONS}). Stopping.")
            break
    
    print(f"\nFinal implementation saved to {IMPLEMENTATION_FILE}")
    print(f"Memory saved to {MEMORY_FILE}")


if __name__ == "__main__":
    main()
