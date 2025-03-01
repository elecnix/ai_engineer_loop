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
MAX_ITERATIONS = 10  # Maximum number of iterations to prevent infinite loops


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
    """Extract code from the response that is enclosed in ```python and ``` markers."""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if not matches:
        # Try without language specifier
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0]
    
    # If no code blocks found, return the entire response
    return response


def save_implementation(code: str, filename: str = IMPLEMENTATION_FILE) -> None:
    """Save the implementation code to a file."""
    with open(filename, 'w') as file:
        file.write(code)
    print(f"Implementation saved to {filename}")


def run_tests(implementation_file: str = IMPLEMENTATION_FILE) -> Tuple[bool, str]:
    """Run the tests in the implementation file and return results."""
    try:
        result = subprocess.run(
            ["python", implementation_file],
            capture_output=True,
            text=True,
            timeout=30  # Set a timeout to prevent hanging
        )
        
        # Check if all tests passed
        all_tests_passed = "FAILED" not in result.stdout and result.returncode == 0
        
        # Combine stdout and stderr for complete output
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
            
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
    Focus on:
    1. What worked well
    2. What didn't work
    3. Specific bugs or issues identified
    4. Patterns or techniques that should be applied or avoided
    
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
    
    Provide a concise list of new learnings that should be remembered for future iterations.
    """
    
    generation = trace.generation(
        name="memory_update",
        model=MODEL,
        prompt=memory_prompt,
        input={"implementation": implementation, "test_output": test_output}
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert software engineer analyzing test results and implementation code."},
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
        2. Include comprehensive tests
        3. Be well-documented with comments
        4. Follow best practices for Python code
        5. Handle edge cases appropriately
        
        Return ONLY the Python code without any explanations or markdown formatting.
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
        2. Include comprehensive tests
        3. Be well-documented with comments
        4. Follow best practices for Python code
        5. Handle edge cases appropriately
        6. Fix all issues identified in the test output
        
        Return ONLY the Python code without any explanations or markdown formatting.
        """
    
    generation = trace.generation(
        name="code_generation",
        model=MODEL,
        prompt=implementation_prompt,
        input={"spec": prompt, "test_output": test_output, "memory": memory}
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Python developer."},
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
        all_tests_passed, test_output = run_tests()
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
