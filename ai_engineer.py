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
import time
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

# Constants - can be overridden by environment variables
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
CONVERSATION_FILE = os.environ.get("CONVERSATION_FILE", "conversation.json")
IMPLEMENTATION_FILE = os.environ.get("IMPLEMENTATION_FILE", "implementation.py")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))  # Maximum number of iterations to prevent infinite loops


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Software Engineer Loop")
    parser.add_argument("prompt", nargs="?", help="Software spec prompt or file containing the prompt")
    parser.add_argument("--usage-file", "-u", help="File to save API usage data (if not specified, usage data won't be tracked)")
    parser.add_argument("--model", "-m", help=f"Model to use (default: {MODEL})")
    parser.add_argument("--implementation-file", "-i", help=f"File to save implementation (default: {IMPLEMENTATION_FILE})")
    parser.add_argument("--conversation-file", "-c", help=f"File to save conversation (default: {CONVERSATION_FILE})")
    parser.add_argument("--max-iterations", type=int, help=f"Maximum number of iterations (default: {MAX_ITERATIONS})")
    args = parser.parse_args()

    # Get prompt
    prompt_text = ""
    if args.prompt:
        # Check if it's a file
        if os.path.isfile(args.prompt):
            with open(args.prompt, 'r') as file:
                prompt_text = file.read()
        # Otherwise, treat it as a direct prompt
        else:
            prompt_text = args.prompt
    else:
        # If no prompt is provided, read from stdin
        print("Enter software spec prompt (Ctrl+D to finish):")
        prompt_text = sys.stdin.read()
    
    return args, prompt_text


def extract_code_from_response(response: str) -> str:
    """Extract code from the response that is enclosed in ```python and ``` markers.
    If no markers are found, clean the response as best as possible.
    Also removes notes, explanations, and trailing backticks that might be included.
    """
    # Check if the response starts with ```python and remove it if present
    if response.strip().startswith('```python'):
        response = response.strip()[len('```python'):].strip()
        # Also check if it ends with ``` and remove it
        if response.strip().endswith('```'):
            response = response.strip()[:-3].strip()
        # Remove any trailing notes or explanations
        response = remove_trailing_notes(response)
        return response
    
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
        code = '\n\n'.join(matches)
        # Remove any trailing notes or explanations
        code = remove_trailing_notes(code)
        return code
    
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
    cleaned_response = remove_trailing_notes(cleaned_response)
    
    print("No code blocks found. Returning cleaned response.")
    return cleaned_response


def remove_trailing_notes(code: str) -> str:
    """Remove trailing notes, explanations, and stray backticks from code."""
    # Remove any trailing backticks that might be left
    code = re.sub(r"```\s*$", "", code, flags=re.MULTILINE)
    
    # Remove common note patterns
    note_patterns = [
        r"\s*Note:.*$",
        r"\s*This code.*$",
        r"\s*The function.*$",
        r"\s*This implementation.*$",
        r"\s*The code above.*$",
        r"\s*In this implementation.*$",
        r"\s*The tests.*$"
    ]
    
    for pattern in note_patterns:
        code = re.sub(pattern, "", code, flags=re.IGNORECASE | re.DOTALL)
    
    return code.strip()


def save_implementation(code: str, filename: str) -> None:
    """Save the implementation code to a file."""
    with open(filename, 'w') as file:
        file.write(code)
    print(f"Implementation saved to {filename}")


def run_tests(implementation_file: str, prompt: str = None) -> Tuple[bool, str]:
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
            
        # Use a specific model (llama3.2:3b) to determine if tests passed
        JUDGE_MODEL = "llama3.2:3b"  # Hardcoded judge model
        trace = langfuse.trace(
            name="evaluate_tests",
            metadata={"returncode": result.returncode}
        )
        
        test_evaluation_prompt = f"""
        You are a test evaluation expert. Your task is to determine if ALL tests have passed in the provided test output.
        
        Specification:
        {prompt}
        
        Implementation:
        ```python
        {implementation_code}
        ```
        
        Test output:
        ```
        {output}
        ```
        
        Carefully analyze the test output above. Look for any indicators of test failures such as:
        - AssertionError
        - SyntaxError
        - NameError
        - Other exceptions or errors
        - Lines containing 'FAILED' or 'FAIL:'
        - Lines showing test failures or errors counts
        
        If ANY tests failed, you MUST answer '<FAILED>'.
        If ALL tests passed successfully, you MUST answer '<PASSED>'.
        
        Your answer (ONLY '<PASSED>' or '<FAILED>'):
        """
        
        generation = trace.generation(
            name="test_evaluation",
            model=JUDGE_MODEL,
            input={
                "prompt": test_evaluation_prompt,
                "test_output": output, 
                "implementation": implementation_code
            }
        )
        
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a test evaluation expert. Analyze the test output and respond with ONLY '<PASSED>' or '<FAILED>'. If ANY tests failed, respond with '<FAILED>'."},
                {"role": "user", "content": test_evaluation_prompt}
            ]
        )
        
        evaluation = response.choices[0].message.content
        
        generation.end(
            output=evaluation,
            metadata={"returncode": result.returncode}
        )
        
        # Determine if tests passed based on the model's evaluation
        all_tests_passed = "<PASSED>" in evaluation.strip()
        
        # Add the evaluation to the output
        output += "\n\nModel Evaluation (by llama3.2:3b): " + evaluation
            
        return all_tests_passed, output
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Error running tests: {str(e)}"


def load_conversation(filename: str) -> List[Dict[str, str]]:
    """Load the conversation history from file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error reading conversation file. Starting with empty conversation.")
    return []


def save_conversation(conversation: List[Dict[str, str]], filename: str) -> None:
    """Save the conversation history to a file."""
    with open(filename, 'w') as file:
        json.dump(conversation, file, indent=2)
    print(f"Conversation saved to {filename}")





def save_usage_data(usage_data: List[Dict[str, Any]], filename: Optional[str] = None) -> None:
    """Save usage data to a JSON file with computed totals."""
    if not filename:
        return  # Skip saving if no filename is provided
    
    # Compute totals
    total_tokens = sum(entry.get("usage", {}).get("total_tokens", 0) for entry in usage_data if entry.get("usage"))
    total_prompt_tokens = sum(entry.get("usage", {}).get("prompt_tokens", 0) for entry in usage_data if entry.get("usage"))
    total_completion_tokens = sum(entry.get("usage", {}).get("completion_tokens", 0) for entry in usage_data if entry.get("usage"))
    total_duration = sum(entry.get("duration_seconds", 0) for entry in usage_data)
    
    # Create a dictionary with usage data and totals
    data_with_totals = {
        "entries": usage_data,
        "totals": {
            "total_api_calls": len(usage_data),
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_duration_seconds": round(total_duration, 2)
        }
    }
    
    try:
        with open(filename, 'w') as file:
            json.dump(data_with_totals, file, indent=2)
        print(f"Usage data saved to {filename}")
    except Exception as e:
        print(f"Error saving usage data: {str(e)}")


def load_usage_data(filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load usage data from a JSON file if it exists."""
    if not filename:
        return []  # Return empty list if no filename is provided
        
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
                # Handle both old format (list) and new format (dict with entries)
                if isinstance(data, dict) and "entries" in data:
                    return data["entries"]
                elif isinstance(data, list):
                    return data
                else:
                    print(f"Warning: Unexpected format in usage data file. Starting with empty data.")
                    return []
        return []
    except Exception as e:
        print(f"Error loading usage data: {str(e)}")
        return []


def generate_implementation(prompt: str, conversation: List[Dict[str, str]], usage_data: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generate implementation based on the prompt and conversation history."""
    trace = langfuse.trace(
        name="generate_implementation",
        metadata={"conversation_length": len(conversation)}
    )
    
    # Create the system message
    system_message = "Write Python code with correct syntax and indentation. Return ONLY code, no explanations."
    
    # Create the messages list for the API call
    messages = [{"role": "system", "content": system_message}]
    
    # Add the initial prompt if conversation is empty
    if not conversation:
        messages.append({"role": "user", "content": f"Write Python code for this spec:\n{prompt}\n\nInclude:\n- Simple unittest tests\n- Minimal comments\n- Handle edge cases\n\nReturn ONLY code, no explanations."})
    else:
        # Add conversation history (limited to last 10 messages to avoid context length issues)
        for msg in conversation[-10:]:
            messages.append(msg)
    
    # Create a generation for the implementation
    generation = trace.generation(
        name="code_generation",
        model=MODEL,
        input=messages
    )
    
    # Record start time
    start_time = time.time()
    
    # Call the API
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    
    # Record end time
    end_time = time.time()
    
    implementation = response.choices[0].message.content
    code = extract_code_from_response(implementation)
    
    # Capture usage data
    usage_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": len(conversation) // 2 + 1 if conversation else 1,
        "model": MODEL,
        "duration_seconds": round(end_time - start_time, 2),
        "usage": None
    }
    
    # Add usage information if available
    if hasattr(response, 'usage') and response.usage:
        usage_entry["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Add detailed token information if available
        if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
            prompt_details = {}
            if response.usage.prompt_tokens_details.audio_tokens:
                prompt_details["audio_tokens"] = response.usage.prompt_tokens_details.audio_tokens
            if response.usage.prompt_tokens_details.cached_tokens:
                prompt_details["cached_tokens"] = response.usage.prompt_tokens_details.cached_tokens
            if prompt_details:
                usage_entry["usage"]["prompt_tokens_details"] = prompt_details
        
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            completion_details = {}
            if response.usage.completion_tokens_details.audio_tokens:
                completion_details["audio_tokens"] = response.usage.completion_tokens_details.audio_tokens
            if response.usage.completion_tokens_details.reasoning_tokens:
                completion_details["reasoning_tokens"] = response.usage.completion_tokens_details.reasoning_tokens
            if response.usage.completion_tokens_details.accepted_prediction_tokens:
                completion_details["accepted_prediction_tokens"] = response.usage.completion_tokens_details.accepted_prediction_tokens
            if response.usage.completion_tokens_details.rejected_prediction_tokens:
                completion_details["rejected_prediction_tokens"] = response.usage.completion_tokens_details.rejected_prediction_tokens
            if completion_details:
                usage_entry["usage"]["completion_tokens_details"] = completion_details
    
    # Add usage entry to the list if usage tracking is enabled
    if usage_data is not None:
        usage_data.append(usage_entry)
    
    # End the generation
    generation.end(
        output=code,
        metadata={
            "conversation_length": len(conversation) + 1,
            "usage": usage_entry["usage"] if usage_entry["usage"] else None
        }
    )
    
    return code


def main():
    """Main function to run the AI software engineering loop."""
    try:
        # Parse arguments
        args, prompt = parse_arguments()
        
        # Set configuration based on arguments
        model = args.model or MODEL
        implementation_file = args.implementation_file or IMPLEMENTATION_FILE
        conversation_file = args.conversation_file or CONVERSATION_FILE
        usage_file = args.usage_file  # This can be None
        max_iterations = args.max_iterations or MAX_ITERATIONS
        
        # Load conversation history
        conversation = load_conversation(conversation_file)
        
        # Load usage data if usage tracking is enabled
        usage_data = load_usage_data(usage_file) if usage_file else None
        
        print(f"\n🚀 Starting AI Software Engineer Loop with {model} model")
        print(f"Implementation file: {implementation_file}")
        print(f"Conversation file: {conversation_file}")
        if usage_file:
            print(f"Usage data file: {usage_file}")
        print(f"Current iteration: {len(conversation) // 2 + 1}")
        
        # Initialize variables
        all_tests_passed = False
        iteration = 0
        
        # Main loop
        while not all_tests_passed and iteration < max_iterations:
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")
            
            # Generate implementation
            print("Generating implementation...")
            implementation = generate_implementation(prompt, conversation, usage_data)
            
            # Save implementation to file
            save_implementation(implementation, implementation_file)
            
            # Save usage data after each generation if usage tracking is enabled
            if usage_data is not None:
                save_usage_data(usage_data, usage_file)
            
            # Run tests
            print("Running tests...")
            all_tests_passed, test_output = run_tests(implementation_file=implementation_file, prompt=prompt)
            
            print(f"Tests {'PASSED ✅' if all_tests_passed else 'FAILED ❌'}")
            print("Test output:")
            print("-" * 40)
            print(test_output)
            print("-" * 40)
            
            # Add to conversation history
            if not conversation:
                # First message should include the prompt
                conversation.append({"role": "user", "content": f"Implement the following specification:\n\n{prompt}"})
            conversation.append({"role": "assistant", "content": implementation})
            conversation.append({"role": "user", "content": f"Test results:\n{test_output}\n\nPlease fix any issues and provide an improved implementation."})
            
            # Save conversation
            save_conversation(conversation, conversation_file)
            
            if all_tests_passed:
                print("\n🎉 All tests passed! Final implementation is ready.")
                break
        
        # Print summary
        print("\n📊 AI Engineer Loop Summary:")
        print(f"- Model: {model}")
        print(f"- Iterations completed: {iteration}/{max_iterations}")
        print(f"- Tests passed: {'Yes ✅' if all_tests_passed else 'No ❌'}")
        
        # Print usage summary if usage tracking is enabled
        if usage_data and usage_file:
            # Save the final usage data to get the updated totals
            save_usage_data(usage_data, usage_file)
            
            # Load the saved file to get the totals
            try:
                with open(usage_file, 'r') as file:
                    data = json.load(file)
                    if isinstance(data, dict) and "totals" in data:
                        totals = data["totals"]
                        
                        print("\n📈 Usage Summary:")
                        print(f"- Total API calls: {totals.get('total_api_calls', len(usage_data))}")
                        print(f"- Total tokens used: {totals.get('total_tokens', 0)}")
                        print(f"- Total prompt tokens: {totals.get('total_prompt_tokens', 0)}")
                        print(f"- Total completion tokens: {totals.get('total_completion_tokens', 0)}")
                        print(f"- Total duration: {totals.get('total_duration_seconds', 0)} seconds")
            except Exception as e:
                # Fallback to calculating totals directly if there's an issue with the file
                total_tokens = sum(entry.get("usage", {}).get("total_tokens", 0) for entry in usage_data if entry.get("usage"))
                total_prompt_tokens = sum(entry.get("usage", {}).get("prompt_tokens", 0) for entry in usage_data if entry.get("usage"))
                total_completion_tokens = sum(entry.get("usage", {}).get("completion_tokens", 0) for entry in usage_data if entry.get("usage"))
                
                print("\n📈 Usage Summary:")
                print(f"- Total API calls: {len(usage_data)}")
                print(f"- Total tokens used: {total_tokens}")
                print(f"- Total prompt tokens: {total_prompt_tokens}")
                print(f"- Total completion tokens: {total_completion_tokens}")
        
        if iteration >= max_iterations and not all_tests_passed:
            print(f"\n⚠️ Reached maximum iterations ({max_iterations}). Stopping.")
        
        print(f"\nFinal implementation saved to {implementation_file}")
        print(f"Conversation saved to {conversation_file}")
        if usage_file:
            print(f"Usage data saved to {usage_file}")
        
        # Return success status for the model evaluator
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Error in AI Engineer Loop: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    # Exit with appropriate code for the model evaluator
    sys.exit(0 if success else 1)
