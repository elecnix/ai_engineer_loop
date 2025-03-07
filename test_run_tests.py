#!/usr/bin/env python3
"""
Unit tests for the run_tests function in ai_engineer.py
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the function to test
from ai_engineer import run_tests, run_with_venv, identify_and_install_libraries

class TestRunTests(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple Flask implementation file for testing
        self.implementation_file = os.path.join(self.test_dir, "test_implementation.py")
        with open(self.implementation_file, "w") as f:
            f.write("""
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

@app.route('/random', methods=['GET'])
def get_random_numbers():
    count = int(request.args.get('count', 10))
    min_num = int(request.args.get('min', 1))
    max_num = int(request.args.get('max', 100))
    
    random_numbers = [random.randint(min_num, max_num) for _ in range(count)]
    return jsonify(random_numbers)

if __name__ == '__main__':
    # Simple test
    print("Running tests...")
    print("All tests passed!")
    app.run(debug=True)
""")

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    @patch('subprocess.run')
    def test_run_tests_system_python(self, mock_run):
        # Mock the subprocess.run to return a successful result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Running tests...\nAll tests passed!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock check_for_tests to return True
        with patch('ai_engineer.check_for_tests', return_value=(True, "Tests found")):
            # Mock the OpenAI client to return a successful evaluation
            with patch('ai_engineer.client.chat.completions.create') as mock_create:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "<PASSED>"
                mock_create.return_value = mock_response
                
                # Run the test
                success, output = run_tests(self.implementation_file, "Test prompt")
                
                # Check the results
                self.assertTrue(success)
                self.assertIn("All tests passed!", output)

    @patch('ai_engineer.run_with_venv')
    def test_run_tests_with_venv(self, mock_run_with_venv):
        # Create a mock venv directory
        venv_dir = os.path.join(self.test_dir, "venv")
        os.makedirs(venv_dir)
        
        # Mock run_with_venv to return a successful result
        mock_run_with_venv.return_value = (True, "Running tests...\nAll tests passed!")
        
        # Mock check_for_tests to return True
        with patch('ai_engineer.check_for_tests', return_value=(True, "Tests found")):
            # Mock the OpenAI client to return a successful evaluation
            with patch('ai_engineer.client.chat.completions.create') as mock_create:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "<PASSED>"
                mock_create.return_value = mock_response
                
                # Mock langfuse trace to avoid errors
                with patch('ai_engineer.langfuse.trace') as mock_trace:
                    mock_generation = MagicMock()
                    mock_trace.return_value.generation.return_value = mock_generation
                    
                    # Run the test
                    success, output = run_tests(self.implementation_file, "Test prompt")
                    
                    # Check the results
                    self.assertTrue(success)
                    self.assertIn("All tests passed!", output)
                    
                    # Verify that run_with_venv was called with the correct arguments
                    mock_run_with_venv.assert_called_once_with(self.implementation_file, os.path.dirname(self.implementation_file))

    def test_run_with_venv(self):
        # Mock the subprocess calls
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Test output"
            mock_run.return_value.stderr = ""
            
            # Test running the implementation file
            success, output = run_with_venv(self.implementation_file, self.test_dir)
            
            # The function should call uv run directly with the basename of the file
            mock_run.assert_called_with(
                f"uv run {os.path.basename(self.implementation_file)}",
                shell=True,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # The test is now complete since we only need to check if uv run is called correctly
            
            # Reset the mock to check that subsequent calls use the same command
            mock_run.reset_mock()
            
            # Test the function again
            success, output = run_with_venv(self.implementation_file, self.test_dir)
            
            # Check that the correct command was run
            mock_run.assert_called_with(
                f"uv run {os.path.basename(self.implementation_file)}",
                shell=True,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

    def test_identify_and_install_libraries(self):
        # Test code that requires Flask
        implementation_code = """
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

@app.route('/random', methods=['GET'])
def get_random_numbers():
    count = int(request.args.get('count', 10))
    min_num = int(request.args.get('min', 1))
    max_num = int(request.args.get('max', 100))
    
    random_numbers = [random.randint(min_num, max_num) for _ in range(count)]
    return jsonify(random_numbers)
"""
        test_output = "ModuleNotFoundError: No module named 'flask'"
        
        # Mock subprocess calls
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Successfully installed flask"
            
            # Mock os.path.exists to simulate venv creation
            with patch('os.path.exists', return_value=True):
                # Mock shutil.rmtree to avoid actually removing directories
                with patch('shutil.rmtree'):
                    # Test the function
                    success, output = identify_and_install_libraries(implementation_code, test_output, self.test_dir)
                    
                    # Check that the function identified Flask as a dependency
                    self.assertTrue(success)
                    self.assertIn("flask", output.lower())


if __name__ == "__main__":
    unittest.main()
