import unittest
import os
import tempfile
import shutil
from ai_engineer import identify_and_install_libraries, run_with_venv

class TestLibraryIdentification(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up test files
        shutil.rmtree(self.test_dir)
    
    def test_identify_libraries_from_imports(self):
        """Test that libraries are correctly identified from import statements."""
        code = """
import flask
from flask import Flask, request, jsonify
import random
import os
import sys

app = Flask(__name__)

@app.route('/random', methods=['GET'])
def get_random_numbers():
    count = request.args.get('count', default=10, type=int)
    min_val = request.args.get('min', default=1, type=int)
    max_val = request.args.get('max', default=100, type=int)
    
    if count <= 0:
        return jsonify({"error": "Count must be positive"}), 400
    
    if min_val >= max_val:
        return jsonify({"error": "Min must be less than max"}), 400
    
    numbers = [random.randint(min_val, max_val) for _ in range(count)]
    return jsonify(numbers)

if __name__ == '__main__':
    app.run(debug=True)
"""
        # Call the function
        success, output = identify_and_install_libraries(code, "", self.test_dir)
        
        # Check that flask was identified
        self.assertTrue("flask" in output.lower(), f"Flask should be identified in: {output}")
    
    def test_identify_libraries_from_pip_install(self):
        """Test that libraries are correctly identified from pip install commands."""
        code = """
# First, make sure to install the required packages:
# pip install flask pytest
import flask
from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route('/random', methods=['GET'])
def get_random_numbers():
    count = request.args.get('count', default=10, type=int)
    min_val = request.args.get('min', default=1, type=int)
    max_val = request.args.get('max', default=100, type=int)
    
    numbers = [random.randint(min_val, max_val) for _ in range(count)]
    return jsonify(numbers)

if __name__ == '__main__':
    app.run(debug=True)
"""
        # Call the function
        success, output = identify_and_install_libraries(code, "", self.test_dir)
        
        # Check that flask and pytest were identified
        self.assertTrue("flask" in output.lower(), f"Flask should be identified in: {output}")
        self.assertTrue("pytest" in output.lower(), f"Pytest should be identified in: {output}")
    
    def test_identify_libraries_from_test_output(self):
        """Test that libraries are correctly identified from test output."""
        code = """
import random
import unittest

def generate_random_numbers(count=10, min_val=1, max_val=100):
    return [random.randint(min_val, max_val) for _ in range(count)]

class TestRandomNumbers(unittest.TestCase):
    def test_generate_random_numbers(self):
        numbers = generate_random_numbers()
        self.assertEqual(len(numbers), 10)
        
if __name__ == '__main__':
    unittest.main()
"""
        test_output = """
Error: ModuleNotFoundError: No module named 'flask'
Hint: You need to install Flask to run this application.
Try: pip install flask
"""
        # Call the function
        success, output = identify_and_install_libraries(code, test_output, self.test_dir)
        
        # Check that flask was identified from the test output
        self.assertTrue("flask" in output.lower(), f"Flask should be identified in: {output}")

if __name__ == '__main__':
    unittest.main()
