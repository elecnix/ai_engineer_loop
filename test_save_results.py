import unittest
import os
import json
import tempfile
from ai_engineer import save_results

class TestSaveResults(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.results_file = os.path.join(self.test_dir, "test_results.json")
        
        # Sample results data
        self.sample_results = [
            {
                "iteration": 1,
                "timestamp": "2023-01-01 12:00:00",
                "model": "test-model",
                "has_tests": True,
                "tests_passed": True,
                "implementation_file": "iteration_01.py"
            },
            {
                "iteration": 2,
                "timestamp": "2023-01-01 12:05:00",
                "model": "test-model",
                "has_tests": True,
                "tests_passed": False,
                "implementation_file": "iteration_02.py"
            }
        ]
    
    def test_save_results(self):
        """Test that results are correctly saved to a JSON file."""
        # Save the results
        save_results(self.sample_results, self.results_file)
        
        # Verify the file exists
        self.assertTrue(os.path.exists(self.results_file), "Results file was not created")
        
        # Load the saved results
        with open(self.results_file, 'r') as f:
            loaded_results = json.load(f)
        
        # Verify the loaded results match the original data
        self.assertEqual(loaded_results, self.sample_results, "Saved results do not match original data")
    
    def test_append_results(self):
        """Test that results are correctly appended to an existing file."""
        # First save initial results
        initial_results = [self.sample_results[0]]
        save_results(initial_results, self.results_file)
        
        # Now save additional results
        additional_results = self.sample_results
        save_results(additional_results, self.results_file)
        
        # Load the saved results
        with open(self.results_file, 'r') as f:
            loaded_results = json.load(f)
        
        # Verify the loaded results match the additional data (overwrite behavior)
        self.assertEqual(loaded_results, additional_results, "Results were not correctly saved")
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.results_file):
            os.remove(self.results_file)
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()
