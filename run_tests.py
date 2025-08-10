#!/usr/bin/env python3
"""
Test runner for WordatroCheater.
Run this script to execute all unit tests.
"""

import sys
import os
import subprocess
import unittest

def run_tests():
    """Run all tests and display results."""
    print("Running WordatroCheater Test Suite")
    print("=" * 50)
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Display summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        return 1

def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"Running specific test: {test_name}")
    print("=" * 50)
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run specific test
    from test_wordatro import TestWordatroCheater, TestWordatroIntegration
    
    # Find the test method
    test_method = None
    for attr_name in dir(TestWordatroCheater):
        if attr_name == test_name:
            test_method = getattr(TestWordatroCheater, attr_name)
            break
    
    if not test_method:
        for attr_name in dir(TestWordatroIntegration):
            if attr_name == test_name:
                test_method = getattr(TestWordatroIntegration, attr_name)
                break
    
    if test_method:
        # Create test suite with just this test
        suite = unittest.TestSuite()
        suite.addTest(TestWordatroCheater(test_name))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("Test passed!")
            return 0
        else:
            print("Test failed!")
            return 1
    else:
        print(f"Test method '{test_name}' not found")
        print("Available tests:")
        for attr_name in dir(TestWordatroCheater):
            if attr_name.startswith('test_'):
                print(f"  - {attr_name}")
        return 1

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("WordatroCheater Test Runner")
            print("\nUsage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py test_name    # Run specific test")
            print("  python run_tests.py --help       # Show this help")
            print("\nAvailable tests:")
            from test_wordatro import TestWordatroCheater, TestWordatroIntegration
            for attr_name in dir(TestWordatroCheater):
                if attr_name.startswith('test_'):
                    print(f"  - {attr_name}")
            return 0
        else:
            # Run specific test
            return run_specific_test(sys.argv[1])
    else:
        # Run all tests
        return run_tests()

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
