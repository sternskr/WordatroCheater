#!/usr/bin/env python3
"""
Test script for WordatroCheater utility
Demonstrates functionality with example inputs
"""

from wordatro import WordatroCheater

def test_wordatro():
    """Test the WordatroCheater with various inputs."""
    
    print("=== WordatroCheater Test Suite ===\n")
    
    # Initialize the cheater
    cheater = WordatroCheater()
    
    # Test cases
    test_cases = [
        "ABCDEFGHI3",      # 9 letters with 3 exchanges
        "SCRABBLE2",       # Word letters with 2 exchanges  
        "QU*ZZ*XY1",       # High-value letters with wildcards
        "GAMEWORD0",       # No exchanges
        "ABC*EFG**2",      # Multiple wildcards
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i} {'='*20}")
        print(f"Input: {test_input}")
        
        try:
            results = cheater.find_best_words(test_input)
            cheater.print_results(results)
            
        except Exception as e:
            print(f"Error processing {test_input}: {e}")
        
        print("\n" + "-"*60)
    
    # Demonstrate individual functions
    print(f"\n{'='*20} INDIVIDUAL FUNCTION TESTS {'='*20}")
    
    # Test scoring
    test_words = ["QUIZ", "EXCELLENT", "WORDGAMES", "SCRABBLE"]
    print("\nWord scoring examples:")
    for word in test_words:
        score = cheater.calculate_word_score(word)
        letter_sum = sum(cheater.letter_scores.get(letter, 10) for letter in word)
        print(f"{word:<12} | Score: {score:4d} | ({letter_sum} × {len(word)})")
    
    # Test input parsing
    print("\nInput parsing examples:")
    test_inputs = ["ABCDEFGHI3", "XYZ*123", "HELLO0", "TEST"]
    for inp in test_inputs:
        letters, exchanges = cheater.parse_input(inp)
        print(f"{inp:<12} → Letters: {letters}, Exchanges: {exchanges}")
    
    print(f"\n{'='*60}")
    print("Test completed! The utility is ready to use.")
    print("Run 'python wordatro.py' for interactive mode.")

if __name__ == "__main__":
    test_wordatro() 