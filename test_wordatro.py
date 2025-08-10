#!/usr/bin/env python3
"""
Test script for WordatroCheater utility
Demonstrates functionality with example inputs
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from wordatro import WordatroCheater

class TestWordatroCheater(unittest.TestCase):
    """Test cases for WordatroCheater class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_dict_file = os.path.join(self.test_dir, "test_dict.txt")
        self.test_cache_file = os.path.join(self.test_dir, "test_dict.word")
        
        # Create a minimal test dictionary
        test_words = [
            "STARE", "STARS", "START", "HELLO", "WORLD", 
            "PYTHON", "SCRABBLE", "GAME", "QUICK", "BROWN", 
            "FOX", "JUMPS", "LAZY", "DOG", "CAT", 
            "BAT", "RAT", "SAT", "EAT", "ATE", 
            "TEA", "SEA", "BEE", "THE", "AND"
        ]
        
        with open(self.test_dict_file, 'w') as f:
            for word in test_words:
                f.write(word + '\n')
        
        # Initialize the cheater with test dictionary
        self.cheater = WordatroCheater(dictionary_file=self.test_dict_file, force_regenerate=True)
    
    def tearDown(self):
        """Clean up after each test."""
        # Don't save cache after every test - it's already saved automatically
        pass
    
    def test_initialization(self):
        """Test that WordatroCheater initializes correctly."""
        self.assertIsNotNone(self.cheater.dictionary)
        self.assertIsNotNone(self.cheater.letter_scores)
        self.assertIsNotNone(self.cheater.wordle_letter_frequencies)
        self.assertEqual(len(self.cheater.dictionary), 25)  # 25 unique words
    
    def test_letter_scores(self):
        """Test letter scoring system."""
        self.assertEqual(self.cheater.letter_scores['E'], 1)  # Common letter
        self.assertEqual(self.cheater.letter_scores['Q'], 10)  # Rare letter
        self.assertEqual(self.cheater.letter_scores['Z'], 10)  # Rare letter
    
    def test_wordle_letter_frequencies(self):
        """Test Wordle letter frequency data is computed from dictionary with sane properties."""
        freqs = self.cheater.wordle_letter_frequencies
        # All A-Z present
        self.assertTrue(all(chr(c) in freqs for c in range(ord('A'), ord('Z')+1)))
        # Sum should be approximately 1.0
        self.assertAlmostEqual(sum(freqs.values()), 1.0, places=3)
        # For the test dictionary, vowels should be relatively common
        self.assertGreater(freqs['E'], freqs['Z'])
        self.assertGreater(freqs['A'], freqs['J'])
    
    def test_calculate_word_score(self):
        """Test Scrabble-style word scoring."""
        # STARE: S(1) + T(1) + A(1) + R(1) + E(1) = 5
        self.assertEqual(self.cheater.calculate_word_score("STARE"), 5)
        # QUICK: Q(10) + U(1) + I(1) + C(3) + K(5) = 20
        self.assertEqual(self.cheater.calculate_word_score("QUICK"), 20)
    
    def test_calculate_wordle_score(self):
        """Test Wordle-specific scoring."""
        score = self.cheater.calculate_wordle_score("STARE")
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        
        # Words with common letters should score higher
        stare_score = self.cheater.calculate_wordle_score("STARE")
        quick_score = self.cheater.calculate_wordle_score("QUICK")
        self.assertGreater(stare_score, quick_score)  # STARE has more common letters
    
    def test_calculate_wordle_commonality_score(self):
        """Test enhanced Wordle commonality scoring."""
        score = self.cheater.calculate_wordle_commonality_score("STARE")
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        
        # Common letters should score higher than rare letters
        stare_score = self.cheater.calculate_wordle_commonality_score("STARE")
        quick_score = self.cheater.calculate_wordle_commonality_score("QUICK")
        self.assertGreater(stare_score, quick_score)
    
    def test_parse_input_wordatro_mode(self):
        """Test parsing of traditional Wordatro input."""
        letters, exchanges, required, length, positional = self.cheater.parse_input("STARE3")
        self.assertEqual(letters, ['S', 'T', 'A', 'R', 'E'])
        self.assertEqual(exchanges, 3)
        self.assertEqual(required, ['S', 'T', 'A', 'R', 'E'])
        self.assertIsNone(length)
        self.assertEqual(positional, {})
        
        # Test with length specification
        letters, exchanges, required, length, positional = self.cheater.parse_input("5STARE3")
        self.assertEqual(length, 5)
        self.assertEqual(letters, ['S', 'T', 'A', 'R', 'E'])
    
    def test_parse_input_with_positional_letters(self):
        """Test parsing with positional letter constraints."""
        letters, exchanges, required, length, positional = self.cheater.parse_input("S.TA.RE3")
        self.assertEqual(letters, ['S', 'T', 'A', 'R', 'E'])
        self.assertEqual(positional, {1: 'T', 3: 'R'})
        self.assertEqual(required, ['S', 'T', 'A', 'R', 'E'])
    
    def test_parse_wordle_input(self):
        """Test Wordle input parsing."""
        result = self.cheater.parse_wordle_input("STARE:GYGYY")
        letters, exchanges, required, length, positional, excluded, game_mode = result
        
        self.assertEqual(letters, ['S', 'T', 'A', 'R', 'E'])
        self.assertEqual(exchanges, 0)
        self.assertEqual(required, ['S', 'T', 'A', 'R', 'E'])
        self.assertIsNone(length)
        self.assertEqual(positional, {0: 'S', 2: 'A'})  # Only green letters
        self.assertEqual(excluded, set())
        self.assertEqual(game_mode, 'wordle')
    
    def test_parse_wordle_input_with_exclusions(self):
        """Test Wordle input parsing with excluded letters."""
        result = self.cheater.parse_wordle_input("STARE:BBGYY")
        letters, exchanges, required, length, positional, excluded, game_mode = result
        
        self.assertEqual(letters, ['A', 'R', 'E'])
        self.assertEqual(required, ['A', 'R', 'E'])
        self.assertEqual(positional, {2: 'A'})
        self.assertEqual(excluded, {'S', 'T'})
        self.assertEqual(game_mode, 'wordle')
    
    def test_parse_wordle_input_with_length(self):
        """Test Wordle input parsing with length specification."""
        result = self.cheater.parse_wordle_input("5STARE:GYGYY")
        letters, exchanges, required, length, positional, excluded, game_mode = result
        
        self.assertEqual(length, 5)
        self.assertEqual(letters, ['S', 'T', 'A', 'R', 'E'])
        self.assertEqual(game_mode, 'wordle')
    
    def test_sort_words_by_criteria(self):
        """Test different sorting criteria."""
        words = {"STARE", "QUICK", "HELLO", "WORLD"}
        
        # Test score sorting
        scored = self.cheater.sort_words_by_criteria(words, 'score', 'wordatro')
        self.assertEqual(len(scored), 4)
        self.assertEqual(scored[0][0], "QUICK")  # Highest Scrabble score
        
        # Test length sorting
        length_sorted = self.cheater.sort_words_by_criteria(words, 'length', 'wordatro')
        self.assertEqual(length_sorted[0][0], "QUICK")  # Longest word
        
        # Test alphabetical sorting
        alpha_sorted = self.cheater.sort_words_by_criteria(words, 'alphabetical', 'wordatro')
        self.assertEqual(alpha_sorted[0][0], "HELLO")  # First alphabetically
        
        # Test commonality sorting
        common_sorted = self.cheater.sort_words_by_criteria(words, 'commonality', 'wordatro')
        self.assertEqual(len(common_sorted), 4)
    
    def test_sort_words_by_criteria_wordle_mode(self):
        """Test sorting in Wordle mode."""
        words = {"STARE", "QUICK", "HELLO", "WORLD"}
        
        # Test Wordle commonality sorting
        common_sorted = self.cheater.sort_words_by_criteria(words, 'commonality', 'wordle')
        self.assertEqual(len(common_sorted), 4)
        
        # STARE should score higher than QUICK in Wordle mode due to common letters
        stare_score = None
        quick_score = None
        for word, score in common_sorted:
            if word == "STARE":
                stare_score = score
            elif word == "QUICK":
                quick_score = score
        
        if stare_score and quick_score:
            self.assertGreater(stare_score, quick_score)
    
    def test_find_best_words_wordatro_mode(self):
        """Test finding best words in Wordatro mode."""
        results = self.cheater.find_best_words("STARE3", "score")
        
        self.assertEqual(results['input'], "STARE3")
        self.assertEqual(results['game_mode'], 'wordatro')
        self.assertEqual(results['sort_by'], 'score')
        self.assertGreater(results['total_words_found'], 0)
        self.assertIn('top_words', results)
        self.assertIn('exchange_suggestions', results)
    
    def test_find_best_words_wordle_mode(self):
        """Test finding best words in Wordle mode."""
        results = self.cheater.find_best_words("STARE:GYGYY", "commonality")
        
        self.assertEqual(results['input'], "STARE:GYGYY")
        self.assertEqual(results['game_mode'], 'wordle')
        self.assertEqual(results['sort_by'], 'commonality')
        self.assertGreater(results['total_words_found'], 0)
        self.assertNotIn('exchange_suggestions', results)  # No exchanges in Wordle mode
    
    def test_error_handling_invalid_wordle_format(self):
        """Test error handling for invalid Wordle input."""
        with self.assertRaises(ValueError):
            self.cheater.parse_wordle_input("STARE:GYG")  # Mismatched lengths
        
        with self.assertRaises(ValueError):
            self.cheater.parse_wordle_input("STARE:XYZ")  # Invalid feedback characters
    
    def test_cache_functionality(self):
        """Test cache saving and loading."""
        # Save cache
        self.cheater.save_cache()
        
        # Verify cache file exists
        self.assertTrue(os.path.exists(self.test_cache_file))
        
        # Create new instance that should load from cache
        new_cheater = WordatroCheater(dictionary_file=self.test_dict_file, force_regenerate=False)
        
        # Should have same dictionary
        self.assertEqual(len(new_cheater.dictionary), len(self.cheater.dictionary))
    
    def test_wildcard_functionality(self):
        """Test wildcard letter handling."""
        # Test with wildcards
        letters, exchanges, required, length, positional = self.cheater.parse_input("ST*RE3")
        self.assertEqual(letters, ['S', 'T', '*', 'R', 'E'])
        self.assertEqual(exchanges, 3)
    
    def test_required_letters_filtering(self):
        """Test that required letters are properly enforced."""
        words = self.cheater.generate_word_combinations(
            ['S', 'T', 'A', 'R', 'E'], 
            required_letters=['S', 'T']
        )
        
        # All words should contain S and T
        for word in words:
            self.assertIn('S', word)
            self.assertIn('T', word)
    
    def test_positional_letters_filtering(self):
        """Test that positional letters are properly enforced."""
        words = self.cheater.generate_word_combinations(
            ['S', 'T', 'A', 'R', 'E'], 
            positional_letters={0: 'S', 2: 'A'}
        )
        
        # All words should have S at position 0 and A at position 2
        for word in words:
            self.assertEqual(word[0], 'S')
            self.assertEqual(word[2], 'A')

class TestWordatroIntegration(unittest.TestCase):
    """Integration tests for WordatroCheater."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_dict_file = os.path.join(self.test_dir, "integration_dict.txt")
        
        # Create a larger test dictionary for integration tests
        test_words = [
            "STARE", "STARS", "START", "STARE", "STARE",
            "HELLO", "WORLD", "PYTHON", "SCRABBLE", "GAME",
            "QUICK", "BROWN", "FOX", "JUMPS", "LAZY",
            "DOG", "CAT", "BAT", "RAT", "SAT",
            "EAT", "ATE", "TEA", "SEA", "BEE",
            "THE", "AND", "FOR", "ARE", "BUT",
            "NOT", "ALL", "CAN", "HER", "WAS",
            "ONE", "OUR", "OUT", "DAY", "GET",
            "HAS", "HIM", "HIS", "HOW", "MAN"
        ]
        
        with open(self.test_dict_file, 'w') as f:
            for word in test_words:
                f.write(word + '\n')
        
        self.cheater = WordatroCheater(dictionary_file=self.test_dict_file, force_regenerate=True)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir)
    
    def test_full_wordle_workflow(self):
        """Test a complete Wordle workflow."""
        # First guess: STARE
        results1 = self.cheater.find_best_words("STARE:BBGYY", "commonality")
        self.assertEqual(results1['game_mode'], 'wordle')
        self.assertGreater(results1['total_words_found'], 0)
        
        # Second guess: HELLO (using feedback from STARE)
        results2 = self.cheater.find_best_words("HELLO:GYBYY", "commonality")
        self.assertEqual(results2['game_mode'], 'wordle')
        self.assertGreater(results2['total_words_found'], 0)
    
    def test_full_wordatro_workflow(self):
        """Test a complete Wordatro workflow."""
        # Find best words with more letters so not all are required
        results = self.cheater.find_best_words("STARETHE3", "score")
        self.assertEqual(results['game_mode'], 'wordatro')
        self.assertGreater(results['total_words_found'], 0)
        
        # Get exchange suggestions
        self.assertIn('exchange_suggestions', results)
        self.assertGreater(len(results['exchange_suggestions']), 0)
    
    def test_sorting_options_workflow(self):
        """Test that all sorting options work correctly."""
        words = {"STARE", "QUICK", "HELLO", "WORLD"}
        
        sort_options = ['score', 'length', 'alphabetical', 'commonality']
        
        for sort_by in sort_options:
            sorted_words = self.cheater.sort_words_by_criteria(words, sort_by, 'wordatro')
            self.assertEqual(len(sorted_words), 4)
            self.assertIsInstance(sorted_words[0], tuple)
            self.assertIsInstance(sorted_words[0][0], str)
            self.assertIsInstance(sorted_words[0][1], (int, float))

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 