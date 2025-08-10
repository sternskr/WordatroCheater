#!/usr/bin/env python3
"""
Main interface class that provides both Wordatro and Wordle functionality.
"""

from typing import Dict, List, Set

from wordatro_game import WordatroGame
from wordle_game import WordleGame


class WordatroCheater:
    """Main interface class that provides both Wordatro and Wordle functionality."""

    def __init__(self, dictionary_file: str = "dictionary.txt", force_regenerate: bool = False):
        """Initialize the cheater with both game modes."""
        # Both game modes will automatically share the same DictionaryManager instance
        # thanks to the singleton pattern
        self.wordatro = WordatroGame(dictionary_file, force_regenerate)
        self.wordle = WordleGame(dictionary_file, force_regenerate)

        # For backward compatibility, expose common attributes
        self.dictionary = self.wordatro.dictionary
        self.letter_scores = self.wordatro.letter_scores
        self.wordle_letter_frequencies = self.wordatro.wordle_letter_frequencies

    def find_best_words(self, input_str: str, sort_by: str = 'score') -> Dict:
        """Find best words, automatically detecting game mode."""
        if ':' in input_str:
            # Wordle mode
            return self.wordle.find_best_words(input_str, sort_by)
        else:
            # Wordatro mode
            return self.wordatro.find_best_words(input_str, sort_by)

    def parse_input(self, input_str: str):
        """Parse input, automatically detecting game mode."""
        if ':' in input_str:
            return self.wordle.parse_wordle_input(input_str)
        else:
            return self.wordatro.parse_input(input_str)
    
    def parse_wordle_input(self, input_str: str):
        """Parse Wordle input specifically."""
        return self.wordle.parse_wordle_input(input_str)

    # Delegate other methods to appropriate game class
    def calculate_word_score(self, word: str) -> int:
        return self.wordatro.calculate_word_score(word)

    def calculate_wordle_score(self, word: str) -> float:
        return self.wordatro.calculate_wordle_score(word)

    def calculate_wordle_commonality_score(self, word: str) -> float:
        return self.wordatro.calculate_wordle_commonality_score(word)

    def sort_words_by_criteria(self, words: Set[str], sort_by: str = 'score', game_mode: str = 'wordatro'):
        return self.wordatro.sort_words_by_criteria(words, sort_by, game_mode)

    def generate_word_combinations(self, letters: List[str], target_length: int = None,
                                 required_letters: List[str] = None,
                                 positional_letters: Dict[int, str] = None):
        return self.wordatro.generate_word_combinations(letters, target_length, required_letters, positional_letters)

    def find_words_with_exclusions(self, letters: List[str], target_length: int = None,
                                 required_letters: List[str] = None,
                                 positional_letters: Dict[int, str] = None,
                                 excluded_letters: Set[str] = None):
        return self.wordle.find_words_with_exclusions(letters, target_length, required_letters, positional_letters, excluded_letters)

    def find_exchange_opportunities(self, letters: List[str], exchanges_remaining: int,
                                  required_letters: List[str] = None, target_length: int = None,
                                  positional_letters: Dict[int, str] = None,
                                  existing_words: Set[str] = None):
        return self.wordatro.find_exchange_opportunities(letters, exchanges_remaining, required_letters, target_length, positional_letters, existing_words)

    def regenerate_cache(self):
        """Regenerate cache for both game modes."""
        self.wordatro.regenerate_cache()
        self.wordle.regenerate_cache()
        self.dictionary = self.wordatro.dictionary

    def save_cache(self):
        """Save both game modes' caches."""
        print("Saving caches...")
        self.wordatro.save_cache()
        self.wordle.save_cache()

    def show_cache_stats(self):
        """Show cache statistics for both game modes."""
        print("=== Wordatro Cache Stats ===")
        self.wordatro.dict_manager.show_cache_stats()
        print("\n=== Wordle Cache Stats ===")
        self.wordle.dict_manager.show_cache_stats()

    def reset_wordle_game(self):
        """Reset Wordle game state (clear accumulated information)."""
        self.wordle.reset_accumulated_info()
        print("🔄 Wordle game state reset! All previous guesses cleared.")

    def show_wordle_state(self):
        """Show current Wordle game state."""
        self.wordle.show_accumulated_info()

    def print_results(self, results: Dict):
        """Print results in a formatted way."""
        if not results:
            print("No results to display.")
            return

        # Determine game mode
        game_mode = results.get('game_mode', 'unknown')

        if game_mode == 'wordle':
            self._print_wordle_results(results)
        else:
            self._print_wordatro_results(results)

    def _print_wordle_results(self, results: Dict):
        """Print Wordle results."""
        print(f"\n🎯 Wordle: {results['input']}")
        print(f"📊 Found {results['total_words_found']} words | Sort: {results['sort_by']}")

        if results['top_words']:
            print(f"\n🏆 Top {len(results['top_words'])} words:")
            for i, (word, score) in enumerate(results['top_words'], 1):
                print(f"  {i:2d}. {word:<8} ({score:.3f})")

        # Show accumulated information
        if results.get('accumulated_info'):
            print(f"\n{results['accumulated_info']}")

    def _print_wordatro_results(self, results: Dict):
        """Print Wordatro results."""
        print(f"\n🎲 Wordatro: {results['input']}")
        print(f"📊 Found {results['total_words_found']} words | Sort: {results['sort_by']}")
        print(f"🔄 Exchanges: {results['exchanges_remaining']}")

        if results['top_words']:
            print(f"\n🏆 Top {len(results['top_words'])} words:")
            for i, (word, score) in enumerate(results['top_words'], 1):
                print(f"  {i:2d}. {word:<8} ({score})")

        if results.get('exchange_suggestions'):
            print(f"\n💡 Exchange suggestions:")
            for letter, potential, replacements in results['exchange_suggestions'][:5]:
                print(f"  {letter} ({potential}) → {', '.join(replacements)}")

        # Show constraints
        if results.get('positional_letters'):
            print(f"\n📍 Positional constraints:")
            for pos, letter in sorted(results['positional_letters'].items()):
                print(f"  Pos {pos+1}: {letter}")

    def _format_for_terminal_width(self, text: str, width: int) -> str:
        """Format text to fit terminal width."""
        if len(text) <= width:
            return text

        # Try to break at word boundaries
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)
