#!/usr/bin/env python3
"""
Wordatro-specific game logic.
"""

import re
from typing import List, Tuple, Dict, Set

from base_game import BaseWordGame


class WordatroGame(BaseWordGame):
    """Wordatro-specific game logic and functionality."""

    def parse_input(self, input_str: str) -> Tuple[List[str], int, List[str], int, Dict[int, str]]:
        """Parse traditional Wordatro input format."""
        target_length = None
        if input_str and input_str[0].isdigit():
            target_length = int(input_str[0])
            input_str = input_str[1:]

        exchanges = 0
        m = re.search(r"(\d+)$", input_str)
        if m:
            exchanges = int(m.group(1))
            letters_part = input_str[:m.start()]
        else:
            letters_part = input_str

        required_letters: List[str] = []
        all_letters: List[str] = []
        positional_letters: Dict[int, str] = {}
        current_position = 0

        i = 0
        while i < len(letters_part):
            char = letters_part[i]
            if char == '.':
                if i + 1 < len(letters_part):
                    pos_char = letters_part[i + 1]
                    if pos_char.isalpha():
                        positional_letters[current_position] = pos_char.upper()
                        all_letters.append(pos_char.upper())
                        required_letters.append(pos_char.upper())
                        i += 2
                        current_position += 1
                        continue
            elif char == '*':
                all_letters.append('*')
                i += 1
                current_position += 1
                continue
            if char.isalpha():
                all_letters.append(char.upper())
                required_letters.append(char.upper())
                current_position += 1
            i += 1

        return all_letters, exchanges, required_letters, target_length, positional_letters

    def find_exchange_opportunities(
        self,
        letters: List[str],
        exchanges_remaining: int,
        required_letters: List[str] = None,
        target_length: int = None,
        positional_letters: Dict[int, str] = None,
        existing_words: Set[str] = None,
    ):
        """Find the best letters to exchange based on current word pool."""
        if not existing_words:
            existing_words = self.generate_word_combinations(letters, target_length, required_letters, positional_letters)
        
        if not existing_words:
            return []
        
        analysis = self._analyze_letter_usage_in_words(letters, existing_words, required_letters, target_length, positional_letters)
        
        exchange_suggestions = []
        baseline_score = max(self.calculate_word_score(w) for w in existing_words)
        
        for letter, letter_analysis in analysis.items():
            if letter_analysis['exchange_potential'] > 0:
                best_replacement = self._find_best_replacement_letter(letter, letters, required_letters, target_length, positional_letters, baseline_score)
                if best_replacement:
                    exchange_suggestions.append((letter, letter_analysis['exchange_potential'], best_replacement))
        exchange_suggestions.sort(key=lambda x: x[1], reverse=True)
        return exchange_suggestions[:exchanges_remaining * 2]

    def _analyze_letter_usage_in_words(
        self,
        input_letters: List[str],
        found_words: Set[str],
        required_letters: List[str] = None,
        target_length: int = None,
        positional_letters: Dict[int, str] = None,
    ):
        from collections import Counter

        analysis = {}
        for letter in set(input_letters):
            if letter == '*':
                continue
            score_potential = sum(self.calculate_word_score(w) for w in found_words)
            words_with_letter = sum(1 for w in found_words if letter in w)
            score_contribution = sum(self.calculate_word_score(w) for w in found_words if letter in w)
            total_words = len(found_words)
            usage_percentage = (words_with_letter / total_words * 100) if total_words > 0 else 0
            score_percentage = (score_contribution / score_potential * 100) if score_potential > 0 else 0
            is_required = letter in (required_letters or [])
            is_positional = any(letter == pos_letter for pos_letter in (positional_letters or {}).values())
            exchange_potential = 0
            if not is_required and not is_positional:
                if usage_percentage < 80:  # Much more lenient threshold
                    exchange_potential = 80 - usage_percentage
                elif score_percentage < 60:  # Much more lenient threshold
                    exchange_potential = 60 - score_percentage
            analysis[letter] = {
                'letter': letter,
                'words_containing': words_with_letter,
                'total_words': total_words,
                'usage_percentage': usage_percentage,
                'score_contribution': score_contribution,
                'total_score_potential': score_potential,
                'score_percentage': score_percentage,
                'is_required': is_required,
                'is_positional': is_positional,
                'exchange_potential': exchange_potential,
                'recommendation': self._get_letter_recommendation(letter, usage_percentage, score_percentage, is_required, is_positional),
            }
        return analysis

    def _get_letter_recommendation(self, letter: str, usage_percentage: float, score_percentage: float, is_required: bool, is_positional: bool) -> str:
        if is_required or is_positional:
            return "Keep (required/positional)"
        elif usage_percentage < 20:
            return "Consider exchanging (low usage)"
        elif score_percentage < 15:
            return "Consider exchanging (low score contribution)"
        elif usage_percentage < 40:
            return "Moderate value"
        else:
            return "Keep (high value)"

    def _find_best_replacement_letter(
        self,
        target_letter: str,
        current_letters: List[str],
        required_letters: List[str],
        target_length: int,
        positional_letters: Dict[int, str],
        baseline_score: int,
    ):
        current_pool = [l for l in current_letters if l != target_letter]
        best_replacements = []
        best_improvement = 0
        test_letters = ['E', 'A', 'R', 'I', 'O', 'T', 'N', 'S', 'L', 'C']
        for replacement in test_letters:
            if replacement in current_pool:
                continue
            test_score = self._test_single_substitution(current_pool + [replacement], replacement, target_length, required_letters, positional_letters)
            if test_score > baseline_score:
                improvement = test_score - baseline_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_replacements = [replacement]
                elif improvement == best_improvement:
                    best_replacements.append(replacement)
        return best_replacements

    def _test_single_substitution(
        self,
        test_letters: List[str],
        substitute_letter: str,
        target_length: int,
        required_letters: List[str],
        positional_letters: Dict[int, str],
    ) -> int:
        test_words = self.generate_word_combinations(test_letters, target_length, required_letters, positional_letters)
        if not test_words:
            return 0
        return max(self.calculate_word_score(w) for w in test_words)

    def find_best_words(self, input_str: str, sort_by: str = 'score'):
        letters, exchanges_remaining, required_letters, target_length, positional_letters = self.parse_input(input_str)
        valid_words = self.generate_word_combinations(letters, target_length, required_letters, positional_letters)
        if not valid_words:
            return {
                'input': input_str,
                'parsed_letters': letters,
                'required_letters': required_letters,
                'target_length': target_length,
                'positional_letters': positional_letters,
                'exchanges_remaining': exchanges_remaining,
                'game_mode': 'wordatro',
                'sort_by': sort_by,
                'total_words_found': 0,
                'top_words': [],
                'exchange_suggestions': [],
            }
        sorted_words = self.sort_words_by_criteria(valid_words, sort_by, 'wordatro')
        top_words = sorted_words[:10]
        print("Analyzing exchange opportunities...")
        exchange_suggestions = self.find_exchange_opportunities(letters, exchanges_remaining, required_letters, target_length, positional_letters, existing_words=valid_words)
        print("Analysis complete!")
        return {
            'input': input_str,
            'parsed_letters': letters,
            'required_letters': required_letters,
            'target_length': target_length,
            'positional_letters': positional_letters,
            'exchanges_remaining': exchanges_remaining,
            'game_mode': 'wordatro',
            'sort_by': sort_by,
            'total_words_found': len(valid_words),
            'top_words': top_words,
            'exchange_suggestions': exchange_suggestions,
        }


