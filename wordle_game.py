#!/usr/bin/env python3
"""
Wordle-specific game logic.
"""

from typing import List, Tuple, Dict, Set
from collections import Counter

from base_game import BaseWordGame


class WordleGame(BaseWordGame):
    """Wordle-specific game logic and functionality."""

    def __init__(self, dictionary_file: str = "dictionary.txt", force_regenerate: bool = False):
        super().__init__(dictionary_file, force_regenerate)
        # Accumulate information from all guesses
        self.accumulated_positional: Dict[int, str] = {}
        self.accumulated_required: Counter = Counter()
        self.accumulated_excluded: Set[str] = set()
        self.accumulated_position_exclusions: Dict[str, Set[int]] = {}  # letter -> set of excluded positions
        self.guess_history: List[Tuple[str, str]] = []  # (word, feedback) pairs

    def parse_wordle_input(self, input_str: str) -> Tuple[List[str], int, List[str], int, Dict[int, str], Set[str], str]:
        if ':' not in input_str:
            raise ValueError("Wordle input must include feedback after colon (e.g., STARE:GYGYY)")
        parts = input_str.split(':')
        if len(parts) != 2:
            raise ValueError("Wordle input must have exactly one colon")
        word_part = parts[0]
        feedback = parts[1].upper()
        target_length = None
        if word_part and word_part[0].isdigit():
            target_length = int(word_part[0])
            word = word_part[1:].upper()
        else:
            word = word_part.upper()
        if len(word) != len(feedback):
            raise ValueError(f"Word length ({len(word)}) doesn't match feedback length ({len(feedback)})")
        
        # Parse this guess
        required_letters: List[str] = []
        positional_letters: Dict[int, str] = {}
        excluded_letters: Set[str] = set()
        all_letters: List[str] = []
        
        # First pass: collect all green and yellow letters to know what's required
        required_in_this_guess = set()
        for i, (letter, color) in enumerate(zip(word, feedback)):
            if color in ['G', 'Y']:
                required_in_this_guess.add(letter)
        
        # Second pass: process each letter with proper logic
        for i, (letter, color) in enumerate(zip(word, feedback)):
            if color == 'G':
                required_letters.append(letter)
                positional_letters[i] = letter
                all_letters.append(letter)
            elif color == 'Y':
                required_letters.append(letter)
                all_letters.append(letter)
            elif color == 'B':
                # Only exclude if this letter doesn't appear as G or Y elsewhere in this word
                if letter not in required_in_this_guess:
                    excluded_letters.add(letter)
            else:
                raise ValueError(f"Invalid feedback character: {color}. Use G (green), Y (yellow), or B (black)")
        
        # Add this guess to history
        self.guess_history.append((word, feedback))
        
        # Update accumulated information
        self._update_accumulated_info(word, feedback)
        
        exchanges = 0
        return all_letters, exchanges, required_letters, target_length, positional_letters, excluded_letters, 'wordle'

    def _update_accumulated_info(self, word: str, feedback: str):
        """Update accumulated information from this guess."""
        # Update positional (green) letters
        for i, (letter, color) in enumerate(zip(word, feedback)):
            if color == 'G':
                self.accumulated_positional[i] = letter
        
        # Update required letters (green + yellow) - handle duplicates correctly
        current_guess_required = Counter()
        current_guess_excluded_positions = {}  # Track which positions exclude which letters
        
        # First pass: collect all green/yellow letters and their positions
        for i, (letter, color) in enumerate(zip(word, feedback)):
            if color in ['G', 'Y']:
                current_guess_required[letter] += 1
        
        # Second pass: handle black letters that might be duplicates
        for i, (letter, color) in enumerate(zip(word, feedback)):
            if color == 'B':
                # If this letter appears as G or Y elsewhere in this word, don't exclude it entirely
                if letter not in current_guess_required:
                    self.accumulated_excluded.add(letter)
                else:
                    # This letter appears both as B and G/Y in the same word
                    # Track that this specific position excludes this letter
                    if letter not in self.accumulated_position_exclusions:
                        self.accumulated_position_exclusions[letter] = set()
                    self.accumulated_position_exclusions[letter].add(i)
        
        # For each letter, calculate the actual required count
        for letter, count in current_guess_required.items():
            # The required count is the number of green/yellow positions
            # A black position doesn't mean "exclude this letter entirely" - 
            # it just means "this letter is not at this specific position"
            self.accumulated_required[letter] = max(self.accumulated_required[letter], count)

    def find_words_with_exclusions(
        self,
        letters: List[str],
        target_length: int = None,
        required_letters: List[str] = None,
        positional_letters: Dict[int, str] = None,
        excluded_letters: Set[str] = None,
    ) -> Set[str]:
        if excluded_letters is None:
            excluded_letters = set()

        # In Wordle, we shouldn't require every input letter be used; instead, filter the full dictionary
        # by constraints only: length, required letters (multi-count), positional matches, excluded letters.
        if target_length is None:
            # For Wordle, default to 5-letter words, not the length of the letters list
            target_length = 5

        valid = set(self.words_by_length.get(target_length, set()))

        # Filter by accumulated excluded letters first
        if self.accumulated_excluded:
            valid = {w for w in valid if not any(l in w for l in self.accumulated_excluded)}

        # Filter by accumulated required letters with counts
        if self.accumulated_required:
            def meets_required(word: str) -> bool:
                wc = Counter(word)
                return all(wc[l] >= c for l, c in self.accumulated_required.items())
            valid = {w for w in valid if meets_required(w)}

        # Filter by accumulated positional greens
        if self.accumulated_positional:
            valid = {w for w in valid if all(i < len(w) and w[i] == l for i, l in self.accumulated_positional.items())}

        # Filter by accumulated position-specific exclusions
        if self.accumulated_position_exclusions:
            def meets_position_exclusions(word: str) -> bool:
                for letter, excluded_positions in self.accumulated_position_exclusions.items():
                    for pos in excluded_positions:
                        if pos < len(word) and word[pos] == letter:
                            return False
                return True
            valid = {w for w in valid if meets_position_exclusions(w)}

        return valid

    def find_best_words(self, input_str: str, sort_by: str = 'commonality'):
        letters, exchanges_remaining, required_letters, target_length, positional_letters, excluded_letters, game_mode = self.parse_wordle_input(input_str)
        valid_words = self.find_words_with_exclusions(letters, target_length, required_letters, positional_letters, excluded_letters)
        
        if not valid_words:
            return {
                'input': input_str,
                'parsed_letters': letters,
                'required_letters': required_letters,
                'target_length': target_length,
                'positional_letters': positional_letters,
                'excluded_letters': excluded_letters,
                'exchanges_remaining': exchanges_remaining,
                'game_mode': game_mode,
                'sort_by': sort_by,
                'total_words_found': 0,
                'top_words': [],
                'accumulated_info': self._get_accumulated_info_display(),
            }
        
        sorted_words = self.sort_words_by_criteria(valid_words, sort_by, 'wordle')
        top_words = sorted_words[:10]
        
        return {
            'input': input_str,
            'parsed_letters': letters,
            'required_letters': required_letters,
            'target_length': target_length,
            'positional_letters': positional_letters,
            'excluded_letters': excluded_letters,
            'exchanges_remaining': exchanges_remaining,
            'game_mode': game_mode,
            'sort_by': sort_by,
            'total_words_found': len(valid_words),
            'top_words': top_words,
            'accumulated_info': self._get_accumulated_info_display(),
        }

    def _get_accumulated_info_display(self) -> str:
        """Get a formatted string showing accumulated information."""
        lines = []
        
        # Combine all constraints into compact lines
        if self.accumulated_positional:
            pos_str = ', '.join(f"{pos+1}:{letter}" for pos, letter in sorted(self.accumulated_positional.items()))
            lines.append(f"📍 Pos: {pos_str}")
        
        if self.accumulated_required:
            req_str = ', '.join(f"{letter}:{count}" for letter, count in sorted(self.accumulated_required.items()))
            lines.append(f"🔤 Req: {req_str}")
        
        if self.accumulated_excluded:
            excl_str = ', '.join(sorted(self.accumulated_excluded))
            lines.append(f"❌ Excl: {excl_str}")
        
        if self.accumulated_position_exclusions:
            pos_excl_str = ', '.join(f"{letter}:{','.join(str(p+1) for p in sorted(positions))}" 
                                   for letter, positions in sorted(self.accumulated_position_exclusions.items()))
            lines.append(f"🚫 Pos-excl: {pos_excl_str}")
        
        if self.guess_history:
            history_str = ', '.join(f"{word}:{feedback}" for word, feedback in self.guess_history)
            lines.append(f"📝 History: {history_str}")
        
        return "\n".join(lines) if lines else "No constraints yet."

    def reset_accumulated_info(self):
        """Reset all accumulated information (start new game)."""
        self.accumulated_positional.clear()
        self.accumulated_required.clear()
        self.accumulated_excluded.clear()
        self.accumulated_position_exclusions.clear()
        self.guess_history.clear()

    def show_accumulated_info(self):
        """Display current accumulated information."""
        print(self._get_accumulated_info_display())


