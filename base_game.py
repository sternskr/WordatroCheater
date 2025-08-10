#!/usr/bin/env python3
"""
Base game logic shared by Wordatro and Wordle variants.
"""

from collections import Counter
from typing import List, Set, Dict, Tuple

from dictionary_manager import DictionaryManager


class BaseWordGame:
    """Base class for word games with shared functionality."""

    def __init__(self, dictionary_file: str = "dictionary.txt", force_regenerate: bool = False):
        self.dict_manager = DictionaryManager(dictionary_file, force_regenerate)

        self.letter_scores = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }

        self.DEFAULT_WORDLE_LETTER_FREQUENCIES = {
            'E': 0.123, 'T': 0.096, 'A': 0.080, 'O': 0.075, 'I': 0.069, 'N': 0.067,
            'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043, 'L': 0.040, 'U': 0.028,
            'C': 0.027, 'M': 0.026, 'W': 0.024, 'F': 0.022, 'G': 0.020, 'Y': 0.020,
            'P': 0.019, 'B': 0.015, 'V': 0.010, 'K': 0.008, 'J': 0.002, 'X': 0.002,
            'Q': 0.008, 'Z': 0.001
        }

        self.wordle_letter_frequencies = self._compute_wordle_letter_frequencies_from_dictionary()

        self.exchange_cache = {}
        self.substitution_cache = {}

    @property
    def dictionary(self) -> Set[str]:
        return self.dict_manager.dictionary

    @property
    def words_by_length(self) -> Dict[int, Set[str]]:
        return self.dict_manager.words_by_length

    @property
    def words_by_pattern(self) -> Dict[str, Set[str]]:
        return self.dict_manager.words_by_pattern

    @property
    def anagram_groups(self) -> Dict[str, Set[str]]:
        return self.dict_manager.anagram_groups

    @property
    def words_containing_letter(self) -> Dict[str, Set[str]]:
        return self.dict_manager.words_containing_letter

    @property
    def wildcard_compatible(self) -> Dict[str, Set[str]]:
        return self.dict_manager.wildcard_compatible

    def regenerate_cache(self):
        self.dict_manager.regenerate_cache()
        self.wordle_letter_frequencies = self._compute_wordle_letter_frequencies_from_dictionary()

    def save_cache(self):
        self.dict_manager.save_cache()

    def _contains_required_letters(self, word: str, required_letters: List[str]) -> bool:
        if not required_letters:
            return True
        word_letters = Counter(word)
        for letter in required_letters:
            if word_letters[letter] < required_letters.count(letter):
                return False
        return True

    def _matches_positional_letters(self, word: str, positional_letters: Dict[int, str]) -> bool:
        for position, letter in positional_letters.items():
            if position >= len(word) or word[position] != letter:
                return False
        return True

    def generate_word_combinations(
        self,
        letters: List[str],
        target_length: int = None,
        required_letters: List[str] = None,
        positional_letters: Dict[int, str] = None,
    ) -> Set[str]:
        if not letters:
            return set()

        valid_words = set()
        
        if target_length is None:
            # Find words of various lengths from the available letters
            for length in range(3, len(letters) + 1):  # Minimum 3 letters, up to letter count
                words_of_length = self._find_words_of_length(letters, length)
                valid_words.update(words_of_length)
        else:
            valid_words = self._find_words_of_length(letters, target_length)

        if required_letters:
            valid_words = {w for w in valid_words if self._contains_required_letters(w, required_letters)}

        if positional_letters:
            valid_words = {w for w in valid_words if self._matches_positional_letters(w, positional_letters)}

        return valid_words

    def _find_words_of_length(self, letters: List[str], length: int) -> Set[str]:
        if length not in self.words_by_length:
            return set()
        available_letters = Counter(letters)
        wildcard_count = available_letters.get('*', 0)
        if wildcard_count == 0:
            return self._find_words_exact_match(available_letters, length)
        return self._find_words_with_wildcards_optimized(available_letters, length, wildcard_count)

    def _find_words_exact_match(self, letter_counts: Counter, length: int) -> Set[str]:
        valid_words = set()
        for word in self.words_by_length[length]:
            if self._can_form_word(word, letter_counts):
                valid_words.add(word)
        return valid_words

    def _can_form_word(self, word: str, available_letters: Counter) -> bool:
        word_letters = Counter(word)
        for letter, count in word_letters.items():
            if available_letters[letter] < count:
                return False
        return True

    def _find_words_with_wildcards_optimized(self, letter_counts: Counter, length: int, wildcard_count: int) -> Set[str]:
        valid_words = set()
        key = f"{wildcard_count}_{length}"
        if key in self.wildcard_compatible:
            candidate_words = self.wildcard_compatible[key]
            for word in candidate_words:
                if self._can_form_word_with_wildcards(word, letter_counts, wildcard_count):
                    valid_words.add(word)
        return valid_words

    def _can_form_word_with_wildcards(self, word: str, available_letters: Counter, wildcards: int) -> bool:
        word_letters = Counter(word)
        wildcards_needed = 0
        for letter, count in word_letters.items():
            available = available_letters[letter]
            if available < count:
                wildcards_needed += count - available
        return wildcards_needed <= wildcards

    def sort_words_by_criteria(self, words: Set[str], sort_by: str = 'score', game_mode: str = 'wordatro') -> List[Tuple[str, float]]:
        if not words:
            return []
        if sort_by == 'score':
            if game_mode == 'wordatro':
                scored_words = [(w, self.calculate_word_score(w)) for w in words]
            else:
                # For Wordle, use information gain as the primary scoring method
                # This prioritizes words that narrow down the possible answers most effectively
                scored_words = [(w, self.calculate_wordle_information_gain_optimized(w, words)) for w in words]
            return sorted(scored_words, key=lambda x: x[1], reverse=True)
        elif sort_by == 'length':
            return sorted(
                [(w, len(w)) for w in words],
                key=lambda x: (x[1], self.calculate_word_score(x[0]), -ord(x[0][0]) if x[0] else 0),
                reverse=True,
            )
        elif sort_by == 'alphabetical':
            return sorted([(w, 0) for w in words], key=lambda x: x[0])
        elif sort_by == 'commonality':
            if game_mode == 'wordle':
                scored_words = [(w, self.calculate_wordle_commonality_score(w)) for w in words]
            else:
                scored_words = [(w, self.calculate_word_score(w)) for w in words]
            return sorted(scored_words, key=lambda x: x[1], reverse=True)
        elif sort_by == 'information_gain':
            if game_mode == 'wordle':
                # Use the optimized version for better performance
                scored_words = [(w, self.calculate_wordle_information_gain_optimized(w, words)) for w in words]
            else:
                # For non-Wordle games, fall back to regular scoring
                scored_words = [(w, self.calculate_word_score(w)) for w in words]
            return sorted(scored_words, key=lambda x: x[1], reverse=True)
        elif sort_by == 'hybrid':
            if game_mode == 'wordle':
                # Use hybrid scoring that adapts based on word count
                scored_words = [(w, self.calculate_wordle_hybrid_score(w, words)) for w in words]
            else:
                # For non-Wordle games, fall back to regular scoring
                scored_words = [(w, self.calculate_word_score(w)) for w in words]
            return sorted(scored_words, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Unknown sort criteria: {sort_by}")

    def calculate_word_score(self, word: str) -> int:
        return sum(self.letter_scores.get(letter.upper(), 10) for letter in word)

    def _compute_wordle_letter_frequencies_from_dictionary(self) -> Dict[str, float]:
        if 5 in self.words_by_length and self.words_by_length[5]:
            corpus_words = self.words_by_length[5]
        elif self.dictionary:
            corpus_words = self.dictionary
        else:
            return dict(self.DEFAULT_WORDLE_LETTER_FREQUENCIES)

        letter_counts: Counter = Counter()
        total_letters = 0
        for word in corpus_words:
            for ch in word:
                if 'A' <= ch <= 'Z':
                    letter_counts[ch] += 1
                    total_letters += 1
        if total_letters == 0:
            return dict(self.DEFAULT_WORDLE_LETTER_FREQUENCIES)
        frequencies: Dict[str, float] = {chr(c): letter_counts[chr(c)] / total_letters for c in range(ord('A'), ord('Z') + 1)}
        s = sum(frequencies.values())
        if s > 0:
            frequencies = {k: v / s for k, v in frequencies.items()}
        return frequencies

    # Wordle-specific scoring helpers are kept here so both modes can use them if needed
    def calculate_wordle_score(self, word: str) -> float:
        score = 0.0
        for letter in word.upper():
            score += self.wordle_letter_frequencies.get(letter, 0.001)
        common_bigrams = ['TH', 'HE', 'AN', 'IN', 'ER', 'RE', 'ON', 'AT', 'ND', 'ST', 'ES', 'EN', 'OF', 'TE', 'ED', 'OR', 'TI', 'HI', 'AS', 'TO']
        for bigram in common_bigrams:
            if bigram in word.upper():
                score += 0.01
        common_trigrams = ['THE', 'AND', 'THA', 'ENT', 'ING', 'ION', 'TIO', 'FOR', 'NDE', 'HAS', 'NCE', 'EDT', 'TIS', 'OFT', 'STH', 'MEN']
        for trigram in common_trigrams:
            if trigram in word.upper():
                score += 0.02
        return score

    def calculate_wordle_commonality_score(self, word: str) -> float:
        score = 0.0
        for letter in word.upper():
            score += self.wordle_letter_frequencies.get(letter, 0.001)
        rare_letters = ['Q', 'Z', 'X', 'J']
        for letter in word.upper():
            if letter in rare_letters:
                score -= 0.05
        common_patterns = ['E', 'A', 'R', 'I', 'O', 'T', 'N', 'S', 'L', 'C']
        for letter in word.upper():
            if letter in common_patterns:
                score += 0.02
        return score

    def calculate_wordle_information_gain(self, word: str, possible_answers: Set[str]) -> float:
        """
        Calculate the expected information gain of a guess word.
        This measures how much the guess will narrow down the possible answers.
        
        Args:
            word: The word to evaluate as a guess
            possible_answers: Set of words that could still be the answer
            
        Returns:
            Float representing expected information gain (higher = better)
        """
        if not possible_answers or len(possible_answers) <= 1:
            return 0.0
        
        # For each possible feedback pattern, calculate how many words would remain
        total_gain = 0.0
        total_patterns = 0
        
        # Generate all possible feedback patterns for this word
        feedback_patterns = self._generate_all_feedback_patterns(word, possible_answers)
        
        for feedback in feedback_patterns:
            # Count how many words would remain after this feedback
            remaining_words = self._count_words_with_feedback(word, feedback, possible_answers)
            
            if remaining_words > 0:
                # Information gain is log2(total_possible / remaining)
                # This measures how much we've narrowed down the search space
                gain = remaining_words / len(possible_answers)
                total_gain += gain
                total_patterns += 1
        
        if total_patterns == 0:
            return 0.0
        
        # Average gain across all patterns, weighted by probability
        avg_gain = total_gain / total_patterns
        
        # Convert to information gain (lower remaining ratio = higher information gain)
        # We want to maximize information gain, so invert the ratio
        information_gain = 1.0 - avg_gain
        
        # Add a small bonus for commonality as a tiebreaker
        commonality_bonus = self.calculate_wordle_commonality_score(word) * 0.1
        
        return information_gain + commonality_bonus

    def calculate_wordle_information_gain_optimized(self, word: str, possible_answers: Set[str]) -> float:
        """
        Optimized version of information gain calculation that uses a more efficient approach.
        This version is designed to work well even with large dictionaries.
        """
        if not possible_answers or len(possible_answers) <= 1:
            return 0.0
        
        # Use a simplified but effective heuristic for information gain
        # Focus on letter diversity and position coverage
        
        # Count unique letters in the word
        unique_letters = len(set(word))
        
        # Calculate position diversity (how many different letters are in different positions)
        position_diversity = 0
        for i in range(len(word)):
            for j in range(i + 1, len(word)):
                if word[i] != word[j]:
                    position_diversity += 1
        
        # Normalize position diversity
        max_position_diversity = (len(word) * (len(word) - 1)) // 2
        if max_position_diversity > 0:
            position_diversity = position_diversity / max_position_diversity
        
        # Base information gain from letter diversity
        base_gain = unique_letters / len(word)
        
        # Add position diversity bonus
        total_gain = base_gain + (position_diversity * 0.3)
        
        # Add commonality bonus as a tiebreaker (smaller weight)
        commonality_bonus = self.calculate_wordle_commonality_score(word) * 0.05
        
        return total_gain + commonality_bonus

    def calculate_wordle_hybrid_score(self, word: str, possible_answers: Set[str], 
                                    info_gain_threshold: int = 200, 
                                    commonality_threshold: int = 50) -> float:
        """
        Continuous hybrid scoring that smoothly transitions between information gain and commonality
        based on the number of remaining possible answers.
        
        Args:
            word: The word to score
            possible_answers: Set of remaining possible answers
            info_gain_threshold: Center point for the transition (default: 200 words)
            commonality_threshold: Controls transition steepness (default: 50 words)
        """
        num_answers = len(possible_answers)
        
        # Use a smooth transition function that prioritizes info gain for many words
        # and commonality for few words
        
        # Simple approach: use the ratio of words to threshold
        # When num_answers is much larger than threshold, info_gain_weight approaches 1.0
        # When num_answers is much smaller than threshold, info_gain_weight approaches 0.0
        
        # Calculate the ratio and apply a smooth curve
        ratio = num_answers / info_gain_threshold
        
        # Use a smooth curve: info_gain_weight = ratio / (1 + ratio)
        # This gives us a smooth transition from 0 to 1
        # BUT: we want the opposite! Many words → high info gain, few words → high commonality
        # So we invert the logic
        info_gain_weight = ratio / (1 + ratio)
        commonality_weight = 1.0 - info_gain_weight
        
        # Calculate both scores
        info_gain_score = self.calculate_wordle_information_gain_optimized(word, possible_answers)
        commonality_score = self.calculate_wordle_commonality_score(word)
        
        # Normalize scores to 0-1 range for fair weighting
        normalized_info_gain = min(info_gain_score, 1.0)  # Cap at 1.0
        normalized_commonality = min(commonality_score / 10.0, 1.0)  # Scale commonality down
        
        # Combine with dynamic weights
        hybrid_score = (normalized_info_gain * info_gain_weight + 
                       normalized_commonality * commonality_weight)
        
        return hybrid_score

    def _generate_all_feedback_patterns(self, word: str, possible_answers: Set[str]) -> Set[str]:
        """
        Generate all possible feedback patterns for a word against possible answers.
        This is a simplified version - in practice, you'd want to simulate actual Wordle feedback.
        """
        patterns = set()
        
        # For each possible answer, generate what the feedback would be
        for answer in possible_answers:
            if len(word) == len(answer):
                feedback = self._simulate_wordle_feedback(word, answer)
                patterns.add(feedback)
        
        return patterns

    def _simulate_wordle_feedback(self, guess: str, answer: str) -> str:
        """
        Simulate Wordle feedback for a guess against an answer.
        Returns a string like 'GYBYY' where G=Green, Y=Yellow, B=Black.
        """
        if len(guess) != len(answer):
            return 'B' * len(guess)
        
        feedback = ['B'] * len(guess)
        answer_letters = list(answer)
        guess_letters = list(guess)
        
        # First pass: mark greens
        for i in range(len(guess)):
            if guess_letters[i] == answer_letters[i]:
                feedback[i] = 'G'
                answer_letters[i] = None  # Mark as used
                guess_letters[i] = None   # Mark as used
        
        # Second pass: mark yellows
        for i in range(len(guess)):
            if guess_letters[i] is not None:
                if guess_letters[i] in answer_letters:
                    feedback[i] = 'Y'
                    # Remove the first occurrence of this letter from answer
                    for j in range(len(answer_letters)):
                        if answer_letters[j] == guess_letters[i]:
                            answer_letters[j] = None
                            break
        
        return ''.join(feedback)

    def _count_words_with_feedback(self, word: str, feedback: str, possible_answers: Set[str]) -> int:
        """
        Count how many words from possible_answers would give the specified feedback
        when guessed against the given word.
        """
        count = 0
        for answer in possible_answers:
            if self._simulate_wordle_feedback(word, answer) == feedback:
                count += 1
        return count


