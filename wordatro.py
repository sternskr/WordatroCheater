import csv
import itertools
import pickle
import os
import sys
import random
import concurrent.futures
from threading import Lock
from collections import defaultdict, Counter
from typing import List, Set, Tuple, Dict
import re

class WordatroCheater:
    def __init__(self, dictionary_file: str = "dictionary.txt", force_regenerate: bool = False):
        """Initialize the word finder with letter scoring and dictionary loading."""
        
        # Letter scoring system
        self.letter_scores = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }
        
        self.dictionary_file = dictionary_file
        self.cache_file = dictionary_file.replace('.txt', '.word')
        
        # Load dictionary and indexes (with caching)
        self._load_or_build_data(force_regenerate)
    
    def _load_or_build_data(self, force_regenerate: bool = False):
        """Load cached data or build from scratch if needed."""
        
        # Check if we should use cached data
        if not force_regenerate and self._should_use_cache():
            print(f"Loading cached dictionary data from {self.cache_file}...")
            if self._load_cached_data():
                print(f"✓ Loaded {len(self.dictionary)} words from cache")
                return
            else:
                print("⚠ Cache load failed, rebuilding...")
        
        # Build from scratch
        print(f"Building dictionary from {self.dictionary_file}...")
        self._build_from_scratch()
        
        # Save to cache
        self._save_cached_data()
        print(f"✓ Dictionary cached to {self.cache_file}")
    
    def _should_use_cache(self) -> bool:
        """Check if cached data exists and is newer than source dictionary."""
        if not os.path.exists(self.cache_file):
            return False
        
        if not os.path.exists(self.dictionary_file):
            return True  # Use cache if source doesn't exist
        
        # Check if cache is newer than source
        cache_time = os.path.getmtime(self.cache_file)
        source_time = os.path.getmtime(self.dictionary_file)
        
        return cache_time >= source_time
    
    def _load_cached_data(self) -> bool:
        """Load pre-processed data from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Validate the cached data structure
            required_keys = ['dictionary', 'words_by_length', 'words_by_pattern', 
                           'anagram_groups', 'words_containing_letter', 
                           'wildcard_compatible', 'exchange_values']
            
            if not all(key in data for key in required_keys):
                return False
            
            # Load all the pre-computed data and convert back to defaultdicts where needed
            self.dictionary = data['dictionary']
            self.words_by_length = defaultdict(set, data['words_by_length'])
            self.words_by_pattern = defaultdict(set, data['words_by_pattern'])
            self.anagram_groups = defaultdict(set, data['anagram_groups'])
            self.words_containing_letter = defaultdict(set, data['words_containing_letter'])
            self.wildcard_compatible = defaultdict(dict, data['wildcard_compatible'])
            self.exchange_values = data['exchange_values']
            
            return True
            
        except (FileNotFoundError, pickle.PickleError, KeyError, EOFError):
            return False
    
    def _save_cached_data(self):
        """Save pre-processed data to cache file."""
        try:
            # Convert all the complex structures to simple dict/set/list for pickling
            data = {
                'dictionary': self.dictionary,
                'words_by_length': dict(self.words_by_length),
                'words_by_pattern': dict(self.words_by_pattern),
                'anagram_groups': dict(self.anagram_groups),
                'words_containing_letter': dict(self.words_containing_letter),
                'wildcard_compatible': dict(self.wildcard_compatible),
                'exchange_values': dict(self.exchange_values)
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except (IOError, pickle.PickleError) as e:
            print(f"⚠ Warning: Could not save cache: {e}")
    
    def _build_from_scratch(self):
        """Build dictionary and indexes from scratch."""
        # Load dictionary into a set for O(1) lookup
        self.dictionary = self._load_dictionary(self.dictionary_file)
        
        # Filter out words longer than 9 letters (Wordatro maximum)
        self.dictionary = {word for word in self.dictionary if len(word) <= 9}
        
        # Pre-compute optimized data structures for fast lookups
        self._build_optimized_indexes()
    
    def regenerate_cache(self):
        """Force regeneration of the cache file."""
        print("Forcing cache regeneration...")
        self._build_from_scratch()
        self._save_cached_data()
        print(f"✓ Cache regenerated and saved to {self.cache_file}")
    
    def _load_dictionary(self, filename: str) -> Set[str]:
        """Load dictionary from CSV file into a set."""
        dictionary = set()
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Handle both CSV format and plain text format
                content = file.read().strip()
                if ',' in content:
                    # CSV format - need to reset file pointer
                    file.seek(0)
                    reader = csv.reader(file)
                    for row in reader:
                        for word in row:
                            word_clean = word.strip().upper()
                            if word_clean and word_clean.isalpha():
                                dictionary.add(word_clean)
                else:
                    # Plain text format (one word per line)
                    for line in content.split('\n'):
                        word_clean = line.strip().upper()
                        if word_clean and word_clean.isalpha():
                            dictionary.add(word_clean)
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Creating empty dictionary.")
            print("Please ensure dictionary.txt exists with one word per line or CSV format.")
        
        return dictionary
    
    def _build_optimized_indexes(self):
        """Build optimized data structures for fast word lookups during runtime."""
        print("Building optimized indexes with multithreading...")
        
        # Initialize data structures
        self.words_by_length = defaultdict(set)
        self.words_by_pattern = defaultdict(set)
        self.anagram_groups = defaultdict(set)
        self.words_containing_letter = defaultdict(set)
        self.wildcard_compatible = defaultdict(dict)
        
        # Pre-compute exchange value improvements (single-threaded, fast)
        self.exchange_values = {}
        for old_letter in self.letter_scores:
            for new_letter in self.letter_scores:
                self.exchange_values[(old_letter, new_letter)] = (
                    self.letter_scores[new_letter] - self.letter_scores[old_letter]
                )
        
        # Thread-safe locks for shared data structures
        self._lock = Lock()
        
        # Split dictionary into chunks for parallel processing
        words_list = list(self.dictionary)
        chunk_size = max(1000, len(words_list) // (os.cpu_count() or 4))
        word_chunks = [words_list[i:i + chunk_size] for i in range(0, len(words_list), chunk_size)]
        
        print(f"Processing {len(words_list)} words in {len(word_chunks)} chunks using {os.cpu_count() or 4} threads...")
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._process_word_chunk, chunk) for chunk in word_chunks]
            
            # Wait for all chunks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing word chunk: {e}")
        
        print(f"Indexes built: {len(self.dictionary)} words processed")
    
    def _process_word_chunk(self, words_chunk):
        """Process a chunk of words in a separate thread."""
        # Local data structures for this thread
        local_words_by_length = defaultdict(set)
        local_words_by_pattern = defaultdict(set)
        local_anagram_groups = defaultdict(set)
        local_words_containing_letter = defaultdict(set)
        local_wildcard_compatible = defaultdict(dict)
        
        # Process each word in this chunk
        for word in words_chunk:
            word_len = len(word)
            letter_counts = Counter(word)
            sorted_letters = ''.join(sorted(word))
            
            # Basic indexing
            local_words_by_length[word_len].add(word)
            
            # Pattern-based indexing for fast letter matching
            pattern = tuple(sorted(letter_counts.items()))
            local_words_by_pattern[pattern].add(word)
            
            # Anagram grouping
            local_anagram_groups[sorted_letters].add(word)
            
            # Letter containment indexing
            for letter in set(word):
                local_words_containing_letter[letter].add(word)
            
            # Wildcard compatibility pre-computation
            for wildcards_needed in range(1, 4):  # Support 1-3 wildcards
                self._index_wildcard_compatibility_local(word, letter_counts, wildcards_needed, local_wildcard_compatible)
        
        # Merge local results into global data structures (thread-safe)
        with self._lock:
            self._merge_local_indexes(local_words_by_length, local_words_by_pattern, 
                                    local_anagram_groups, local_words_containing_letter, 
                                    local_wildcard_compatible)
    
    def _merge_local_indexes(self, local_words_by_length, local_words_by_pattern, 
                           local_anagram_groups, local_words_containing_letter, 
                           local_wildcard_compatible):
        """Merge local thread results into global indexes."""
        
        # Merge words_by_length
        for length, words in local_words_by_length.items():
            self.words_by_length[length].update(words)
        
        # Merge words_by_pattern
        for pattern, words in local_words_by_pattern.items():
            self.words_by_pattern[pattern].update(words)
        
        # Merge anagram_groups
        for sorted_letters, words in local_anagram_groups.items():
            self.anagram_groups[sorted_letters].update(words)
        
        # Merge words_containing_letter
        for letter, words in local_words_containing_letter.items():
            self.words_containing_letter[letter].update(words)
        
        # Merge wildcard_compatible
        for wildcards_needed, patterns in local_wildcard_compatible.items():
            for pattern, words in patterns.items():
                if pattern not in self.wildcard_compatible[wildcards_needed]:
                    self.wildcard_compatible[wildcards_needed][pattern] = set()
                self.wildcard_compatible[wildcards_needed][pattern].update(words)
    
    def _index_wildcard_compatibility_local(self, word, letter_counts, wildcards_needed, local_wildcard_compatible):
        """Local version of wildcard compatibility indexing for thread safety."""
        word_len = len(word)
        unique_letters = set(word)
        
        from itertools import combinations
        
        min_real_letters = max(0, len(unique_letters) - wildcards_needed)
        max_real_letters = len(unique_letters)
        
        for real_letter_count in range(min_real_letters, max_real_letters + 1):
            if real_letter_count == 0:
                continue
                
            for letter_subset in combinations(unique_letters, real_letter_count):
                partial_counts = {}
                wildcards_used = 0
                
                for letter in letter_subset:
                    partial_counts[letter] = letter_counts[letter]
                
                for letter in unique_letters:
                    if letter not in letter_subset:
                        wildcards_used += letter_counts[letter]
                
                if wildcards_used == wildcards_needed:
                    pattern = tuple(sorted(partial_counts.items()))
                    if pattern not in local_wildcard_compatible[wildcards_needed]:
                        local_wildcard_compatible[wildcards_needed][pattern] = set()
                    local_wildcard_compatible[wildcards_needed][pattern].add(word)
    

    
    def calculate_word_score(self, word: str) -> int:
        """Calculate score for a word: (sum of letter scores) * word length."""
        letter_sum = sum(self.letter_scores.get(letter.upper(), 10) for letter in word)
        return letter_sum * len(word)
    
    def parse_input(self, input_str: str) -> Tuple[List[str], int]:
        """Parse input string to extract letters and number of exchanges."""
        # Extract number at the end
        match = re.search(r'(\d+)$', input_str.strip())
        if match:
            exchanges = int(match.group(1))
            letters_part = input_str[:match.start()].strip()
        else:
            exchanges = 0
            letters_part = input_str.strip()
        
        # Convert to list of individual letters
        letters = list(letters_part.upper())
        return letters, exchanges
    
    def generate_word_combinations(self, letters: List[str], target_length: int = None) -> Set[str]:
        """Generate all possible word combinations from given letters using parallel processing."""
        letter_counts = Counter(letters)
        
        # If no target length specified, try all lengths up to 9 letters (Wordatro max)
        max_length = min(len(letters), 9)  # Wordatro maximum is 9 letters
        lengths_to_try = [target_length] if target_length else range(3, max_length + 1)
        
        if len(lengths_to_try) == 1:
            # Single length - no need for threading
            return self._find_words_of_length(letters, lengths_to_try[0])
        
        # Multiple lengths - use parallel processing
        valid_words = set()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(lengths_to_try), os.cpu_count())) as executor:
            # Submit tasks for each length
            future_to_length = {
                executor.submit(self._find_words_of_length, letters, length): length 
                for length in lengths_to_try if length <= max_length
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_length):
                try:
                    words_for_length = future.result()
                    valid_words.update(words_for_length)
                except Exception as e:
                    length = future_to_length[future]
                    print(f"Error processing length {length}: {e}")
        
        return valid_words
    
    def _find_words_of_length(self, letters: List[str], length: int) -> Set[str]:
        """Find all valid words of specific length from available letters using optimized indexes."""
        valid_words = set()
        letter_counts = Counter(letters)
        wildcard_count = letter_counts.get('*', 0)
        
        if wildcard_count == 0:
            # No wildcards: use direct pattern matching
            return self._find_words_exact_match(letter_counts, length)
        else:
            # Has wildcards: use pre-computed wildcard compatibility
            return self._find_words_with_wildcards_optimized(letter_counts, length, wildcard_count)
    
    def _find_words_exact_match(self, letter_counts: Counter, length: int) -> Set[str]:
        """Find words that can be formed with exact letter matches (no wildcards)."""
        valid_words = set()
        
        # Check if we have enough letters to form words of target length
        total_letters = sum(letter_counts.values())
        if total_letters < length:
            return valid_words
        
        # Use anagram groups for perfect matches
        available_letters = ''.join(sorted(letter_counts.elements()))
        
        # Check all possible subsets of our letters that are exactly 'length' long
        from itertools import combinations
        
        if total_letters == length:
            # Exact match - check anagram groups
            if available_letters in self.anagram_groups:
                valid_words.update(self.anagram_groups[available_letters])
        else:
            # Need to find subsets - use the optimized pattern matching
            for word in self.words_by_length.get(length, set()):
                if self._can_form_word(word, letter_counts):
                    valid_words.add(word)
        
        return valid_words
    
    def _find_words_with_wildcards_optimized(self, letter_counts: Counter, length: int, wildcard_count: int) -> Set[str]:
        """Find words using pre-computed wildcard compatibility data."""
        valid_words = set()
        
        # Remove wildcards from letter counts
        real_letter_counts = letter_counts.copy()
        if '*' in real_letter_counts:
            del real_letter_counts['*']
        
        # Use pre-computed wildcard compatibility if available
        if wildcard_count <= 3 and wildcard_count in self.wildcard_compatible:
            # Create pattern from real letters
            pattern = tuple(sorted(real_letter_counts.items()))
            
            # Check pre-computed compatible words
            if pattern in self.wildcard_compatible[wildcard_count]:
                candidate_words = self.wildcard_compatible[wildcard_count][pattern]
                for word in candidate_words:
                    if len(word) == length:
                        valid_words.add(word)
        
        # Fallback: check words of target length manually
        if not valid_words or wildcard_count > 3:
            for word in self.words_by_length.get(length, set()):
                if self._can_form_word_with_wildcards(word, real_letter_counts, wildcard_count):
                    valid_words.add(word)
        
        return valid_words
    
    def _can_form_word(self, word: str, available_letters: Counter) -> bool:
        """Check if a word can be formed from available letters."""
        word_counts = Counter(word.upper())
        
        for letter, needed in word_counts.items():
            if available_letters.get(letter, 0) < needed:
                return False
        return True
    
    def _can_form_word_with_wildcards(self, word: str, available_letters: Counter, wildcards: int) -> bool:
        """Check if word can be formed using available letters plus wildcards."""
        word_counts = Counter(word.upper())
        wildcards_needed = 0
        
        for letter, needed in word_counts.items():
            available = available_letters.get(letter, 0)
            if available < needed:
                wildcards_needed += needed - available
        
        return wildcards_needed <= wildcards
    

    
    def find_exchange_opportunities(self, letters: List[str], exchanges_remaining: int) -> List[Tuple[str, int, List[str]]]:
        """Analyze least useful letters based on their usage in found words."""
        if exchanges_remaining <= 0:
            return []
        
        current_letters = letters.copy()
        
        # Get current words
        current_words = self.generate_word_combinations(current_letters)
        
        if not current_words:
            return []
        
        # Analyze letter usage in found words to identify least useful letters
        letter_usage_analysis = self._analyze_letter_usage_in_words(current_letters, current_words)
        
        # Format as least useful letters ranked by usefulness
        suggestions = self._format_least_useful_letters(letter_usage_analysis, exchanges_remaining)
        
        return suggestions
    
    def _analyze_letter_usage_in_words(self, input_letters: List[str], found_words: Set[str]) -> Dict:
        """Analyze how frequently each input letter appears in the found words."""
        input_letter_counts = Counter(input_letters)
        
        # Count how many words each input letter appears in
        letter_word_appearances = defaultdict(int)  # How many words contain each letter
        letter_total_usage = defaultdict(int)  # Total times each letter is used across all words
        letter_contribution_scores = defaultdict(float)  # Average score contribution
        
        for word in found_words:
            word_letters = Counter(word.upper())
            word_score = self.calculate_word_score(word)
            
            for letter in input_letter_counts:
                if letter == '*':  # Skip wildcards
                    continue
                    
                if letter in word_letters:
                    # Count word appearances (each word counts once, regardless of letter frequency in that word)
                    letter_word_appearances[letter] += 1
                    
                    # Count total usage across all words
                    letter_total_usage[letter] += word_letters[letter]
                    
                    # Calculate contribution to word score
                    letter_score = self.letter_scores.get(letter, 10)
                    contribution = (letter_score * len(word)) / word_score if word_score > 0 else 0
                    letter_contribution_scores[letter] += contribution
        
        # Calculate usage efficiency for each letter
        letter_efficiency = {}
        for letter in input_letter_counts:
            if letter == '*':
                continue
                
            available_count = input_letter_counts[letter]
            word_appearances = letter_word_appearances[letter]
            total_usage = letter_total_usage[letter]
            avg_contribution = letter_contribution_scores[letter] / max(1, word_appearances)
            
            # Calculate usage rate: what percentage of your available letters are actually being used
            # This is total_usage divided by available_count (how many times you could use this letter)
            max_possible_usage = available_count * len(found_words)  # If every word used every available letter
            actual_usage_rate = total_usage / max_possible_usage if max_possible_usage > 0 else 0
            
            # Efficiency = (word appearances * actual usage rate * avg contribution)
            efficiency = word_appearances * actual_usage_rate * avg_contribution
                
            letter_efficiency[letter] = {
                'efficiency': efficiency,
                'total_usage': total_usage,
                'available_count': available_count,
                'word_appearances': word_appearances,
                'avg_contribution': avg_contribution,
                'usage_rate': actual_usage_rate
            }
        
        return letter_efficiency
    
    def _format_least_useful_letters(self, letter_analysis: Dict, exchanges_remaining: int) -> List[Tuple[str, int, List[str]]]:
        """Format the least useful letters analysis for display."""
        
        # Sort letters by usefulness (least useful first - best candidates for swapping)
        letters_by_usefulness = []
        for letter, analysis in letter_analysis.items():
            if letter != '*' and analysis['available_count'] > 0:
                # Create a usefulness score based on word appearances and usage rate
                usefulness_score = analysis['word_appearances'] * analysis['usage_rate']
                letters_by_usefulness.append((letter, usefulness_score, analysis))
        
        letters_by_usefulness.sort(key=lambda x: x[1])  # Sort by usefulness (lowest first)
        
        if not letters_by_usefulness:
            return []
        
        # Format the results
        suggestions = []
        
        # Create a single suggestion showing least useful letters ranked
        if letters_by_usefulness:
            suggestion_lines = ["Least useful letters to consider exchanging (ranked worst to best):"]
            
            for i, (letter, usefulness, analysis) in enumerate(letters_by_usefulness, 1):
                if analysis['word_appearances'] == 0:
                    description = f"  {i}. {letter} - unused in any words"
                elif analysis['word_appearances'] == 1:
                    description = f"  {i}. {letter} - appears in 1 word ({analysis['usage_rate']:.0%} of available used)"
                else:
                    description = f"  {i}. {letter} - appears in {analysis['word_appearances']} words ({analysis['usage_rate']:.0%} of available used)"
                suggestion_lines.append(description)
            
            suggestion_text = "\n".join(suggestion_lines)
            suggestions.append((suggestion_text, 0, []))  # No score or new letters since this is just analysis
        
        return suggestions
    
    def _generate_random_exchange_scenario(self, current_letters: List[str], letter_pool: List[str], exchange_count: int) -> Tuple[str, List[str], List[str]] or None:
        """Generate a random exchange scenario."""
        if exchange_count > len(current_letters):
            return None
        
        # Don't exchange wildcards
        exchangeable_positions = [i for i, letter in enumerate(current_letters) if letter != '*']
        if len(exchangeable_positions) < exchange_count:
            return None
        
        # Randomly select positions to exchange
        positions_to_exchange = random.sample(exchangeable_positions, exchange_count)
        
        # Get random replacement letters
        new_letters = random.sample(letter_pool, exchange_count)
        
        # Create the test letter set
        test_letters = current_letters.copy()
        old_letters = []
        
        for i, pos in enumerate(positions_to_exchange):
            old_letters.append(current_letters[pos])
            test_letters[pos] = new_letters[i]
        
        description = f"Exchange {exchange_count} letters: {old_letters} → {new_letters}"
        return (description, test_letters, old_letters)
    
    def _test_exchange_scenario(self, scenario: Tuple[str, List[str], List[str]], current_best_score: int) -> Tuple[str, int, List[str]] or None:
        """Test a single exchange scenario."""
        description, test_letters, old_letters = scenario
        
        try:
            # Generate words with new letter combination
            test_words = self.generate_word_combinations(test_letters)
            if not test_words:
                return None
            
            # Calculate best score
            best_test_score = max([self.calculate_word_score(w) for w in test_words])
            
            # Only return if it's an improvement
            if best_test_score > current_best_score:
                improvement = best_test_score - current_best_score
                return (description, improvement, test_letters)
            
        except Exception:
            pass
        
        return None
    
    def find_best_words(self, input_str: str) -> Dict:
        """Main function to find best words and suggestions."""
        letters, exchanges_remaining = self.parse_input(input_str)
        
        # Generate all possible words
        valid_words = self.generate_word_combinations(letters)
        
        if not valid_words:
            return {
                'input': input_str,
                'parsed_letters': letters,
                'exchanges_remaining': exchanges_remaining,
                'top_words': [],
                'exchange_suggestions': [],
                'message': 'No valid words found with current letters.'
            }
        
        # Score all words and get top 10
        scored_words = [(word, self.calculate_word_score(word)) for word in valid_words]
        scored_words.sort(key=lambda x: x[1], reverse=True)
        top_10 = scored_words[:10]
        
        # Get exchange suggestions
        exchange_suggestions = self.find_exchange_opportunities(letters, exchanges_remaining)
        
        return {
            'input': input_str,
            'parsed_letters': letters,
            'exchanges_remaining': exchanges_remaining,
            'top_words': top_10,
            'exchange_suggestions': exchange_suggestions,
            'total_words_found': len(valid_words)
        }
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"WORDATRO CHEATER RESULTS")
        print(f"{'='*60}")
        print(f"Input: {results['input']}")
        print(f"Letters: {' '.join(results['parsed_letters'])}")
        print(f"Exchanges remaining: {results['exchanges_remaining']}")
        print(f"Total valid words found: {results['total_words_found']}")
        
        print(f"\n{'='*30} TOP 10 WORDS {'='*30}")
        if results['top_words']:
            for i, (word, score) in enumerate(results['top_words'], 1):
                letter_score = sum(self.letter_scores.get(letter, 10) for letter in word)
                print(f"{i:2d}. {word:<12} | Score: {score:4d} | ({letter_score} × {len(word)})")
        else:
            print("No valid words found.")
        
        if results.get('exchange_suggestions'):
            print(f"\n{'='*25} LETTER ANALYSIS {'='*25}")
            for i, (suggestion, improvement, new_letters) in enumerate(results['exchange_suggestions'], 1):
                print(f"{suggestion}")
                print()

def main():
    """Main function to run the WordatroCheater utility."""
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--regenerate', '-r', 'regenerate']:
            print("🔄 Regenerating dictionary cache...")
            cheater = WordatroCheater(force_regenerate=True)
            print("✅ Cache regeneration complete!")
            return
        elif sys.argv[1] in ['--help', '-h', 'help']:
            print("WordatroCheater - Find the highest scoring words!")
            print("\nUsage:")
            print("  python wordatro.py                    # Run interactive mode")
            print("  python wordatro.py --regenerate       # Force regenerate cache")
            print("  python wordatro.py --help             # Show this help")
            print("\nCommands in interactive mode:")
            print("  Enter letters + number (e.g., 'ABCDEFGHI3')")
            print("  Use '*' for wildcard letters")
            print("  Type 'quit' to exit")
            return
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    
    # Interactive mode
    cheater = WordatroCheater()
    
    print("\n" + "="*60)
    print("🎯 WordatroCheater - Find the highest scoring words!")
    print("="*60)
    print("Format: Enter letters followed by number of exchanges")
    print("Examples: 'ABCDEFGHI3', 'scrabble2', 'q*xyz*w1'")
    print("Use '*' for wildcard letters. Type 'quit' to exit.")
    print("Type 'regenerate' to rebuild the dictionary cache.\n")
    
    while True:
        try:
            user_input = input("Enter letters and exchanges: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if user_input.lower() in ['regenerate', 'regen', 'rebuild']:
                print("\n🔄 Regenerating dictionary cache...")
                cheater.regenerate_cache()
                print("✅ Cache regenerated! Continuing...\n")
                continue
            
            if not user_input:
                continue
            
            results = cheater.find_best_words(user_input)
            cheater.print_results(results)
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please check your input format.")

if __name__ == "__main__":
    main()
