import csv
import itertools
import pickle
import os
import sys
import random
import concurrent.futures
import atexit
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
        
        # Scrabble tile frequencies for weighted probability calculations
        self.tile_frequencies = {
            'E': 12, 'A': 9, 'I': 9, 'O': 8, 'N': 6, 'R': 6, 'T': 6, 'L': 4, 'S': 4, 'U': 4, 'D': 4, 'G': 3,
            'B': 2, 'C': 2, 'M': 2, 'P': 2, 'F': 2, 'H': 2, 'V': 2, 'W': 2, 'Y': 2,
            'K': 1, 'J': 1, 'X': 1, 'Q': 1, 'Z': 1
        }
        self.total_tiles = sum(self.tile_frequencies.values())  # 98 total tiles
        
        self.dictionary_file = dictionary_file
        self.cache_file = dictionary_file.replace('.txt', '.word')
        
        # Load dictionary and indexes (with caching)
        self._load_or_build_data(force_regenerate)
        
        # Register cache persistence on exit
        atexit.register(self._save_cache_on_exit)
    
    def _load_or_build_data(self, force_regenerate: bool = False):
        """Load cached data or build from scratch if needed."""
        
        # Check if we should use cached data
        if not force_regenerate and self._should_use_cache():
            print(f"Loading cached dictionary data from {self.cache_file}...")
            if self._load_cached_data():
                print(f"Loaded {len(self.dictionary)} words from cache")
                # Prime the substitution cache with common letter patterns if it's relatively empty
                if len(self.substitution_cache) < 250:  # Less than 2.5% full
                    self._prime_substitution_cache(target_fullness=0.025)  # Prime to 2.5%
                return
            else:
                print("⚠ Cache load failed, rebuilding...")
        
        # Build from scratch
        print(f"Building dictionary from {self.dictionary_file}...")
        self._build_from_scratch()
        
        # Save to cache
        self._save_cached_data()
        print(f"Dictionary cached to {self.cache_file}")
        
        # Prime the substitution cache with common letter patterns
        self._prime_substitution_cache(target_fullness=0.025)  # Prime to 2.5% for fresh builds
    
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
            # Load substitution cache if available (optional for backward compatibility)
            self.substitution_cache = data.get('substitution_cache', {})
            # Set cache limit (not saved in cache file to allow runtime adjustment)
            self.substitution_cache_limit = 10000
            
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
                'exchange_values': dict(self.exchange_values),
                'substitution_cache': self.substitution_cache
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
        print(f"Cache regenerated and saved to {self.cache_file}")
    
    def save_cache(self):
        """Manually save the current cache to disk."""
        self._save_cached_data()
        print(f"Cache saved with {len(self.substitution_cache)} entries")
    
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
        
        # Cache for letter substitution results to avoid redundant calculations
        # Use ordered dict for LRU behavior with size limit
        self.substitution_cache = {}
        self.substitution_cache_limit = 10000  # ~1MB memory, covers 500+ analyses
        
        # Thread-safe locks for shared data structures
        self._lock = Lock()
        
        # Split dictionary into chunks for parallel processing
        words_list = list(self.dictionary)
        safe_workers = self._get_safe_worker_count()
        chunk_size = max(1000, len(words_list) // safe_workers)
        word_chunks = [words_list[i:i + chunk_size] for i in range(0, len(words_list), chunk_size)]
        
        print(f"Processing {len(words_list)} words in {len(word_chunks)} chunks using {safe_workers} threads...")
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
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
    
    def parse_input(self, input_str: str) -> Tuple[List[str], int, List[str], int, Dict[int, str]]:
        """Parse input string to extract letters, exchanges, required letters, target length, and positional letters.
        
        Format: [length]letters[exchanges]
        - Number at start = target word length (optional)
        - Capitalized letters = required (must be in any found word)
        - Lowercase letters = optional (can be used but not required)
        - .Letter = positional letter (Letter must be at that specific position)
        - Number at end = exchanges remaining
        
        Examples:
        - "ABCdef3" = any length, A,B,C required, d,e,f optional, 3 exchanges
        - "7ABCdef3" = 7-letter words, A,B,C required, d,e,f optional, 3 exchanges  
        - "A.Bcd.Ef2" = any length, A required, B at position 2, c,d optional, E at position 5, f optional, 2 exchanges
        - "8A.Bcd.Ef2" = 8-letter words, A required, B at position 2, c,d optional, E at position 5, f optional, 2 exchanges
        
        Returns:
            Tuple of (all_letters, exchanges, required_letters, target_length, positional_letters)
        """
        # Extract number at the end (exchanges)
        match = re.search(r'(\d+)$', input_str.strip())
        if match:
            exchanges = int(match.group(1))
            remaining_part = input_str[:match.start()].strip()
        else:
            exchanges = 0
            remaining_part = input_str.strip()
        
        # Extract number at the start (target length)
        target_length = None
        match = re.match(r'^(\d+)', remaining_part)
        if match:
            target_length = int(match.group(1))
            letters_part = remaining_part[match.end():].strip()
        else:
            letters_part = remaining_part
        
        # Parse letters and identify required letters, positional letters
        required_letters = []
        all_letters = []
        positional_letters = {}  # position -> letter
        current_position = 0  # Track position for .Letter syntax
        
        i = 0
        while i < len(letters_part):
            char = letters_part[i]
            
            if char == '.':
                # Next character is a positional letter
                if i + 1 < len(letters_part):
                    pos_char = letters_part[i + 1]
                    if pos_char != '*' and pos_char.isalpha():
                        # Position 1-indexed for user, but we'll store 0-indexed
                        positional_letters[current_position] = pos_char.upper()
                        all_letters.append(pos_char.upper())
                        if pos_char.isupper():
                            required_letters.append(pos_char.upper())
                        current_position += 1
                        i += 2  # Skip the . and the letter
                        continue
                i += 1
            elif char == '*':
                all_letters.append('*')
                current_position += 1
                i += 1
            elif char.isupper():
                # Capitalized = required letter
                required_letters.append(char)
                all_letters.append(char)
                current_position += 1
                i += 1
            elif char.islower():
                # Lowercase = optional letter, convert to uppercase for processing
                all_letters.append(char.upper())
                current_position += 1
                i += 1
            else:
                # Skip unknown characters
                i += 1
        
        return all_letters, exchanges, required_letters, target_length, positional_letters
    
    def _contains_required_letters(self, word: str, required_letters: List[str]) -> bool:
        """Check if word contains all required letters."""
        if not required_letters:
            return True
        
        word_upper = word.upper()
        word_counts = Counter(word_upper)
        required_counts = Counter(required_letters)
        
        # Check if word has at least the required count of each required letter
        for letter, count in required_counts.items():
            if word_counts.get(letter, 0) < count:
                return False
        
        return True
    
    def _matches_positional_letters(self, word: str, positional_letters: Dict[int, str]) -> bool:
        """Check if word matches positional letter requirements."""
        if not positional_letters:
            return True
        
        word_upper = word.upper()
        for position, required_letter in positional_letters.items():
            if position >= len(word_upper) or word_upper[position] != required_letter:
                return False
        
        return True
    
    def generate_word_combinations(self, letters: List[str], target_length: int = None, required_letters: List[str] = None, positional_letters: Dict[int, str] = None) -> Set[str]:
        """Generate all possible word combinations from given letters using parallel processing."""
        if required_letters is None:
            required_letters = []
        if positional_letters is None:
            positional_letters = {}
            
        letter_counts = Counter(letters)
        
        # If no target length specified, try all lengths up to 9 letters (Wordatro max)
        max_length = min(len(letters), 9)  # Wordatro maximum is 9 letters
        lengths_to_try = [target_length] if target_length else range(3, max_length + 1)
        
        if len(lengths_to_try) == 1:
            # Single length - no need for threading
            words = self._find_words_of_length(letters, lengths_to_try[0])
            # Filter for required letters and positional letters
            return {word for word in words 
                   if self._contains_required_letters(word, required_letters) 
                   and self._matches_positional_letters(word, positional_letters)}
        
        # Multiple lengths - use parallel processing
        valid_words = set()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(lengths_to_try), self._get_safe_worker_count())) as executor:
            # Submit tasks for each length
            future_to_length = {
                executor.submit(self._find_words_of_length, letters, length): length 
                for length in lengths_to_try if length <= max_length
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_length):
                try:
                    words_for_length = future.result()
                    # Filter for required letters and positional letters
                    filtered_words = {word for word in words_for_length 
                                    if self._contains_required_letters(word, required_letters) 
                                    and self._matches_positional_letters(word, positional_letters)}
                    valid_words.update(filtered_words)
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
    

    
    def find_exchange_opportunities(self, letters: List[str], exchanges_remaining: int, required_letters: List[str] = None, target_length: int = None, positional_letters: Dict[int, str] = None) -> List[Tuple[str, int, List[str]]]:
        """Analyze least useful letters based on their usage in found words."""
        if exchanges_remaining <= 0:
            return []
        
        if required_letters is None:
            required_letters = []
        if positional_letters is None:
            positional_letters = {}
        
        current_letters = letters.copy()
        
        # Get current words with progress indication
        print("🔍 Generating word combinations...")
        current_words = self.generate_word_combinations(current_letters, target_length=target_length,
                                                      required_letters=required_letters, 
                                                      positional_letters=positional_letters)
        
        if not current_words:
            return []
        
        print(f"✓ Found {len(current_words)} words")
        print("🧮 Analyzing exchange potential...")
        
        # Analyze letter usage in found words to identify least useful letters
        letter_usage_analysis = self._analyze_letter_usage_in_words(current_letters, current_words, required_letters, target_length, positional_letters)
        
        # Format as least useful letters ranked by usefulness
        suggestions = self._format_least_useful_letters(letter_usage_analysis, exchanges_remaining)
        
        return suggestions
    
    def _analyze_letter_usage_in_words(self, input_letters: List[str], found_words: Set[str], required_letters: List[str] = None, target_length: int = None, positional_letters: Dict[int, str] = None) -> Dict:
        """Analyze how destructive removing each letter would be to scoring potential."""
        if required_letters is None:
            required_letters = []
        if positional_letters is None:
            positional_letters = {}
            
        input_letter_counts = Counter(input_letters)
        
        # Calculate the destructive impact of removing each letter
        letter_analysis = {}
        
        # Get baseline stats
        total_words = len(found_words)
        if total_words == 0:
            return {}
        
        total_score_potential = sum(self.calculate_word_score(word) for word in found_words)
        
        # Parallelize letter analysis for better performance
        unique_letters = [letter for letter in input_letter_counts.keys() if letter != '*']
        
        if not unique_letters:
            return letter_analysis
        
        # Use thread pool to analyze letters in parallel
        max_workers = min(len(unique_letters), self._get_safe_worker_count())  # Safe core count
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit analysis tasks for each letter
            future_to_letter = {}
            for letter in unique_letters:
                future = executor.submit(self._analyze_single_letter, 
                                       letter, input_letter_counts, input_letters,
                                       found_words, total_words, total_score_potential,
                                       required_letters, target_length, positional_letters)
                future_to_letter[future] = letter
            
            # Collect results with progress bar
            completed = 0
            total_letters = len(unique_letters)
            
            for future in concurrent.futures.as_completed(future_to_letter):
                letter = future_to_letter[future]
                try:
                    analysis_result = future.result()
                    if analysis_result:
                        letter_analysis[letter] = analysis_result
                    
                    # Update progress
                    completed += 1
                    progress = completed / total_letters
                    bar_length = 20
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f'\r  [{bar}] {progress:.0%} Analyzing letter {letter} ({completed}/{total_letters})', end='', flush=True)
                    
                except Exception as e:
                    # Log error but continue with other letters
                    completed += 1
                    print(f"\rError analyzing letter {letter}: {e}")
            
            # Clear progress bar
            print('\r' + ' ' * 70 + '\r', end='', flush=True)
        
        return letter_analysis
    
    def _get_safe_worker_count(self) -> int:
        """Get a safe number of worker threads, accounting for hyperthreading and system resources."""
        try:
            # Try to get physical cores (not hyperthreaded)
            physical_cores = os.cpu_count()
            try:
                # If psutil is available, get physical cores
                import psutil
                physical_cores = psutil.cpu_count(logical=False)
            except ImportError:
                # Fallback: assume half of logical cores are physical (typical hyperthreading)
                logical_cores = os.cpu_count() or 4
                physical_cores = max(1, logical_cores // 2)
            
            # Reserve 2 cores for system, use at least 1
            safe_count = max(1, physical_cores - 2)
            return safe_count
        except Exception:
            # Conservative fallback
            return 2
    
    def _save_cache_on_exit(self):
        """Save cache to disk when the program exits."""
        try:
            if hasattr(self, 'substitution_cache') and len(self.substitution_cache) > 0:
                self._save_cached_data()
                print(f"\nCache saved with {len(self.substitution_cache)} entries")
        except Exception:
            pass  # Silent fail on exit
    
    def _analyze_single_letter(self, letter: str, input_letter_counts: Counter, input_letters: List[str],
                             found_words: Set[str], total_words: int, total_score_potential: int,
                             required_letters: List[str], target_length: int, positional_letters: Dict[int, str]) -> Dict:
        """Analyze a single letter for exchange potential and removal impact."""
        available_count = input_letter_counts[letter]
        
        # Calculate duplicate analysis first
        words_using_letter = sum(1 for word in found_words if letter in word.upper())
        max_usage_in_word = max((Counter(word.upper())[letter] for word in found_words if letter in word.upper()), default=0)
        
        # Determine how many of this letter are actually excess
        truly_excess = max(0, available_count - max_usage_in_word)
        
        # Test impact of removing letters for any duplicates
        if available_count > 1:
            # Test removing different numbers of this letter to find the safe removal count
            removal_impacts = {}
            
            # Test sequential replacement: 1, 2, 3, ... up to available_count
            for num_to_remove in range(1, min(available_count, 3) + 1):  # Limit to 3 max for performance
                test_letters = input_letters.copy()
                
                # Remove exactly num_to_remove instances of this letter
                removed_count = 0
                for _ in range(num_to_remove):
                    try:
                        test_letters.remove(letter)
                        removed_count += 1
                    except ValueError:
                        # No more instances to remove
                        break
                
                if removed_count == num_to_remove:  # Only proceed if we removed the exact amount
                    # Add wildcards for the removed letters
                    test_letters.extend(['*'] * removed_count)
                    
                    # Calculate exchange potential using the same method as single letters
                    exchange_potential = self._calculate_exchange_potential(
                        test_letters, '*', required_letters, target_length, positional_letters, total_score_potential
                    )
                    
                    # Calculate the destruction impact (score lost by removal)
                    test_letters_no_wildcards = input_letters.copy()
                    for _ in range(removed_count):
                        test_letters_no_wildcards.remove(letter)
                    
                    remaining_words = self.generate_word_combinations(test_letters_no_wildcards, target_length=target_length, 
                                                                    required_letters=required_letters, positional_letters=positional_letters)
                    remaining_score_potential = sum(self.calculate_word_score(word) for word in remaining_words) if remaining_words else 0
                    
                    words_lost = total_words - len(remaining_words)
                    score_lost = total_score_potential - remaining_score_potential
                    
                    # Net effect: exchange potential minus destruction impact
                    exchange_gain = exchange_potential.get('exchange_gain', 0)
                    net_score_change = exchange_gain - score_lost
                    
                    removal_impacts[removed_count] = {
                        'words_lost': words_lost,
                        'score_lost': score_lost,
                        'exchange_potential': exchange_potential,
                        'net_score_change': net_score_change,
                        'destruction_percentage': (score_lost / total_score_potential * 100) if total_score_potential > 0 else 0
                    }
            
            # Use the impact of removing 1 letter for the main removability score
            if 1 in removal_impacts:
                words_lost = removal_impacts[1]['words_lost']
                score_lost = removal_impacts[1]['score_lost']
                removability_score = score_lost
            else:
                words_lost = 0
                score_lost = 0
                removability_score = 0
        else:
            # Single letter - removing it will be destructive
            test_letters = input_letters.copy()
            test_letters.remove(letter)
            
            remaining_words = self.generate_word_combinations(test_letters, target_length=target_length, 
                                                            required_letters=required_letters, positional_letters=positional_letters)
            remaining_score_potential = sum(self.calculate_word_score(word) for word in remaining_words) if remaining_words else 0
            
            words_lost = total_words - len(remaining_words)
            score_lost = total_score_potential - remaining_score_potential
            
            # Single letter removal impact is always high
            removability_score = score_lost
            removal_impacts = {}  # No removal breakdown for single letters
        
        # Calculate exchange potential by substituting with wildcard
        exchange_potential = self._calculate_exchange_potential(input_letters, letter, 
                                                              required_letters, target_length, 
                                                              positional_letters, total_score_potential)
        
        # Build the analysis record
        return {
            'available_count': available_count,
            'words_using_letter': words_using_letter,
            'words_lost_if_removed': words_lost,
            'score_lost_if_removed': score_lost,
            'max_usage_in_word': max_usage_in_word,
            'excess_letters': truly_excess,
            'removability_score': removability_score,
            'destruction_percentage': (score_lost / total_score_potential * 100) if total_score_potential > 0 else 0,
            'is_excess_available': truly_excess > 0,
            'removal_impacts': removal_impacts,  # Always show removal impacts for duplicates
            'exchange_potential': exchange_potential
        }
    
    def _calculate_exchange_potential(self, input_letters: List[str], target_letter: str, 
                                    required_letters: List[str], target_length: int, 
                                    positional_letters: Dict[int, str], baseline_score: int) -> Dict:
        """Calculate the average score potential if we exchange target_letter for each possible letter."""
        
        # Quick optimization: if baseline score is 0, any exchange has potential
        if baseline_score == 0:
            return {
                'average_score_potential': 0,
                'best_substitute': None,
                'best_substitute_score': 0,
                'exchange_gain': 0,
                'exchange_percentage_gain': 0
            }
        
        # Create test letters with wildcard substitution
        test_letters = input_letters.copy()
        
        # Find first occurrence of target_letter and replace with wildcard
        if target_letter in test_letters:
            idx = test_letters.index(target_letter)
            test_letters[idx] = '*'
        else:
            # Letter not found, no exchange potential
            return {
                'average_score_potential': 0,
                'best_substitute': None,
                'best_substitute_score': 0,
                'exchange_gain': 0,
                'exchange_percentage_gain': 0
            }
        
        # Test all 26 letters as substitutions in parallel
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Filter alphabet based on constraints
        valid_substitutes = []
        for substitute_letter in alphabet:
            # Skip if this would violate required letter constraints
            if target_letter in required_letters and substitute_letter not in required_letters:
                continue
            valid_substitutes.append(substitute_letter)
        
        if not valid_substitutes:
            return {
                'average_score_potential': 0,
                'best_substitute': None,
                'best_substitute_score': 0,
                'exchange_gain': 0,
                'exchange_percentage_gain': 0
            }
        
        # Use parallel processing for substitution testing
        substitute_results = []  # Will store (letter, score) tuples
        best_substitute = None
        best_score = 0
        
        # Create thread pool for parallel substitution testing
        max_workers = min(len(valid_substitutes), self._get_safe_worker_count())  # Safe core count
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all substitution tests
            future_to_letter = {}
            for substitute_letter in valid_substitutes:
                future = executor.submit(self._test_single_substitution, 
                                       test_letters, substitute_letter, 
                                       target_length, required_letters, positional_letters)
                future_to_letter[future] = substitute_letter
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_letter):
                substitute_letter = future_to_letter[future]
                try:
                    substitute_score = future.result()
                    substitute_results.append((substitute_letter, substitute_score))
                    
                    if substitute_score > best_score:
                        best_score = substitute_score
                        best_substitute = substitute_letter
                        
                except Exception:
                    substitute_results.append((substitute_letter, 0))
        
        # Calculate weighted average potential based on tile frequencies
        if substitute_results:
            weighted_sum = 0
            total_weight = 0
            
            for substitute_letter, score in substitute_results:
                # Get tile frequency weight for this letter
                weight = self.tile_frequencies.get(substitute_letter, 1)  # Default weight of 1 for missing letters
                weighted_sum += score * weight
                total_weight += weight
            
            average_score_potential = weighted_sum / total_weight if total_weight > 0 else 0
            exchange_gain = average_score_potential - baseline_score
            exchange_percentage_gain = (exchange_gain / baseline_score * 100) if baseline_score > 0 else 0
        else:
            average_score_potential = 0
            exchange_gain = 0
            exchange_percentage_gain = 0
        
        return {
            'average_score_potential': average_score_potential,
            'best_substitute': best_substitute,
            'best_substitute_score': best_score,
            'exchange_gain': exchange_gain,
            'exchange_percentage_gain': exchange_percentage_gain,
            'tested_substitutes': len(substitute_results),
            'weighted_calculation': True  # Indicates this used tile frequency weighting
        }
    
    def _test_single_substitution(self, test_letters: List[str], substitute_letter: str,
                                target_length: int, required_letters: List[str], 
                                positional_letters: Dict[int, str]) -> int:
        """Test a single letter substitution and return the total score."""
        try:
            # Create cache key for this substitution scenario
            cache_key = (
                tuple(sorted(test_letters)), substitute_letter, 
                target_length, tuple(sorted(required_letters)), 
                tuple(sorted(positional_letters.items())) if positional_letters else ()
            )
            
            # Check cache first
            if cache_key in self.substitution_cache:
                return self.substitution_cache[cache_key]
            
            # Create test scenario with this substitute
            substitute_test_letters = test_letters.copy()
            substitute_test_letters[substitute_test_letters.index('*')] = substitute_letter
            
            # Generate words with this substitution
            substitute_words = self.generate_word_combinations(substitute_test_letters, 
                                                             target_length=target_length,
                                                             required_letters=required_letters, 
                                                             positional_letters=positional_letters)
            
            if substitute_words:
                result = sum(self.calculate_word_score(word) for word in substitute_words)
            else:
                result = 0
            
            # Cache the result with LRU management
            self._add_to_cache(cache_key, result)
            return result
                
        except Exception:
            return 0
    
    def _add_to_cache(self, cache_key, result):
        """Add entry to substitution cache with LRU eviction if needed."""
        # If cache is full, remove oldest entries (simple FIFO for now)
        if len(self.substitution_cache) >= self.substitution_cache_limit:
            # Remove oldest 10% of entries to avoid frequent cleanup
            entries_to_remove = max(1, len(self.substitution_cache) // 10)
            oldest_keys = list(self.substitution_cache.keys())[:entries_to_remove]
            for old_key in oldest_keys:
                del self.substitution_cache[old_key]
        
        self.substitution_cache[cache_key] = result
    
    def _get_cache_info(self):
        """Get cache statistics for debugging."""
        return {
            'size': len(self.substitution_cache),
            'limit': self.substitution_cache_limit,
            'utilization': len(self.substitution_cache) / self.substitution_cache_limit * 100
        }
    
    def _format_least_useful_letters(self, letter_analysis: Dict, exchanges_remaining: int) -> List[Tuple[str, int, List[str]]]:
        """Format the least useful letters analysis for display."""
        
        # Sort letters by exchange potential vs removal impact
        letters_by_exchange_value = []
        for letter, analysis in letter_analysis.items():
            if letter != '*' and analysis['available_count'] > 0:
                # Calculate exchange value: positive gain is good, negative loss is bad
                exchange_gain = analysis['exchange_potential']['exchange_gain']
                # Combined score: high exchange gain = good, low removal impact = good
                # Use negative removal impact so higher combined score = better exchange candidate
                combined_score = exchange_gain - analysis['removability_score']
                letters_by_exchange_value.append((letter, combined_score, analysis))
        
        letters_by_exchange_value.sort(key=lambda x: x[1], reverse=True)  # Sort by exchange value (highest first)
        
        if not letters_by_exchange_value:
            return []
        
        # Format the results
        suggestions = []
        
        # Create a single suggestion showing best exchange candidates
        if letters_by_exchange_value:
            suggestion_lines = ["Letters ranked by exchange potential (best candidates first):"]
            
            for i, (letter, exchange_value, analysis) in enumerate(letters_by_exchange_value, 1):
                if analysis['words_using_letter'] == 0:
                    # Show exchange potential for unused letters
                    ep = analysis['exchange_potential']
                    if ep['best_substitute']:
                        description = f"  {i}. {letter} - unused, exchange for {ep['best_substitute']} = +{ep['exchange_percentage_gain']:.1f}% avg gain"
                    else:
                        description = f"  {i}. {letter} - unused in any words (safe to exchange)"
                else:
                    # Build description with exchange potential vs removal impact
                    ep = analysis['exchange_potential']
                    
                    # Core info: removal impact
                    removal_info = f"removing loses {analysis['destruction_percentage']:.1f}%"
                    
                    # Exchange potential info
                    if ep['exchange_gain'] > 0:
                        exchange_info = f"exchange avg +{ep['exchange_percentage_gain']:.1f}%"
                        if ep['best_substitute']:
                            exchange_info += f" (best: {ep['best_substitute']})"
                        verdict = "GOOD TRADE"
                    elif ep['exchange_gain'] < 0:
                        exchange_info = f"exchange avg {ep['exchange_percentage_gain']:.1f}%"
                        verdict = "BAD TRADE"
                    else:
                        exchange_info = "exchange neutral"
                        verdict = "NEUTRAL"
                    
                    # Build compact description
                    base_desc = f"  {i}. {letter} - {removal_info}, {exchange_info} ({verdict})"
                    
                    # Add detailed duplicate removal recommendations
                    if analysis['excess_letters'] > 0:
                        removal_recommendations = self._get_removal_recommendations(analysis)
                        if removal_recommendations:
                            description = base_desc + f" | {removal_recommendations}"
                        else:
                            excess_info = f" | {analysis['excess_letters']} excess"
                            description = base_desc + excess_info
                    elif analysis['available_count'] > 1:
                        # Show removal impacts even for non-excess duplicates
                        removal_recommendations = self._get_removal_recommendations(analysis)
                        if removal_recommendations:
                            description = base_desc + f" | {removal_recommendations}"
                        else:
                            description = base_desc
                    else:
                        description = base_desc
                        
                suggestion_lines.append(description)
            
            suggestion_text = "\n".join(suggestion_lines)
            suggestions.append((suggestion_text, 0, []))  # No score or new letters since this is just analysis
        
        return suggestions
    
    def _get_removal_recommendations(self, analysis: Dict) -> str:
        """Generate specific recommendations for removing duplicate letters."""
        # Show removal impacts for any letter that appears multiple times
        if not analysis['removal_impacts'] or analysis['available_count'] <= 1:
            return ""
        
        recommendations = []
        
        for remove_count in sorted(analysis['removal_impacts'].keys()):
            impact = analysis['removal_impacts'][remove_count]
            destruction_pct = impact['destruction_percentage']
            net_change = impact.get('net_score_change', 0)
            
            # Categorize safety level and trade verdict based on net score change
            if net_change > 100:
                safety = "GAIN"
                trade_verdict = "GOOD TRADE"
            elif net_change > 0:
                safety = "BENEFIT"
                trade_verdict = "GOOD TRADE"
            elif destruction_pct == 0:
                safety = "SAFE"
                trade_verdict = "NEUTRAL"
            elif destruction_pct < 5:
                safety = "OK"
                trade_verdict = "MINOR LOSS"
            elif destruction_pct < 15:
                safety = "CAUTION"
                trade_verdict = "BAD TRADE"
            else:
                safety = "RISKY"
                trade_verdict = "BAD TRADE"
            
            if net_change != 0:
                recommendations.append(f"rm{remove_count}:{destruction_pct:.1f}%({safety}:{net_change:+.0f} {trade_verdict})")
            else:
                recommendations.append(f"rm{remove_count}:{destruction_pct:.1f}%({safety} {trade_verdict})")
        
        if recommendations:
            return "Remove: " + ", ".join(recommendations)
        return ""
    
    def _prime_substitution_cache(self, target_fullness=0.10):
        """Prime the substitution cache with realistic letter combinations based on tile frequencies.
        
        Args:
            target_fullness (float): Target cache fullness as a fraction (0.10 = 10%, 0.50 = 50%)
        """
        target_entries = int(self.substitution_cache_limit * target_fullness)
        current_entries = len(self.substitution_cache)
        
        if current_entries >= target_entries:
            print(f"Cache already at target fullness ({current_entries}/{target_entries} entries, {current_entries/self.substitution_cache_limit*100:.1f}%)")
            return
        
        entries_needed = target_entries - current_entries
        print(f"Priming cache to {target_fullness*100:.0f}% fullness ({target_entries:,} entries)...")
        print(f"Current: {current_entries:,} entries, Need: {entries_needed:,} more")
        
        # Tile frequency distribution (like Scrabble)
        tile_pool = (
            ['E'] * 12 + ['A'] * 9 + ['I'] * 9 + ['O'] * 8 + 
            ['N'] * 6 + ['R'] * 6 + ['T'] * 6 + 
            ['L'] * 4 + ['S'] * 4 + ['U'] * 4 + ['D'] * 4 + 
            ['G'] * 3 + 
            ['B'] * 2 + ['C'] * 2 + ['M'] * 2 + ['P'] * 2 + 
            ['F'] * 2 + ['H'] * 2 + ['V'] * 2 + ['W'] * 2 + ['Y'] * 2 + 
            ['K'] * 1 + ['J'] * 1 + ['X'] * 1 + ['Q'] * 1 + ['Z'] * 1
        )
        
        initial_cache_size = len(self.substitution_cache)
        
        # Generate random combinations based on tile frequencies
        import random
        random.seed(42)  # Reproducible results
        
        # Calculate combinations needed (estimate ~3 entries per combination)
        estimated_combinations = max(20, min(200, entries_needed // 3))
        total_combinations = estimated_combinations
        for i in range(total_combinations):
            try:
                # Update progress bar with cache stats
                progress = (i + 1) / total_combinations
                bar_length = 25
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                current_cache_size = len(self.substitution_cache)
                new_entries = current_cache_size - initial_cache_size
                print(f'\r  [{bar}] {progress:.0%} ({i + 1}/{total_combinations}) | Cache: +{new_entries} entries', end='', flush=True)
                
                # Shuffle the tile pool and draw 7-10 letters
                shuffled_pool = tile_pool.copy()
                random.shuffle(shuffled_pool)
                
                # Draw a random number of letters (7-10)
                draw_count = random.randint(7, 10)
                drawn_letters = shuffled_pool[:draw_count]
                
                # Convert to string and add random exchange count (1-3)
                letter_string = ''.join(drawn_letters)
                exchange_count = random.randint(1, 3)
                test_input = f'{letter_string}{exchange_count}'
                
                # Run analysis to populate cache (suppress all output)
                self._find_best_words_silent(test_input)
                
            except Exception:
                # Skip combinations that cause issues
                continue
        
        # Clear progress bar line and show completion
        print('\r' + ' ' * 80 + '\r', end='', flush=True)
        
        final_cache_size = len(self.substitution_cache)
        new_entries = final_cache_size - initial_cache_size
        
        if new_entries > 0:
            print(f"Cache primed with {new_entries} realistic letter combination entries")
    
    def _find_best_words_silent(self, input_str: str) -> Dict:
        """Silent version of find_best_words for cache priming - no progress output."""
        try:
            letters, exchanges_remaining, required_letters, target_length, positional_letters = self.parse_input(input_str)
            
            # Generate all possible words (silent)
            valid_words = self.generate_word_combinations(letters, target_length=target_length, 
                                                        required_letters=required_letters, 
                                                        positional_letters=positional_letters)
            
            if not valid_words:
                return {}
            
            # Score all words and get top 10 (silent)
            scored_words = [(word, self.calculate_word_score(word)) for word in valid_words]
            scored_words.sort(key=lambda x: x[1], reverse=True)
            top_10 = scored_words[:10]
            
            # Get exchange suggestions (silent)
            exchange_suggestions = self._find_exchange_opportunities_silent(letters, exchanges_remaining, required_letters, target_length, positional_letters)
            
            return {
                'input': input_str,
                'parsed_letters': letters,
                'required_letters': required_letters,
                'target_length': target_length,
                'positional_letters': positional_letters,
                'exchanges_remaining': exchanges_remaining,
                'top_words': top_10,
                'exchange_suggestions': exchange_suggestions,
                'total_words_found': len(valid_words)
            }
        except Exception:
            return {}
    
    def _find_exchange_opportunities_silent(self, letters: List[str], exchanges_remaining: int, required_letters: List[str] = None, target_length: int = None, positional_letters: Dict[int, str] = None) -> List:
        """Silent version of find_exchange_opportunities for cache priming."""
        try:
            if exchanges_remaining <= 0:
                return []
            
            if required_letters is None:
                required_letters = []
            if positional_letters is None:
                positional_letters = {}
            
            current_letters = letters.copy()
            
            # Get current words (silent)
            current_words = self.generate_word_combinations(current_letters, target_length=target_length,
                                                          required_letters=required_letters, 
                                                          positional_letters=positional_letters)
            
            if not current_words:
                return []
            
            # Analyze letter usage (silent)
            letter_usage_analysis = self._analyze_letter_usage_in_words_silent(current_letters, current_words, required_letters, target_length, positional_letters)
            
            # Format results (silent)
            suggestions = self._format_least_useful_letters(letter_usage_analysis, exchanges_remaining)
            
            return suggestions
        except Exception:
            return []
    
    def _analyze_letter_usage_in_words_silent(self, input_letters: List[str], found_words: Set[str], required_letters: List[str] = None, target_length: int = None, positional_letters: Dict[int, str] = None) -> Dict:
        """Silent version of letter usage analysis for cache priming."""
        try:
            if required_letters is None:
                required_letters = []
            if positional_letters is None:
                positional_letters = {}
                
            input_letter_counts = Counter(input_letters)
            letter_analysis = {}
            
            # Get baseline stats
            total_words = len(found_words)
            if total_words == 0:
                return {}
            
            total_score_potential = sum(self.calculate_word_score(word) for word in found_words)
            
            # Analyze each letter (silent - no progress bars)
            unique_letters = [letter for letter in input_letter_counts.keys() if letter != '*']
            
            for letter in unique_letters:
                try:
                    analysis_result = self._analyze_single_letter(letter, input_letter_counts, input_letters,
                                                                found_words, total_words, total_score_potential,
                                                                required_letters, target_length, positional_letters)
                    if analysis_result:
                        letter_analysis[letter] = analysis_result
                except Exception:
                    continue
            
            return letter_analysis
        except Exception:
            return {}
    
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
    
    def _test_exchange_scenario(self, scenario: Tuple[str, List[str], List[str]], current_best_score: int, required_letters: List[str] = None) -> Tuple[str, int, List[str]] or None:
        """Test a single exchange scenario."""
        if required_letters is None:
            required_letters = []
            
        description, test_letters, old_letters = scenario
        
        try:
            # Generate words with new letter combination
            test_words = self.generate_word_combinations(test_letters, required_letters=required_letters)
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
        print(f"📝 Parsing input: {input_str}")
        letters, exchanges_remaining, required_letters, target_length, positional_letters = self.parse_input(input_str)
        
        print("🔍 Generating word combinations...")
        # Generate all possible words (filtering for required letters, target length, and positional letters)
        valid_words = self.generate_word_combinations(letters, target_length=target_length, 
                                                    required_letters=required_letters, 
                                                    positional_letters=positional_letters)
        
        if not valid_words:
            print("❌ No valid words found")
            return {
                'input': input_str,
                'parsed_letters': letters,
                'required_letters': required_letters,
                'target_length': target_length,
                'positional_letters': positional_letters,
                'exchanges_remaining': exchanges_remaining,
                'top_words': [],
                'exchange_suggestions': [],
                'total_words_found': 0,
                'message': 'No valid words found with current letters and constraints.'
            }
        
        print(f"✓ Found {len(valid_words)} valid words")
        print("📊 Calculating scores...")
        
        # Score all words and get top 10
        scored_words = [(word, self.calculate_word_score(word)) for word in valid_words]
        scored_words.sort(key=lambda x: x[1], reverse=True)
        top_10 = scored_words[:10]
        
        print("⚡ Analyzing exchange opportunities...")
        # Get exchange suggestions
        exchange_suggestions = self.find_exchange_opportunities(letters, exchanges_remaining, required_letters, target_length, positional_letters)
        print("✓ Analysis complete!")
        
        return {
            'input': input_str,
            'parsed_letters': letters,
            'required_letters': required_letters,
            'target_length': target_length,
            'positional_letters': positional_letters,
            'exchanges_remaining': exchanges_remaining,
            'top_words': top_10,
            'exchange_suggestions': exchange_suggestions,
            'total_words_found': len(valid_words)
        }
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        # Calculate terminal width-friendly formatting
        terminal_width = 80  # Default conservative width
        header_width = min(terminal_width, 60)
        
        print(f"\n{'='*header_width}")
        print(f"WORDATRO CHEATER RESULTS".center(header_width))
        print(f"{'='*header_width}")
        print(f"Input: {results['input']}")
        print(f"Letters: {' '.join(results['parsed_letters'])}")
        
        # Show constraints if any
        constraints = []
        if results.get('target_length'):
            constraints.append(f"Length: {results['target_length']}")
        if results.get('required_letters'):
            constraints.append(f"Required: {', '.join(results['required_letters'])}")
        if results.get('positional_letters'):
            pos_info = [f"pos{pos+1}:{letter}" for pos, letter in results['positional_letters'].items()]
            constraints.append(f"Positions: {', '.join(pos_info)}")
        
        if constraints:
            print(f"Constraints: {' | '.join(constraints)}")
        
        print(f"Exchanges: {results['exchanges_remaining']} | Words found: {results['total_words_found']}")
        
        print(f"\n{'='*20} TOP 10 WORDS {'='*20}")
        if results['top_words']:
            for i, (word, score) in enumerate(results['top_words'], 1):
                letter_score = sum(self.letter_scores.get(letter, 10) for letter in word)
                # Compact format for smaller terminals
                print(f"{i:2d}. {word:<10} | {score:3d}pts | ({letter_score}×{len(word)})")
        else:
            print("No valid words found.")
        
        if results.get('exchange_suggestions'):
            print(f"\n{'='*15} LETTER ANALYSIS {'='*15}")
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
                print("Saving cache...")
                cheater.save_cache()
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
            print("\nSaving cache...")
            cheater.save_cache()
            print("Goodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please check your input format.")

if __name__ == "__main__":
    main()
