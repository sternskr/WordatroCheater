#!/usr/bin/env python3
"""
Dictionary manager module: loads dictionary, caches data, and builds indexes.
"""

import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Set, Dict


class ProgressBar:
    """Simple text-based progress bar for terminal output."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, increment: int = 1):
        self.current += increment
        current_time = time.time()
        
        # Update every 0.1 seconds or when complete
        if current_time - self.last_update >= 0.1 or self.current >= self.total:
            self._display()
            self.last_update = current_time
    
    def _display(self):
        if self.total == 0:
            return
            
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Estimate remaining time
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"ETA: {remaining:.1f}s"
        else:
            eta_str = "ETA: --"
        
        # Create progress bar
        bar_width = 30
        filled = int((self.current / self.total) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Format output
        status = f"{self.description}: {self.current:,}/{self.total:,} ({percentage:.1f}%)"
        progress = f"[{bar}]"
        timing = f"{elapsed:.1f}s elapsed, {eta_str}"
        
        # Clear line and print progress
        print(f"\r{status} {progress} {timing}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


class DictionaryManager:
    """Manages dictionary loading, caching, and optimized indexing."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, dictionary_file: str = "dictionary.txt", force_regenerate: bool = False):
        # Only initialize once
        if self._initialized:
            return
            
        self.dictionary_file = dictionary_file
        self.cache_file = dictionary_file.replace('.txt', '.word')
        self.dictionary: Set[str] = set()

        # Optimized indexes for fast word lookups
        self.words_by_length: Dict[int, Set[str]] = defaultdict(set)
        self.words_by_pattern: Dict[str, Set[str]] = defaultdict(set)
        self.anagram_groups: Dict[str, Set[str]] = defaultdict(set)
        self.words_containing_letter: Dict[str, Set[str]] = defaultdict(set)
        self.wildcard_compatible: Dict[str, Set[str]] = defaultdict(set)

        self._load_or_build_data(force_regenerate)

        import atexit
        atexit.register(self._save_cache_on_exit)
        
        self._initialized = True

    def _load_or_build_data(self, force_regenerate: bool = False):
        if not force_regenerate and self._should_use_cache():
            if not self._load_cached_data():
                self._build_from_scratch()
        else:
            self._build_from_scratch()

    def _should_use_cache(self) -> bool:
        return (
            os.path.exists(self.cache_file)
            and os.path.getmtime(self.cache_file) > os.path.getmtime(self.dictionary_file)
        )

    def _load_cached_data(self) -> bool:
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.dictionary = cached_data['dictionary']
                self.words_by_length = cached_data['words_by_length']
                self.words_by_pattern = cached_data['words_by_pattern']
                self.anagram_groups = cached_data['anagram_groups']
                self.words_containing_letter = cached_data['words_containing_letter']
                self.wildcard_compatible = cached_data['wildcard_compatible']
                print(f"Loaded {len(self.dictionary)} words from cache")
                return True
        except Exception as e:
            print(f"Cache load failed: {e}")
            return False

    def show_cache_stats(self):
        """Display current cache statistics without saving."""
        total_words = len(self.dictionary)
        length_indexes = sum(len(words) for words in self.words_by_length.values())
        pattern_indexes = sum(len(words) for words in self.words_by_pattern.values())
        anagram_indexes = sum(len(words) for words in self.anagram_groups.values())
        letter_indexes = sum(len(words) for words in self.words_containing_letter.values())
        wildcard_indexes = sum(len(words) for words in self.wildcard_compatible.values())
        
        total_index_entries = length_indexes + pattern_indexes + anagram_indexes + letter_indexes + wildcard_indexes
        
        print(f"Cache Stats:")
        print(f"  Words: {total_words:,} | Index entries: {total_index_entries:,}")
        print(f"  Indexes: length({len(self.words_by_length)}), pattern({len(self.words_by_pattern)})")
        print(f"           anagram({len(self.anagram_groups)}), letter({len(self.words_containing_letter)})")
        print(f"           wildcard({len(self.wildcard_compatible)})")
        
        if os.path.exists(self.cache_file):
            file_size = os.path.getsize(self.cache_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"  Cache file: {file_size_mb:.1f}MB")

    def _save_cached_data(self):
        try:
            cached_data = {
                'dictionary': self.dictionary,
                'words_by_length': self.words_by_length,
                'words_by_pattern': self.words_by_pattern,
                'anagram_groups': self.anagram_groups,
                'words_containing_letter': self.words_containing_letter,
                'wildcard_compatible': self.wildcard_compatible,
            }
            
            # Calculate cache statistics
            total_words = len(self.dictionary)
            length_indexes = sum(len(words) for words in self.words_by_length.values())
            pattern_indexes = sum(len(words) for words in self.words_by_pattern.values())
            anagram_indexes = sum(len(words) for words in self.anagram_groups.values())
            letter_indexes = sum(len(words) for words in self.words_containing_letter.values())
            wildcard_indexes = sum(len(words) for words in self.wildcard_compatible.values())
            
            # Total index entries (words can appear in multiple indexes)
            total_index_entries = length_indexes + pattern_indexes + anagram_indexes + letter_indexes + wildcard_indexes
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Get file size for context
            file_size = os.path.getsize(self.cache_file)
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"Cache saved: {total_words:,} words → {total_index_entries:,} optimized index entries")
            print(f"  Index breakdown: length={len(self.words_by_length)}, pattern={len(self.words_by_pattern)}, anagram={len(self.anagram_groups)}, letter={len(self.words_containing_letter)}, wildcard={len(self.wildcard_compatible)}")
            print(f"  Cache file size: {file_size_mb:.1f}MB")
            
        except Exception as e:
            print(f"Cache save failed: {e}")

    def _build_from_scratch(self):
        print("Building optimized indexes...")
        start_time = time.time()

        # Load dictionary with progress
        print("Loading dictionary...")
        self._load_dictionary(self.dictionary_file)
        
        # Build indexes with progress
        print("Building indexes...")
        self._build_optimized_indexes()

        elapsed = time.time() - start_time
        print(f"Built indexes in {elapsed:.2f}s")
        # Cache will be saved automatically by atexit handler

    def _load_dictionary(self, filename: str) -> Set[str]:
        try:
            # First pass: count lines for progress bar
            with open(filename, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            # Second pass: load words with progress
            words = set()
            with open(filename, 'r', encoding='utf-8') as f:
                progress = ProgressBar(total_lines, "Loading dictionary")
                for line_num, line in enumerate(f):
                    word = line.strip()
                    if word:
                        words.add(word.upper())
                    progress.update(1)
            
            print(f"Loaded {len(words)} words from {filename}")
            self.dictionary = words
            return words
        except FileNotFoundError:
            print(f"Dictionary file '{filename}' not found. Creating empty dictionary.")
            self.dictionary = set()
            return set()
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            self.dictionary = set()
            return set()

    def _build_optimized_indexes(self):
        self.words_by_length.clear()
        self.words_by_pattern.clear()
        self.anagram_groups.clear()
        self.words_containing_letter.clear()
        self.wildcard_compatible.clear()

        words_list = list(self.dictionary)
        total_words = len(words_list)
        
        print(f"Building indexes for {total_words:,} words...")

        # Heuristic: avoid multiprocessing for small corpora or under test harness
        disable_mp = (
            len(words_list) < 50000
            or os.environ.get('WORDATRO_DISABLE_MP') == '1'
            or 'unittest' in sys.modules
        )

        if disable_mp or not words_list:
            # Single-threaded processing with progress
            progress = ProgressBar(total_words, "Building indexes")
            results = [self._process_word_chunk_with_progress(words_list, progress)]
        else:
            # Multi-threaded processing - show chunk progress
            chunk_size = max(1, len(words_list) // (cpu_count() * 4))
            chunks = [words_list[i:i + chunk_size] for i in range(0, len(words_list), chunk_size)]
            print(f"Processing {len(chunks)} chunks with {cpu_count()} cores...")
            
            with Pool() as pool:
                results = pool.map(self._process_word_chunk, chunks)

        # Merge results with progress
        print("Merging index results...")
        progress = ProgressBar(len(results), "Merging indexes")
        for result in results:
            self._merge_local_indexes(*result)
            progress.update(1)

    def _process_word_chunk_with_progress(self, words_chunk, progress):
        """Process a chunk of words with progress updates."""
        local_words_by_length = defaultdict(set)
        local_words_by_pattern = defaultdict(set)
        local_anagram_groups = defaultdict(set)
        local_words_containing_letter = defaultdict(set)
        local_wildcard_compatible = defaultdict(set)

        for word in words_chunk:
            local_words_by_length[len(word)].add(word)
            pattern = self._get_word_pattern(word)
            local_words_by_pattern[pattern].add(word)
            sorted_letters = ''.join(sorted(word))
            local_anagram_groups[sorted_letters].add(word)
            for letter in set(word):
                local_words_containing_letter[letter].add(word)

            letter_counts = Counter(word)
            for wildcards_needed in range(1, 4):
                self._index_wildcard_compatibility_local(
                    word, letter_counts, wildcards_needed, local_wildcard_compatible
                )
            
            progress.update(1)

        return (
            local_words_by_length,
            local_words_by_pattern,
            local_anagram_groups,
            local_words_containing_letter,
            local_wildcard_compatible,
        )

    def _process_word_chunk(self, words_chunk):
        local_words_by_length = defaultdict(set)
        local_words_by_pattern = defaultdict(set)
        local_anagram_groups = defaultdict(set)
        local_words_containing_letter = defaultdict(set)
        local_wildcard_compatible = defaultdict(set)

        for word in words_chunk:
            local_words_by_length[len(word)].add(word)
            pattern = self._get_word_pattern(word)
            local_words_by_pattern[pattern].add(word)
            sorted_letters = ''.join(sorted(word))
            local_anagram_groups[sorted_letters].add(word)
            for letter in set(word):
                local_words_containing_letter[letter].add(word)

            letter_counts = Counter(word)
            for wildcards_needed in range(1, 4):
                self._index_wildcard_compatibility_local(
                    word, letter_counts, wildcards_needed, local_wildcard_compatible
                )

        return (
            local_words_by_length,
            local_words_by_pattern,
            local_anagram_groups,
            local_words_containing_letter,
            local_wildcard_compatible,
        )

    def _merge_local_indexes(
        self,
        local_words_by_length,
        local_words_by_pattern,
        local_anagram_groups,
        local_words_containing_letter,
        local_wildcard_compatible,
    ):
        for length, words in local_words_by_length.items():
            self.words_by_length[length].update(words)
        for pattern, words in local_words_by_pattern.items():
            self.words_by_pattern[pattern].update(words)
        for sorted_letters, words in local_anagram_groups.items():
            self.anagram_groups[sorted_letters].update(words)
        for letter, words in local_words_containing_letter.items():
            self.words_containing_letter[letter].update(words)
        for wildcard_key, words in local_wildcard_compatible.items():
            self.wildcard_compatible[wildcard_key].update(words)

    def _index_wildcard_compatibility_local(self, word, letter_counts, wildcards_needed, local_wildcard_compatible):
        """Index word for wildcard compatibility using key like '2_5' for 2 wildcards, length 5."""
        wildcard_key = f"{wildcards_needed}_{len(word)}"
        local_wildcard_compatible[wildcard_key].add(word)

    def _get_word_pattern(self, word: str) -> str:
        letter_map = {}
        pattern = []
        next_letter = 'A'
        for letter in word:
            if letter not in letter_map:
                letter_map[letter] = next_letter
                next_letter = chr(ord(next_letter) + 1)
            pattern.append(letter_map[letter])
        return ''.join(pattern)

    def regenerate_cache(self):
        print("Regenerating cache...")
        self._build_from_scratch()

    def save_cache(self):
        self._save_cached_data()
        # Set timestamp to prevent duplicate saves from atexit handler
        self._last_save_time = time.time()

    def _save_cache_on_exit(self):
        # Check if cache was already saved recently (within last second)
        if hasattr(self, 'dictionary') and self.dictionary:
            current_time = time.time()
            if not hasattr(self, '_last_save_time') or (current_time - self._last_save_time) > 1:
                self._save_cached_data()
                self._last_save_time = current_time


