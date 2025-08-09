# WordatroCheater

A powerful utility to find the highest-scoring words for word games like Wordatro, optimized for up to 9-letter combinations (maximum word length) with strategic letter exchanges.

## Features

- **Fast Dictionary Loading**: Loads dictionary into optimized data structures for O(1) word lookup
- **Smart Caching**: Pre-processes dictionary once, then loads instantly from cache on subsequent runs
- **Case Insensitive**: Handles both uppercase and lowercase input automatically
- **Advanced Scoring System**: Implements the exact Wordatro scoring: (sum of letter scores) × word length
- **Wildcard Support**: Handles `*` wildcards using pre-computed compatibility tables
- **Realistic Exchange Simulation**: Simulates random letter exchanges (1-3 letters) like real game mechanics
- **Top 10 Results**: Returns the highest-scoring words with detailed score breakdowns
- **Interactive CLI**: User-friendly command-line interface

## Letter Scoring System

```
A=1  B=3  C=3  D=2  E=1  F=4  G=2  H=4  I=1  J=8
K=5  L=1  M=3  N=1  O=1  P=3  Q=10 R=1  S=1  T=1
U=1  V=4  W=4  X=8  Y=4  Z=10 *=10 (wildcard)
```

**Score Calculation**: `(sum of letter scores) × (number of letters)`

## Setup

1. **Dictionary File**: Create a `dictionary.txt` file with valid words (one per line or CSV format)
2. **Run the Utility**: Execute `python wordatro.py`

## Usage

### Interactive Mode
```bash
python wordatro.py                    # Normal interactive mode
python wordatro.py --regenerate       # Force rebuild dictionary cache
python wordatro.py --help             # Show help information
```

### Input Format
Enter letters followed by the number of exchanges available (case insensitive):
- `ABCDEFGHI3` or `abcdefghi3` - 9 letters with 3 exchanges (maximum word length is 9)
- `SCRABBLE2` or `scrabble2` - 8 letters with 2 exchanges  
- `QU*ZZ*XY1` or `qu*zz*xy1` - Letters with wildcards (*) and 1 exchange
- `GAMEWORD0` or `gameword0` - No exchanges available

### Example Session
```
WordatroCheater - Find the highest scoring words!
Format: Enter letters followed by number of exchanges (e.g., 'ABCDEFGHI3' or 'ABC*EFG**2')
Use '*' for wildcard letters. Type 'quit' to exit.

Enter letters and exchanges: SCRABBLE2

============================================================
WORDATRO CHEATER RESULTS
============================================================
Input: SCRABBLE2
Letters: S C R A B B L E
Exchanges remaining: 2
Total valid words found: 15

============================== TOP 10 WORDS ==============================
 1. SCRABBLE     | Score:  156 | (12 × 8)
 2. RABBLES      | Score:  126 | (18 × 7)
 3. BABLERS      | Score:  105 | (15 × 7)
 4. CABLES       | Score:   90 | (15 × 6)
 5. BRACE        | Score:   40 | (8 × 5)

========================= EXCHANGE SUGGESTIONS =========================
1. Exchange 1 letters: ['A'] → ['Q']
   Potential improvement: +45 points
   New letters: S C R Q B B L E

2. Exchange 2 letters: ['A', 'E'] → ['X', 'Z']
   Potential improvement: +89 points
   New letters: S C R X B B L Z
```

## API Usage

```python
from wordatro import WordatroCheater

# Initialize
cheater = WordatroCheater("dictionary.txt")

# Find best words
results = cheater.find_best_words("ABCDEFGHI3")

# Print formatted results
cheater.print_results(results)

# Access results programmatically
top_words = results['top_words']          # [(word, score), ...]
exchanges = results['exchange_suggestions']  # Exchange recommendations
total_found = results['total_words_found']   # Number of valid words
```

## Key Functions

### `calculate_word_score(word)`
Calculates the score for a given word using the Wordatro scoring system.

### `generate_word_combinations(letters)`
Generates all possible valid words from the given letters, handling wildcards.

### `find_exchange_opportunities(letters, exchanges)`
Analyzes potential letter exchanges and suggests the most beneficial swaps.

### `parse_input(input_str)`
Parses user input to extract letters and number of exchanges.

## Testing

Run the test suite to verify functionality:
```bash
python test_wordatro.py
```

### Cache Management

The utility automatically creates a `dictionary.word` cache file after first run:
- **Instant Loading**: Subsequent runs load in seconds instead of minutes
- **Auto-Update**: Cache rebuilds automatically if `dictionary.txt` is modified
- **Manual Rebuild**: Use `python wordatro.py --regenerate` or type `regenerate` in interactive mode
- **Cache Location**: Same directory as `dictionary.txt`, with `.word` extension

## Dictionary Format

The utility supports both formats:
- **Plain text**: One word per line
- **CSV**: Comma-separated words

Example `dictionary.txt`:
```
WORD
GAME
SCRABBLE
EXCELLENT
WORDGAMES
```

## Performance Features

- **Smart Caching**: Dictionary processed once, cached as `dictionary.word` for instant loading
- **Multithreaded Processing**: Uses all CPU cores for dictionary indexing and word generation
- **Parallel Exchange Testing**: Tests multiple exchange scenarios simultaneously
- **Set-based Dictionary**: O(1) word lookup time
- **Pre-computed Indexes**: Letter patterns, anagram groups, and wildcard compatibility tables
- **Concurrent Word Generation**: Parallel processing for different word lengths
- **Automatic Cache Management**: Detects when source dictionary is updated and rebuilds cache

## Tips for Best Results

1. **Use High-Value Letters**: Q, X, Z, J provide the highest scores
2. **Maximize Word Length**: Longer words get higher multipliers
3. **Smart Exchanges**: The utility simulates realistic random exchanges - try suggested swaps
4. **Wildcard Placement**: Use wildcards for high-value letter positions
5. **Multiple Attempts**: Exchange suggestions are based on random sampling - run multiple times for different options

## Requirements

- Python 3.6+
- Standard library modules only (no external dependencies)

## License

This utility is provided as-is for educational and entertainment purposes. 