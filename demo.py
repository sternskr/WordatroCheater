#!/usr/bin/env python3
"""
Quick demonstration of WordatroCheater functionality
"""

from wordatro import WordatroCheater

def demo():
    """Demonstrate the WordatroCheater with example inputs."""
    
    print("🎯 WordatroCheater Demo")
    print("=" * 50)
    
    # Initialize the cheater (will use cache if available)
    cheater = WordatroCheater()
    
    # Example inputs (demonstrating case insensitivity)
    examples = [
        ("excellent3", "9-letter word with 3 exchanges (lowercase input)"),
        ("ScRaBbLe2", "Classic word game letters (mixed case)"),
        ("q*xyz*w1", "High-value letters with wildcards (lowercase)"),
        ("WORDGAME0", "Simple input, no exchanges (uppercase)")
    ]
    
    for input_str, description in examples:
        print(f"\n📝 Example: {description}")
        print(f"Input: {input_str}")
        print("-" * 40)
        
        results = cheater.find_best_words(input_str)
        
        # Show top 3 results
        if results['top_words']:
            print("🏆 Top scoring words:")
            for i, (word, score) in enumerate(results['top_words'][:3], 1):
                letter_sum = sum(cheater.letter_scores.get(letter, 10) for letter in word)
                print(f"  {i}. {word:<12} | Score: {score:4d} | ({letter_sum} × {len(word)})")
        else:
            print("❌ No valid words found")
        
        # Show exchange suggestions
        if results['exchange_suggestions']:
            print("💡 Best exchange suggestion:")
            suggestion = results['exchange_suggestions'][0]
            print(f"  {suggestion[0]}")
            print(f"  Improvement: +{suggestion[1]} points")
    
    print(f"\n🎉 Demo complete!")
    print("Run 'python wordatro.py' for interactive mode")

if __name__ == "__main__":
    demo() 