#!/usr/bin/env python3
"""
WordatroCheater - A word game assistant supporting both Wordatro and Wordle modes

This is the main entry point that imports from the modular structure.
"""

import sys

from wordatro_cheater import WordatroCheater


def main():
    """Main interactive mode for the Wordatro Cheater."""
    
    # Initialize the cheater
    cheater = WordatroCheater()
    
    try:
        while True:
            print("\n" + "="*40)
            print("🎯 Wordatro Cheater - Choose Game Mode")
            print("="*40)
            print("1. Wordatro Mode (letters + exchanges)")
            print("2. Wordle Mode (word:feedback)")
            print("3. Show Cache Statistics")
            print("4. Wordle Game Management")
            print("q. Quit and Save Cache")
            print("="*40)
            
            choice = input("Enter your choice (1/2/3/4/q): ").strip().lower()
            
            if choice == 'q':
                print("Saving cache and exiting...")
                cheater.save_cache()
                print("Goodbye! 🎯")
                break
                
            elif choice == '3':
                cheater.show_cache_stats()
                continue
                
            elif choice == '4':
                # Wordle Game Management submenu
                while True:
                    print("\n" + "-"*30)
                    print("🎯 Wordle Game Management")
                    print("-"*30)
                    print("1. Show current game state")
                    print("2. Reset game (clear all guesses)")
                    print("3. Back to main menu")
                    
                    subchoice = input("Enter choice (1/2/3): ").strip()
                    
                    if subchoice == '1':
                        cheater.show_wordle_state()
                    elif subchoice == '2':
                        cheater.reset_wordle_game()
                    elif subchoice == '3':
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, or 3.")
                continue
                
            elif choice not in ['1', '2']:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or q.")
                continue
            
            # Game mode selected - enter the sub-loop
            while True:
                print("\n" + "-"*30)
                
                if choice == '1':
                    print("🎲 Wordatro Mode")
                    print("Format: letters + exchanges (e.g., STARE3, 5STARE3)")
                else:
                    print("🎯 Wordle Mode") 
                    print("Format: word:feedback (e.g., STARE:GYGYY)")
                    print("G=Green, Y=Yellow, B=Black")
                
                print("Commands: 'back' to menu, 'q' to quit")
                print("-"*30)
                
                # Get user input
                user_input = input("Enter input: ").strip()
                
                if user_input.lower() == 'q':
                    print("Saving cache and exiting...")
                    cheater.save_cache()
                    print("Goodbye! 🎯")
                    return
                    
                elif user_input.lower() == 'back':
                    break
                    
                elif not user_input:
                    print("❌ Please enter some input or use 'back'/'q'")
                    continue
                
                try:
                    # Simplified sort options
                    print("\nSort by: score (Enter), commonality (c), length (l), alpha (a)")
                    
                    sort_choice = input("Sort choice: ").strip().lower()
                    if not sort_choice:
                        sort_choice = 'score'
                    elif sort_choice == 'c':
                        sort_choice = 'commonality'
                    elif sort_choice == 'l':
                        sort_choice = 'length'
                    elif sort_choice == 'a':
                        sort_choice = 'alphabetical'
                    
                    # Validate sort choice
                    valid_sorts = ['score', 'commonality', 'length', 'alphabetical']
                    if sort_choice not in valid_sorts:
                        print(f"Using default: score")
                        sort_choice = 'score'
                    
                    print(f"Processing with '{sort_choice}' sorting...")
                    
                    # Process the input with chosen sorting
                    results = cheater.find_best_words(user_input, sort_choice)
                    
                    # Display results
                    cheater.print_results(results)
                    
                    # Small separator to show we're ready for next input
                    print("-" * 30)
                    
                    # No need to ask about retrying sorting - just continue to next input
                    # The loop will continue automatically
                    
                except ValueError as e:
                    print(f"❌ Invalid input format: {e}")
                    print("Please check your input and try again.")
                except Exception as e:
                    print(f"❌ Error processing input: {e}")
                    print("Please check your input format and try again.")
                
                # Automatically continue in same mode - no prompt needed
                # The loop will continue automatically
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user. Saving cache and exiting...")
        cheater.save_cache()
        print("Goodbye! 🎯")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Saving cache and exiting...")
        cheater.save_cache()
        raise


if __name__ == "__main__":
    main()
