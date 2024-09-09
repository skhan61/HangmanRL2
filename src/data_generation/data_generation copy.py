import argparse
import json
import logging
import math
import random
import shutil
import string
from itertools import chain, combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path
from queue import Queue
from threading import Thread
from tqdm import tqdm

import numpy as np
import pandas as pd
import functools

# Ensure that these utility functions are properly implemented
from src.utils import read_word_list, \
            calculate_frequencies, get_character_probability
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
import concurrent.futures
from tqdm import tqdm
import os

def recreate_directory(path):
    path = Path(path)  # Ensure path is a Path object
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)  # Use exist_ok to avoid raising 
                                             # an error if the directory already exists
    logging.info(f"Directory recreated at {path}")

# Mimicking API
def update_game_stage(masked_word, guessed_letters):
    word_len = len(masked_word)
    incorrect_guesses = {letter for letter in guessed_letters \
                         if all(letter != ch for ch in masked_word if ch != '_')}
    incorrect_guesses_count = len(incorrect_guesses)
    remaining_attempts = 6 - incorrect_guesses_count
    return remaining_attempts


## Generator
# Guess + guess_character with simulated guesser
def simulated_guess_function(original_word, masked_word, \
                             solver, guessed_letters):
    """
    Determine the next character to guess using the solver, 
    considering guessed letters.
    Also, update the guessed letters list internally.
    """
    # Generate the real likelihood vector using the updated function
    ## a loss less machine
    real_likelihood_vector = np.array(calculate_dynamic_label(original_word, masked_word))

    # Convert guessed letters to a set for processing
    # generating action guessed_letter set should be set()
    guessed_letter_set = set(guessed_letters)

    tries_remains = update_game_stage(masked_word, guessed_letters)

    best_char_index = solver.generate_next_guess(masked_word, real_likelihood_vector, \
                                        guessed_letters, tries_remains=tries_remains)
    best_guess = string.ascii_lowercase[best_char_index]

    if best_guess and best_guess not in guessed_letter_set:
        guessed_letters.append(best_guess)  # Update the guessed letters list

    # print(f"Original word {original_word} given masked word {masked_word}, guess {best_guess}")
    # print('\n')
    return best_guess


def calculate_dynamic_label(word, masked_word):
    labels = np.zeros(len(string.ascii_lowercase), dtype=int)
    unmasked_letters = set(masked_word.replace('_', ''))  # Letters that are visible

    for char in set(word):
        if char in string.ascii_lowercase and char not in unmasked_letters:
            index = string.ascii_lowercase.index(char)
            labels[index] = 1  # This letter is still hidden and can be guessed

    return labels

def update_word_state(word, masked_word, guessed_char):
    """Updates the masked word based on the guessed character 
    and returns a boolean if the guess was correct."""
    new_masked_word = ""
    guess_correct = False
    for i in range(len(word)):
        if word[i] == guessed_char and masked_word[i] == '_':
            new_masked_word += guessed_char
            guess_correct = True
        else:
            new_masked_word += masked_word[i]
    return new_masked_word, guess_correct

# Generating: same singature as play_a_game_with_a_word
def generate_a_game_with_a_word(word, guess_function, model, \
                    solver, transform, initial_masked_word=None, sanity_test=False, process_word_fn=None):
    
    masked_word = "_" * len(word) if not initial_masked_word else initial_masked_word
    guessed_letters = []
    attempts_remaining = 6  # Total allowed incorrect guesses
    total_attempts = 0  # Total guesses made (incorrect ones)
    game_progress = []

    while "_" in masked_word and attempts_remaining > 0:
        # Log the current state before guessing
        game_progress.append((masked_word, None))  # Temporary None for the guess

        # guessed_char = guess_function(word, masked_word, solver, guessed_letters)  # simulate guessing
        # def simulated_guess_function(original_word, masked_word, \
        #                             solver, guessed_letters):        
        guessed_char = guess_function(original_word=word, masked_word=masked_word, \
                                    solver=solver, guessed_letters=guessed_letters)
        
        
        guessed_letters.append(guessed_char)  # Update guessed letters list

        new_masked_word, guess_correct = update_word_state(word, masked_word, guessed_char)
        if not guess_correct:  # Guess was incorrect
            total_attempts += 1
            attempts_remaining -= 1

        masked_word = new_masked_word

        # Update the last entry with the guessed character
        game_progress[-1] = (game_progress[-1][0], guessed_char)

        if masked_word == word:  # Check if the game is won
            break

    win = masked_word == word
    return win, total_attempts, game_progress

def save_game_progress_batch(game_progress_list, output_dir, batch_id):
    """Save the game progress to a Parquet file."""
    df = pd.DataFrame(game_progress_list)
    output_path = os.path.join(output_dir, f'game_progress_batch_{batch_id}.parquet')
    df.to_parquet(output_path)
    # print(f"Batch {batch_id} saved with {len(game_progress_list)} records.")

def process_games(word_list, guess_function, model, solver, transform, output_dir, batch_size=100):
    """Processes a list of words to generate game data in parallel and saves them in batches."""
    game_progress_list = []
    batch_id = 0
    total = len(word_list)

    # Use concurrent.futures for parallel processing with real-time updates
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {
            executor.submit(generate_a_game_with_a_word, word, \
                            guess_function, model, solver, transform): word
            for word in word_list
        }

        # As each future completes, process the result
        with tqdm(total=total, desc="Generating and processing games", unit="game") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    word = futures[future]
                    _, _, game_progress = future.result()
                    for masked_word, guessed_char in game_progress:
                        game_progress_list.append({
                            "Masked Word": masked_word,
                            "Guessed Char": guessed_char
                        })
                        # game_progress_list.append({
                        #     masked_word,
                        #     guessed_char
                        # })
                    # print(game_progress_list)
                    # Check if it's time to save a batch
                    if len(game_progress_list) >= batch_size:
                        save_game_progress_batch(game_progress_list, output_dir, batch_id)
                        game_progress_list = []  # Clear the list for the next batch
                        batch_id += 1
                        # tqdm.write(f"Starting new batch {batch_id}")
                except Exception as exc:
                    tqdm.write(f'Word {word} generated an exception: {exc}')
                finally:
                    pbar.update(1)

    # Handle the last batch separately if any data remains
    if game_progress_list:
        save_game_progress_batch(game_progress_list, output_dir, batch_id)
        tqdm.write(f"Final batch {batch_id} saved with {len(game_progress_list)} records")


from src.data_generation.strategy import HangmanFreqSolver
import argparse

def main(num_samples):
    corpus_path = '/media/sayem/510B93E12554BBD1/Hangman/data/words_250000_train.txt'
    corpus = read_word_list(corpus_path, num_samples=250_000)
    
    train_list = read_word_list(corpus_path, num_samples=int(250_000 * 0.8))
    test_list = corpus - train_list
    
    # for trian data    
    solver = HangmanFreqSolver(corpus=corpus)
    word_list = read_word_list(corpus_path, num_samples=num_samples)
    output_dir = f'/media/sayem/510B93E12554BBD1/dataset/{num_samples}'
    recreate_directory(output_dir)
    
    # Assuming `simulated_guess_function` is defined elsewhere
    process_games(word_list, simulated_guess_function, None, solver, None, output_dir)
    print("Simulation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Hangman game simulations.')
    parser.add_argument('num_samples', type=int, help='Number of samples to process')
    
    args = parser.parse_args()
    main(args.num_samples)

# def process_games(word_list, guess_function, model, solver, transform, output_dir, batch_size=100):
#     """Processes a list of words to generate game data in parallel and saves them in batches."""
#     with Pool(cpu_count()) as pool:
#         results = pool.starmap(generate_a_game_with_a_word, [
#             (word, guess_function, model, solver, transform) for word in word_list
#         ])

#     game_progress_list = []
#     batch_id = 0

#     for result in results:
#         # print(result)
#         _, _, game_progress = result
#         print(game_progress)
#         for masked_word, guessed_char in game_progress:
#             game_progress_list.append({
#                 "Masked Word": masked_word,
#                 "Guessed Char": guessed_char
#             })

#         if len(game_progress_list) >= batch_size:
#             save_game_progress_batch(game_progress_list, output_dir, batch_id)
#             game_progress_list = []  # Clear the list for the next batch
#             batch_id += 1
#             print(f"Starting new batch {batch_id}")

#     # Save any remaining data in the last batch
#     if game_progress_list:
#         save_game_progress_batch(game_progress_list, output_dir, batch_id)
#         print(f"Final batch {batch_id} saved with {len(game_progress_list)} records")


# def process_games(word_list, guess_function, model, solver, transform, output_dir, batch_size=100):
#     """Processes a list of words to generate game data in parallel and saves them in batches."""
#     with Pool(cpu_count()) as pool:
#         results = pool.starmap(generate_a_game_with_a_word, [
#             (word, guess_function, model, solver, transform) for word in word_list
#         ])

#     game_progress_list = []
#     batch_id = 0

#     # Wrap results processing with tqdm for progress display
#     for result in tqdm(results, desc="Processing games", unit="game"):
#         _, _, game_progress = result  # Unpacking the result tuple, ensure this matches your actual result structure
#         print(game_progress)
#         for masked_word, guessed_char in game_progress:
#             game_progress_list.append({
#                 "Masked Word": masked_word,
#                 "Guessed Char": guessed_char
#             })

#         if len(game_progress_list) >= batch_size:
#             save_game_progress_batch(game_progress_list, output_dir, batch_id)
#             game_progress_list = []  # Clear the list for the next batch
#             batch_id += 1
#             print(f"Starting new batch {batch_id}")

#     # Save any remaining data in the last batch
#     if game_progress_list:
#         save_game_progress_batch(game_progress_list, output_dir, batch_id)
#         print(f"Final batch {batch_id} saved with {len(game_progress_list)} records")


# def process_games(word_list, guess_function, model, solver, transform, output_dir, batch_size=100):
#     """Processes a list of words to generate game data in parallel and saves them in batches."""
#     # Setup a partial function with fixed arguments
#     partial_generate_game = partial(generate_a_game_with_a_word,
#                                     guess_function=guess_function,
#                                     model=model,
#                                     solver=solver,
#                                     transform=transform)
    
#     game_progress_list = []  # Initialize here before use
#     batch_id = 0  # Initialize batch_id before use

#     with Pool(cpu_count()) as pool:
#         print(len(word_list))
#         # Use imap_unordered for real-time updates with all arguments properly passed
#         tasks = [(word,) for word in word_list]  # Create a list of tuples with each word wrapped in a tuple
#         for result in tqdm(pool.imap_unordered(partial_generate_game, tasks),
#                            total=len(word_list), desc="Generating games", unit="game"):
            
#             print(result)

    #         print(result)
    #         _, _, game_progress = result
    #         print(game_progress)
    #         for masked_word, guessed_char in game_progress:
    #             game_progress_list.append({
    #                 "Masked Word": masked_word,
    #                 "Guessed Char": guessed_char
    #             })
            
    #         print(game_progress_list)

    #         # Check if it's time to save a batch
    #         if len(game_progress_list) >= batch_size:
    #             save_game_progress_batch(game_progress_list, output_dir, batch_id)
    #             game_progress_list = []  # Clear the list for the next batch
    #             batch_id += 1
    #             # print(f"Starting new batch {batch_id}")
    #             print('\n')

    # # Handle the last batch separately if any data remains
    # if game_progress_list:
    #     save_game_progress_batch(game_progress_list, output_dir, batch_id)
    #     print(f"Final batch {batch_id} saved with {len(game_progress_list)} records")
    #     print('\n')


# def process_games(word_list, guess_function, model, solver, transform, output_dir, batch_size=100):
#     """Processes a list of words to generate game data in parallel and saves them in batches."""
#     game_progress_list = []
#     batch_id = 0

#     # Use a manual tqdm bar to show progress since starmap doesn't support direct tqdm integration
#     with Pool(cpu_count()) as pool:
#         total = len(word_list)
#         with tqdm(total=total, desc="Generating games", unit="game") as pbar:
#             results = [pool.apply_async(generate_a_game_with_a_word, 
#                                         (word, guess_function, model, solver, transform),
#                                         callback=lambda _: pbar.update()) for word in word_list]
#             game_results = [result.get() for result in results]  # Collect results
            
#             # print(game_results)

#             # Process each game result with tqdm for real-time updates
#             for _, _, game_progress in tqdm(game_results, desc="Processing games", unit="game"):
#                 for masked_word, guessed_char in game_progress:
#                     game_progress_list.append({
#                         "Masked Word": masked_word,
#                         "Guessed Char": guessed_char
#                     })
#                 print(game_progress)
#                 # Check if it's time to save a batch
#                 if len(game_progress_list) >= batch_size:
#                     save_game_progress_batch(game_progress_list, output_dir, batch_id)
#                     game_progress_list = []  # Clear the list for the next batch
#                     batch_id += 1
#                     tqdm.write(f"Starting new batch {batch_id}")

#             # Handle the last batch separately if any data remains
#             if game_progress_list:
#                 save_game_progress_batch(game_progress_list, output_dir, batch_id)
#                 tqdm.write(f"Final batch {batch_id} saved with {len(game_progress_list)} records")
