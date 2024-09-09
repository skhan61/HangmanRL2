import torch
import string
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
# from tqdm.notebook import tqdm
from supervised_learning.model.inference_utils import single_batch_mask_maker

from tqdm import tqdm
import torch
import string

from src.env import HangmanEnv

def guess(model, word, guessed_letters): # TODO

    # print(model.get_device()) 
    # print(word)
    # # Clean and prepare the word for processing
    # masked_word = word[::2] # .replace("_",".")
    # print(f"Cleaned word: {len(masked_word)}")

    # # word == masked_word 
    # print(word)
    # print(type(model))

    # print(f"-------------------------------------------")
    # print(f"Data available to make next guess...")
    # print(f"current state: {word}")
    # print(f"Gussed letters: {guessed_letters}")
    # # print(f"Tries remainig: {type(tries_remains)} / {6}")
    # print(f"-------------------------------------------")

    single_batch, mask = single_batch_mask_maker(word, guessed_letters)

    # print(model.device)
    # # print(single_batch)
    # # print(mask)

    guessed_char = model.next_guess(single_batch, mask)

    # Update the guessed letters list
    if guessed_char not in guessed_letters:
        guessed_letters.append(guessed_char)

    return guessed_char

def update_word_state(word, masked_word, guessed_char):
    """Updates the masked word based on the guessed character 
    and returns a boolean if the guess was correct."""
    new_masked_word = ""
    guess_correct = False
    for i in range(len(word)):
        if word[i] == guessed_char:
            if masked_word[i] == '_':
                new_masked_word += guessed_char
                guess_correct = True
            else:
                new_masked_word += word[i]
        else:
            new_masked_word += masked_word[i]

    # print(f"Debug: Word='{word}', Guessed='{guessed_char}', Old Masked='{masked_word}', New Masked='{new_masked_word}', Correct={guess_correct}")
    return new_masked_word, guess_correct


## Motoring 
def play_a_game_with_a_word(word, guess_function,
                            model, initial_masked_word=None, 
                            ):
    
    # # print(f"play_a_game_with_a_word {type(model)}")
    # print(model)
    # print('\n')

    if initial_masked_word:
        masked_word = initial_masked_word
    else:
        masked_word = "_" * len(word)

    guessed_letters = [char for char in masked_word if char != "_"]
    attempts_remaining = 6  # Total allowed incorrect guesses
    total_attempts = 0  # Total guesses made (incorrect ones)
    game_progress = []

    while "_" in masked_word and attempts_remaining > 0:
        # print(guessed_letters)
        guessed_char = guess_function(
                    word=masked_word, 
                    model=model,
                    guessed_letters=guessed_letters,   
                )
        
        # print(guessed_char)

        new_masked_word, guess_correct = update_word_state(word, masked_word, \
                                                           guessed_char)
        # print(new_masked_word)
        if not guess_correct:  # Guess was incorrect
            total_attempts += 1
            attempts_remaining -= 1

        masked_word = new_masked_word
        game_progress.append((guessed_char, masked_word, guess_correct))

        if masked_word == word:
            break

        # print(f"Guess: '{guessed_char}', New Masked Word: '{masked_word}', Correct: {guess_correct}")

        # print('\n')

    win = masked_word == word
    return win, attempts_remaining, game_progress


#v2
def simulate_games_for_word_list(word_list, guess_function, play_function, model): # TODO: aggregated_data=None: remove later

    results = {}
    total_wins = 0
    total_losses = 0
    total_tries_remaining = []

    for idx, word in enumerate(tqdm(word_list, desc="Simulating games", unit="word")):
        win, tries_remaining, game_progress \
            = play_function(word=word, guess_function=guess_function,
                        model=model, initial_masked_word=None)

        word_length = len(word)

        if word_length not in results:
            results[word_length] = {'wins': 0, 'losses': 0, \
                        'total_tries_remaining': [], 'games': []}

        results[word_length]['games'].append({
            'word': word,
            'win': win,
            'tries_remaining': tries_remaining,
            'progress': game_progress
        })

        results[word_length]['total_tries_remaining'].append(tries_remaining)
        if win:
            results[word_length]['wins'] += 1
            total_wins += 1
        else:
            results[word_length]['losses'] += 1
            total_losses += 1

        total_tries_remaining.append(tries_remaining)

    overall_win_rate = total_wins / len(word_list) if word_list else 0
    average_tries_remaining = sum(total_tries_remaining) / len(total_tries_remaining) if total_tries_remaining else 0

    aggregated_results = {}
    for length, data in results.items():
        avg_tries_remaining = sum(data['total_tries_remaining']) / len(data['total_tries_remaining']) if data['total_tries_remaining'] else 0
        win_rate = data['wins'] / (data['wins'] + data['losses']) if (data['wins'] + data['losses']) > 0 else 0
        aggregated_results[length] = {
            'average_tries_remaining': avg_tries_remaining,
            'win_rate': win_rate,
            'total_games': data['wins'] + data['losses']
        }

    return {
        'results_by_length': aggregated_results,
        'overall': {
            'total_games': len(word_list),
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': overall_win_rate,
            'average_tries_remaining': average_tries_remaining
        }
    }





# from tqdm import tqdm

# def simulate_games_for_word_list(word_list, guess_function, play_function, 
#                                  model, solver, transform, process_word_fn):
#     results = {}
#     total_wins = 0
#     total_losses = 0
#     total_tries_remaining = []
#     total_steps_per_game = []  # Track number of steps taken to conclude each game

#     for idx, word in enumerate(tqdm(word_list, desc="Simulating games", unit="word")):
#         win, tries_remaining, game_progress = play_function(word=word, guess_function=guess_function,
#                                                             model=model, solver=solver, transform=transform, 
#                                                             initial_masked_word=None, process_word_fn=process_word_fn)

#         unique_letters_count = len(set(word))  # Number of unique letters in the word

#         if unique_letters_count not in results:
#             results[unique_letters_count] = {
#                 'wins': 0, 
#                 'losses': 0, 
#                 'total_tries_remaining': [], 
#                 'games': [],
#                 'total_steps': []  # Track steps for games grouped by unique letter count
#             }

#         results[unique_letters_count]['games'].append({
#             'word': word,
#             'win': win,
#             'tries_remaining': tries_remaining,
#             'progress': game_progress,
#             'steps': len(game_progress)  # Assuming game_progress records each step
#         })

#         results[unique_letters_count]['total_tries_remaining'].append(tries_remaining)
#         results[unique_letters_count]['total_steps'].append(len(game_progress))

#         if win:
#             results[unique_letters_count]['wins'] += 1
#         results[unique_letters_count]['losses'] += 1

#         total_wins += win
#         total_losses += 1 - win
#         total_tries_remaining.append(tries_remaining)
#         total_steps_per_game.append(len(game_progress))

#     overall_win_rate = total_wins / len(word_list) if word_list else 0
#     average_tries_remaining = sum(total_tries_remaining) / len(total_tries_remaining) if total_tries_remaining else 0
#     average_steps_per_game = sum(total_steps_per_game) / len(total_steps_per_game) if total_steps_per_game else 0

#     # Compute macro average for win rate
#     aggregated_results = {}
#     total_games_per_unique_count = [data['wins'] + data['losses'] for data in results.values()]
#     macro_avg_win_rate = sum((data['wins'] / (data['wins'] + data['losses']) \
#                             if (data['wins'] + data['losses']) > 0 else 0 for data in results.values())) \
#                             / len(results)

#     for unique_letters, data in results.items():
#         avg_tries_remaining = sum(data['total_tries_remaining']) / len(data['total_tries_remaining'])
#         win_rate = data['wins'] / (data['wins'] + data['losses']) if (data['wins'] + data['losses']) > 0 else 0
#         avg_steps = sum(data['total_steps']) / len(data['total_steps'])
#         aggregated_results[unique_letters] = {
#             'average_tries_remaining': avg_tries_remaining,
#             'win_rate': win_rate,
#             'total_games': data['wins'] + data['losses'],
#             'average_steps_per_game': avg_steps
#         }

#     return {
#         'results_by_unique_letters': aggregated_results,
#         'overall': {
#             'total_games': len(word_list),
#             'wins': total_wins,
#             'losses': total_losses,
#             'win_rate': macro_avg_win_rate,  # Use macro average win rate
#             'average_tries_remaining': average_tries_remaining,
#             'average_steps_per_game': average_steps_per_game
#         }
#     }




def play_a_game_until_first_correct_guess(word, guess_function, model, solver, transform, 
                                        initial_masked_word=None, aggregated_data=None, \
                                        sanity_test=False):
    if initial_masked_word:
        masked_word = initial_masked_word
    else:
        masked_word = "_" * len(word)

    guessed_letters = [char for char in masked_word if char != "_"]
    attempts_remaining = 6
    total_attempts = 0
    game_progress = []

    while "_" in masked_word and attempts_remaining > 0:
        guessed_char = guess_function(word=masked_word, model=model, 
                                      solver=solver,
                                      guessed_letters=guessed_letters, 
                                      transform=transform,
                                      aggregated_data=aggregated_data)

        # Update the guessed letters list
        if guessed_char not in guessed_letters:
            guessed_letters.append(guessed_char)
        # print(f"Updated guessed letters: {guessed_letters}")
    
        new_masked_word, guess_correct = update_word_state(word, masked_word, guessed_char)
        total_attempts += 1

        game_progress.append((guessed_char, masked_word, guess_correct))
        
        if guess_correct:
            return True, attempts_remaining, game_progress  # Return immediately upon the first correct guess

        if not guess_correct:
            attempts_remaining -= 1  # Decrement attempts only on incorrect guesses

        masked_word = new_masked_word  # Update the masked word
        guessed_letters.append(guessed_char)

    return False, attempts_remaining, game_progress  # Return if no correct guess is made



# ## Motoring 
# def play_a_game_with_a_batch(words, guess_function,
#                             model, solver, transform,
#                             process_word_fn, 
#                             initial_masked_word=None, 
#                             ): # TODO: aggregated_data=None: remove later
    
#     # # print(f"play_a_game_with_a_word {type(model)}")
#     # print(model)
#     # print('\n')
#     env = HangmanEnv(word) # -> call it as vectorized way. for each word there will be a independt game
#     env.reset()

#     if initial_masked_word:
#         masked_word = initial_masked_word
#     else:
#         masked_word = "_" * len(word)

#     guessed_letters = [char for char in masked_word if char != "_"]
#     attempts_remaining = 6  # Total allowed incorrect guesses
#     total_attempts = 0  # Total guesses made (incorrect ones)
#     game_progress = []

#     while "_" in masked_word and attempts_remaining > 0:
#         # # if sanity_test:
#         # #     guessed_char = guess_function(word, masked_word, \
#         # solver, guessed_letters) # dummy for testing
#         # # else:
#         ## dont delete
#         # guessed_char = guess_function(word=masked_word, model=model, \
#         #             solver=solver, guessed_letters=guessed_letters, \
#         #             transform=transform, process_word_fn=process_word_fn) 
#         # # else:

#         # action generator
#         guessed_char = guess_function(word=masked_word, model=model, \
#                     solver=solver, guessed_letters=guessed_letters, \
#                     transform=transform, process_word_fn=process_word_fn) 
#         # else:

#         # # src.datageneration.simulated_guess_function signature
#         # guessed_char = guess_function(word, masked_word, solver, guessed_letters) 
#         # # TODO: aggregated_data=None: remove later) # guess function
        
#         # Update the guessed letters list
#         if guessed_char not in guessed_letters:
#             guessed_letters.append(guessed_char)
#         # print(f"Updated guessed letters: {guessed_letters}")

#         new_masked_word, guess_correct = update_word_state(word, masked_word, \
#                                                            guessed_char)
        

#         if not guess_correct:  # Guess was incorrect
#             total_attempts += 1
#             attempts_remaining -= 1

#         masked_word = new_masked_word
#         game_progress.append((guessed_char, masked_word, guess_correct))

#         if masked_word == word:
#             break

#         # print(f"Guess: '{guessed_char}', New Masked Word: '{masked_word}', Correct: {guess_correct}")

#         # print('\n')

#     win = masked_word == word
#     return win, attempts_remaining, game_progress
