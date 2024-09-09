import torch
import string
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm.notebook import tqdm

import torch
import string


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

## Motoring 
def play_a_game_with_a_word(word, guess_function,
                            model, solver, transform,
                            process_word_fn, 
                            initial_masked_word=None, 
                            ): # TODO: aggregated_data=None: remove later
    
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
        # # if sanity_test:
        # #     guessed_char = guess_function(word, masked_word, solver, guessed_letters) # dummy for testing
        # # else:
        ## dont delete
        # guessed_char = guess_function(word=masked_word, model=model, \
        #             solver=solver, guessed_letters=guessed_letters, \
        #             transform=transform, process_word_fn=process_word_fn) 
        # # else:

        # action
        guessed_char = guess_function(word=masked_word, model=model, \
                    solver=solver, guessed_letters=guessed_letters, \
                    transform=transform, process_word_fn=process_word_fn) 
        # else:

        # # src.datageneration.simulated_guess_function signature
        # guessed_char = guess_function(word, masked_word, solver, guessed_letters) # TODO: aggregated_data=None: remove later) # guess function
            
        # Update the guessed letters list
        if guessed_char not in guessed_letters:
            guessed_letters.append(guessed_char)
        # print(f"Updated guessed letters: {guessed_letters}")

        new_masked_word, guess_correct = update_word_state(word, masked_word, \
                                                           guessed_char)
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
def simulate_games_for_word_list(word_list, guess_function, play_function, 
                                 model, solver, transform, process_word_fn): # TODO: aggregated_data=None: remove later

    results = {}
    total_wins = 0
    total_losses = 0
    total_tries_remaining = []

    for idx, word in enumerate(tqdm(word_list, desc="Simulating games", unit="word")):
        win, tries_remaining, game_progress = play_function(word=word, guess_function=guess_function,
                                                            model=model, solver=solver, transform=transform, 
                                                            initial_masked_word=None, process_word_fn=process_word_fn)

        word_length = len(word)

        if word_length not in results:
            results[word_length] = {'wins': 0, 'losses': 0, 'total_tries_remaining': [], 'games': []}

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



# #v1
# def simulate_games_for_word_list(word_list, guess_function, play_function, \
#                                 model, solver, transform, sanity_test=True, \
#                                 aggregated_data=None): # TODO: aggregated_data=None: remove later):
    
#     # print(f"simulate_games_for_word_list {type(model)}")
#     # print('\n')

#     results = {}
#     total_wins = 0
#     total_losses = 0
#     total_attempts = []

#     for idx, word in enumerate(tqdm(word_list, desc="Simulating games", unit="word")):
#         # win, attempts, game_progress = play_a_game_with_a_word(word=word, guess_function=guess_function,
#         #                     model=model, solver=solver, transform=transform, \
#         #                     initial_masked_word=None, sanity_test=sanity_test, \
#         #                     aggregated_data=aggregated_data) # TODO: aggregated_data=None: remove later)

#         win, attempts_remaining, game_progress = play_function(word=word, guess_function=guess_function,
#                             model=model, solver=solver, transform=transform, \
#                             initial_masked_word=None, sanity_test=sanity_test, \
#                             aggregated_data=aggregated_data) # TODO: aggregated_data=None: remove later)
        
#         word_length = len(word)

#         if word_length not in results:
#             results[word_length] = {'wins': 0, 'losses': 0, 'total_attempts': [], 'games': []}

#         results[word_length]['games'].append({
#             'word': word,
#             'win': win,
#             'attempts': attempts_remaining,
#             'progress': game_progress
#         })

#         results[word_length]['total_attempts'].append(attempts_remaining)
#         if win:
#             results[word_length]['wins'] += 1
#             total_wins += 1
#         else:
#             results[word_length]['losses'] += 1
#             total_losses += 1

#         total_attempts.append(attempts_remaining)

#     overall_win_rate = total_wins / len(word_list) if word_list else 0
#     average_attempts = sum(total_attempts) / len(total_attempts) if total_attempts else 0

#     return {
#         'results_by_length': results,
#         'overall': {
#             'total_games': len(word_list),
#             'wins': total_wins,
#             'losses': total_losses,
#             'win_rate': overall_win_rate,
#             'average_attempts': average_attempts
#         }
#     }


