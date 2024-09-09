import torch
import string
import random
import collections
# from src.datamodule import ProcessWordTransform
# from src.datamodule import encode_character
# from src.model.strategy import HangmanFreqSolver
import numpy as np
## solver will take care the gussing algos with any fallback
from sklearn.preprocessing import MultiLabelBinarizer
from stable_baselines3.common.utils import obs_as_tensor
from src.model.infer_utils import create_observation_and_action_mask
import torch

## This will take care the NN input output

## mimicing API
def update_game_stage(masked_word, guessed_letters):
    word_len = len(masked_word)
    # print(word_len)
    # Determine incorrect guesses based on their presence in the masked word
    incorrect_guesses = {letter for letter in guessed_letters \
            if all(letter != ch for ch in masked_word if ch != '_')}
    # print(incorrect_guesses)
    incorrect_guesses_count = len(incorrect_guesses)
    remaining_attempts = 6 - incorrect_guesses_count
    # print(remaining_attempts)
    return remaining_attempts

# def guess(model, word, solver, guessed_letters, \
#           transform, process_word_fn):

def guess(model, word, \
        guessed_letters, previous_action, \
        recurrent_states_actor, recurrent_states_critic): # TODO

    # print(model.get_device()) 
    # print(word)
    # # Clean and prepare the word for processing
    # masked_word = word[::2] # .replace("_",".")
    # print(f"Cleaned word: {len(masked_word)}")

    # # word == masked_word 
    # print(word)
    # print(type(model))
    tries_remains = update_game_stage(word, guessed_letters) 
    # print(f"-------------------------------------------")
    # print(f"Data available to make next guess...")
    # print(f"current state: {type(word)}")
    # print(f"Gussed letters: {type(guessed_letters)}")
    # print(f"Tries remainig: {type(tries_remains)} / {6}")
    # print(f"-------------------------------------------")

    obs, action_masks, recurrent_states_actor, recurrent_states_critic \
        = create_observation_and_action_mask(word, \
                guessed_letters, tries_remains, previous_action, \
                recurrent_states_actor, recurrent_states_critic, \
                recurrent_state_size=model.recurrent_state_size)


    # print(obs.shape)
    # print(action_masks.shape)
    # print(action_masks.shape)

    # print(recurrent_states_actor.shape)
    # print(recurrent_states_critic.shape)


    actions, recurrent_states_actor, recurrent_states_critic \
        = model.predict(
        obs.to(model.get_device()),
        action_masks.to(model.get_device()),
        recurrent_states_actor.to(model.get_device()),
        recurrent_states_critic.to(model.get_device()),
    )

    # print(actions)

    # print(action_masks)

    # print(type(action_distribution)) # .logits())

    # print(action_distribution.logits)

    # if action_masks is not None:
    #     action_distribution.apply_mask(action_masks)

    # guessed_char_index = action_distribution.mode()  # Gets the scalar value from the tensor
    
    guessed_char_index = actions.item()
    # print(guessed_char_index)
    guessed_char = chr(ord('a') + int(guessed_char_index))
    
    # Update the guessed letters list
    if guessed_char not in guessed_letters:
        guessed_letters.append(guessed_char)
    # print(f"Updated guessed letters: {guessed_letters}")
    
    return guessed_char, recurrent_states_actor, \
        recurrent_states_critic

# def guess_character(
#                     model, \
#                     masked_word, \
#                     guessed_letters, \
#                     tries_remains,
#                     ):
    
#     # # This will take care the NN input output
#     print(f"-------------------------------------------")
#     print(f"Data available to make next guess...")
#     print(f"current state: {type(masked_word)}")
#     print(f"Gussed letters: {type(guessed_letters)}")
#     print(f"Tries remainig: {type(tries_remains)} / {6}")
#     print(f"-------------------------------------------")
#     # # print(model.device)
#     # batch = process_word_fn(masked_word, transform)
#     obs, action_mask = create_observation_and_action_mask(masked_word, \
#                                             guessed_letters, tries_remains)
    
#     print(obs)
#     # masked_word = batch['states']
#     # print(type(masked_word))
#     # print(type(guessed_letters))
#     # print(type(tries_remains))

#     # print(obs)

#     # batched_input = prepare_batch(obs)
#     # print(batched_input)

#     # input_dict = {Columns.OBS: obs}

#     # print(input_dict)

#     # print(batched_input)

#     # character_indx, state_outs, info \
#     #     = model.compute_single_action(input_dict=input_dict)

#     # print(character_indx)
#     # print(state_outs)
#     # print(info)
#     # obs = obs_as_tensor(obs, device='cpu')
#     # # # print(obs)
#     # # print(obs['action_mask'])

#     # character_indx, _ = model.predict(obs, action_masks=action_mask)

#     # best_char_index = character_indx.item()
#     # best_guess = string.ascii_lowercase[best_char_index]

#     best_guess = 'a'

#     return best_guess