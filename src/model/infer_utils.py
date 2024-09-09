import numpy as np
import torch
from collections import OrderedDict

def convert_state_to_numeric(state_str, max_len=35):
    """Convert state string to numeric array where 'a' = 1, ..., 'z' = 26, and '_' = 27.
       Pad the rest of the array up to max_len with 0 to indicate no character."""
    numeric = [ord(char) - ord('a') + 1 if 'a' <= char <= 'z' else 27 for char in state_str]
    padded_numeric = numeric + [0] * (max_len - len(numeric))
    return np.array(padded_numeric)

def convert_guessed_letters_to_binary(guessed_letters):
    """Convert a list of guessed letters to a binary representation where 
       each position corresponds to a letter (a-z)."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    binary = np.zeros(26, dtype=int)
    for letter in guessed_letters:
        index = ord(letter.lower()) - ord('a')
        binary[index] = 1
    return binary


def create_action_mask(guessed_letters_binary):
    """Create an action mask based on guessed letters 
    where guessed letters are marked as 0 (not selectable)."""
    return 1 - guessed_letters_binary

def one_hot_encode_action(action):
    """
    One-hot encode a given action. If the action is a character, convert it
    to the corresponding index first.

    Args:
        action (int or str): Action to be one-hot encoded. Can be an integer index or a character.

    Returns:
        numpy.ndarray: One-hot encoded vector.
    """
    one_hot_action = np.zeros(26, dtype=np.int32)
    if action is not None:
        if isinstance(action, str):
            # Convert character to index
            action = ord(action.lower()) - ord('a')
        one_hot_action[action] = 1
    return one_hot_action



def create_observation_and_action_mask(masked_word, guessed_letters, remaining_attempts,
                                       previous_action, recurrent_states_actor,
                                       recurrent_states_critic, recurrent_state_size):
    """
    Create an observation array and a separate array for action masks including previous action, reward, and done.
    If recurrent states are None, initialize them.
    """
    guessed_letters_binary = convert_guessed_letters_to_binary(guessed_letters)

    guessed_letters_array = guessed_letters_binary.reshape(1, -1)
    masked_word_array = convert_state_to_numeric(masked_word).reshape(1, -1)
    remaining_attempts_array = np.array([[remaining_attempts]], dtype=int)    
    action_mask = create_action_mask(guessed_letters_binary).reshape(1, -1)


    # game observation
    # Concatenating all arrays to form a single observation array
    observation = np.concatenate([
        masked_word_array,
        guessed_letters_array,
        remaining_attempts_array,
        action_mask
    ], axis=1)  # Make sure all arrays are aligned on the second axis (columns)

    # print(observation.shape)

    # rl^2 observation
    # Determine reward and done status based on the letter
    if previous_action:
        # Previous action is a letter
        letter = previous_action
        reward = 1 if letter in masked_word else -1
        done = False
    else:
        # No previous action given
        letter = None
        reward = 0.0
        done = False

    # print(one_hot_encode_action(letter))

    # One-hot encode the previous action if it exists
    previous_action_array = one_hot_encode_action(letter).reshape(1, -1) \
        if letter else np.zeros((1, 26))

    reward_array = np.array([[reward]], dtype=float)
    done_array = np.array([[done]], dtype=int)

    # Initialize recurrent states if they are None
    if recurrent_states_actor is None or recurrent_states_critic is None:
        # Example initialization for a single instance, adjust dimensions as necessary for your application
        recurrent_states_actor = torch.zeros(1, recurrent_state_size)
        recurrent_states_critic = torch.zeros(1, recurrent_state_size)

    # Combine all parts of the observation into a single array
    observation_array = np.concatenate([
        observation,
        previous_action_array,
        reward_array,
        done_array
        ], axis=1)  # Ensure correct axis for concatenation

    observation_array = torch.from_numpy(observation_array).float()
    action_mask = torch.from_numpy(action_mask).float()

    return observation_array, action_mask, \
           recurrent_states_actor, recurrent_states_critic


# def main():
#     # Example usage for a single batch
#     masked_word = '___'
#     guessed_letters = ['e', 'i']  # Example where some letters have been guessed
#     remaining_attempts = 4

#     observation_dict, action_mask \
#         = create_observation_and_action_mask(masked_word, \
#                         guessed_letters, remaining_attempts)
#     print("Observation Dict:", observation_dict)
#     print("Action Mask:", action_mask)

# if __name__ == '__main__':
#     main()
