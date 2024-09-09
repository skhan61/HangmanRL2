import torch
import numpy as np

def single_batch_mask_maker(input_string, guessed_letters):
    """
    Encodes a string where 'a' to 'z' are mapped to 0 to 25 and '_' is mapped to 26,
    then creates a tensor along with a mask for a randomly chosen missed letter that is 
    not in the input string or guessed letters.

    Args:
        input_string (str): The string to encode, containing characters from 'a' to 'z' and '_'.
        guessed_letters (list): List of characters that have been guessed.

    Returns:
        dict: Contains the input tensor, mask for a randomly selected missed letter, and length of input.
    """
    char_to_index = {chr(i): i - 97 for i in range(97, 123)}  # Mapping 'a' to 'z'
    char_to_index['_'] = 26  # Mapping '_' for blanks
    
    # Encode the string using the mapping
    encoded = [char_to_index.get(char, char_to_index['_']) for char in input_string]
    inputs = torch.tensor([encoded], dtype=torch.long)

    # # Determine all possible characters
    all_chars = set(char_to_index.keys()) - {'_'}
    # used_chars = set(input_string.replace('_', '')) | set(guessed_letters)

    missing_characters = np.array(sorted(list(all_chars - set(input_string))))
    # print(missing_characters)
    number_of_misses = np.random.randint(0, 10)
    missing_character_indices = np.random.choice(missing_characters, number_of_misses)

    missing_character_indices = list(set([char_to_index[char] for char in missing_character_indices]))

    missing_vector = np.zeros(26)
    missing_vector[missing_character_indices] = 1

    # missing_vector = torch.from_numpy(missing_vector).float()

    missing_vector = torch.from_numpy(missing_vector).float().unsqueeze(0)  # Convert to tensor and add batch dimension


    # print(missing_vector.shape)

    # Prepare the batch dictionary
    single_batch = {
        'inputs': inputs,
        'miss_chars': missing_vector,
        'lengths': torch.tensor([len(encoded)], dtype=torch.long)
    }

    # Create a mask for guessed letters
    mask = np.zeros(26)  # Only 'a' to 'z', no need to mask '_'
    for letter in guessed_letters:
        if letter in char_to_index and letter != '_':  # Ensure '_' is not included in the mask
            mask[char_to_index[letter]] = 1

    # Convert mask to numpy array first and then to tensor
    mask_array = np.array([mask])  # Convert list to numpy array
    mask_tensor = torch.from_numpy(mask_array).float()  # Create tensor from numpy array

    # print(mask_tensor.shape)

    return single_batch, mask_tensor

# # Example usage
# input_string = "worl_"
# guessed_letters = ['w', 'o', 'r', 'l', 'a', 'b']  # Example set of guessed letters
# single_batch, single_mask = single_batch_mask_maker(input_string, guessed_letters)
# # print("Batch data:", batch)
# # print("Mask:", mask)