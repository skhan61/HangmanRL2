import numpy as np
import torch
import numpy as np


def encode_word(word, vocab_size, \
                min_length, char_to_index):
    """
    Encodes a word into a one-hot encoded matrix and gathers information about character positions.
    
    Args:
    - word (str): The word to encode.
    - vocab_size (int): The number of unique characters in the vocabulary.
    - min_length (int): Minimum length for a word to be considered.
    - char_to_index (dict): Mapping from character to their index in the vocabulary.
    
    Returns:
    - np.array: One-hot encoded matrix for the word if it meets the length requirement, None otherwise.
    - list: List of character positions in the word if applicable, None otherwise.
    - set: Set of unique characters in the word, None if word is below the minimum length.
    """
    
    # Clean and verify the word's length
    word = word.strip().lower()
    if len(word) < min_length:
        return None, None, None

    # Reserve an additional spot in the vocabulary for an extra character, such as 'blank'
    extra_vocab = 1 # we dont need it here

    # Initialize the one-hot encoded matrix
    encoded_matrix = np.zeros((len(word), vocab_size + extra_vocab))

    # Dictionary to track positions of each character in the word
    character_positions = {index: [] for index in range(vocab_size)}

    # Fill the encoded matrix and character positions
    for position, char in enumerate(word):
        char_index = char_to_index[char]  # Get the index of the character from the mapping
        character_positions[char_index].append(position)
        encoded_matrix[position][char_index] = 1  # Set the corresponding position in the matrix to 1

    # List of positions for each character that appears in the word
    positions_list = [positions for positions in character_positions.values() if positions]

    return encoded_matrix, positions_list, set(word)

def custom_collate_fn(batch):
    # print(batch)
    """Custom collate function to handle padding and batch creation.

    Args:
        batch (list of tuples): Each tuple contains (input_array, label_array, miss_chars_array)

    Returns:
        dict: Dictionary containing tensors for inputs, labels, miss_chars, and lengths.
    """
    vocab_size = 28  # Include all letters plus special tokens like 'MASK'
    padded_value = vocab_size - 1  # Assuming the last index is used for padding

    # Determine the maximum sequence length in this batch
    max_len = max(len(item[0]) for item in batch)

    # Create batches for inputs, labels, miss_chars with appropriate padding
    inputs = torch.stack([
        torch.tensor(np.pad(item[0], (0, max_len - len(item[0])), mode='constant', constant_values=padded_value))
        for item in batch
    ])
    labels = torch.stack([torch.tensor(item[1]) for item in batch])
    miss_chars = torch.stack([torch.tensor(item[2]) for item in batch])
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)

    return {
        'inputs': inputs,
        'labels': labels,
        'miss_chars': miss_chars,
        'lengths': lengths
    }


# # Example usage:
# vocab_size = 26  # Typically 26 for lowercase English letters
# min_length = 4
# char_to_index = {chr(97 + i): i for i in range(vocab_size)}  # Creates a mapping for 'a' to 'z'

# word = "hello"
# encoded_matrix, positions_list, unique_chars = encode_word(word, vocab_size, min_length, char_to_index)
# print("Encoded Matrix:\n", encoded_matrix)
# print("Character Positions:", positions_list)
# print("Unique Characters:", unique_chars)
