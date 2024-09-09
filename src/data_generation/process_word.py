import torch
from src.datamodule.dataset import encode_character

def process_word(word, transform):
    """
    Process a single word to get its encoded state, length, and transformed features.

    Args:
    word (str): The word to process.
    encode_character (function): Function to encode each character of the word.
    solver (function): Not used directly in this example but can be integrated if needed.
    transform (function): Function to apply transformations to the encoded word tensor.

    Returns:
    dict: A dictionary containing the states, lengths, and features of the word.
    """
    # Step 1: Encode the characters of the word into state sequences
    state_sequence = [encode_character(char) for char in word]
    state_tensor = torch.tensor(state_sequence, dtype=torch.long) #.unsqueeze(0)  # Adding batch dimension

    # print(state_tensor.shape)

    # # Wrap the tensor in a structure that transform expects (e.g., list of dicts)
    batch_for_transform = [{'states': state_tensor}]

    # Step 2: Apply transformations if any
    features = transform(batch_for_transform)  # Assuming transform can handle this structure

    state_tensor = torch.tensor(state_sequence, dtype=torch.long).unsqueeze(0)  # Adding batch dimension

    # Step 3: Create a tensor for the length of the word
    lengths_tensor = torch.tensor([len(word)], dtype=torch.long)

    # Package the results into a dictionary to return
    result = {
        'states': state_tensor,
        'lengths': lengths_tensor,
        'features': features
    }

    return result