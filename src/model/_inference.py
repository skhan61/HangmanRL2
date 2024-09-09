import torch
import string
import random
import collections
from src.datamodule import ProcessWordTransform
from src.datamodule import encode_character
# from src.model.strategy import HangmanFreqSolver
import numpy as np
## solver will take care the gussing algos with any fallback
from sklearn.preprocessing import MultiLabelBinarizer

## This will take care the NN input output
def guess_character(
                    masked_word, model, \
                    guessed_letters, \
                    tries_remains,
                    solver, \
                    transform, \
                    process_word_fn):
    
    # # This will take care the NN input output
    # print(f"-------------------------------------------")
    # print(f"Data available to make next guess...")
    # print(f"current state: {masked_word}")
    # print(f"Gussed letters: {set(guessed_letters)}")
    # print(f"Tries remainig: {tries_remains} / {6}")
    # print(f"-------------------------------------------")

    batch = process_word_fn(masked_word, transform)
    # # print(batch)

    # # print(model)

    # # print(model.device)

    # model.eval()
    # with torch.no_grad():
    #     # # _, letter_likelihoods = model(batch)
    #     _, ranking_score, _ = model(batch)

    # ranking_score = ranking_score.squeeze().cpu() # it cant be all 0

    # # if torch.all(letter_likelihoods == 1):
    # #     print(f"All 1 happens")

    # # # Check if all elements are zero or one
    # # if torch.all(letter_likelihoods == 0):
    # #     # if torch.all(letter_likelihoods == 0):
    # #     print("All letter likelihoods are zero, triggering fallback mechanism.")
    # #     # else:
    # #     #     print("All letter likelihoods are one, triggering fallback mechanism.")
    # #     # Fallback mechanism: Could be to reinitialize the model, adjust parameters, or use a different model
    # #     # Example fallback: Random guess or heuristic-based approach
    # #     letter_likelihoods = torch.randint(low=0, high=2, size=(26,))  # Randomly initialize likelihoods as a simple example
    # #     if torch.all(letter_likelihoods == 0) or torch.all(letter_likelihoods == 1):
    # #         raise ValueError("Fallback also resulted in zero/one likelihoods, \
    # #                          check model configuration.")

    # # # # Convert the PyTorch tensor to a NumPy array
    # # letter_likelihoods = letter_likelihoods.numpy()
    # # # Convert the PyTorch tensor to a NumPy array
    # ranking_score = ranking_score.numpy()
    
    # # print(letter_likelihoods)

    # # # # print(letter_likelihoods.shape)
    # # print(model.base_rate)

    # # position_frequencies, position_counts = analyze_corpus(solver.corpus)

    # # letter_likelihoods = calculate_aggregated_letter_probabilities(masked_word, \
    # #                                         position_frequencies, position_counts)

    # # letter_likelihoods = np.full(26, 0) # np.random.randint(0, 2, size=26) #np.full(26, 0) # TODO: remove latter

    # best_char_index = solver.next_guess(masked_word, \
    #                         ranking_score, guessed_letters, tries_remains)
    # # print("Determined best character index from solver:", best_char_index)
    best_char_index = 0
    best_guess = string.ascii_lowercase[best_char_index]

    return best_guess

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
def guess(model, word, solver, guessed_letters, \
          transform, process_word_fn): # TODO 
    # print(word)
    # # Clean and prepare the word for processing
    # masked_word = word[::2] # .replace("_",".")
    # print(f"Cleaned word: {len(masked_word)}")

    # print(type(model))
    tries_remains = update_game_stage(word, guessed_letters)
    # print(tries_remains)    
    guessed_char = guess_character(masked_word=word, model=model, \
                        guessed_letters=guessed_letters, \
                        tries_remains=tries_remains, solver=solver, \
                        transform=transform,
                        process_word_fn=process_word_fn)

    # guessed_char = 'a'
    
    # # Update the guessed letters list
    # if guessed_char not in guessed_letters:
    #     guessed_letters.append(guessed_char)
    # # print(f"Updated guessed letters: {guessed_letters}")
    
    return guessed_char


# ## This will take care the NN input output
# def guess_character(original_word, # TODO
#                     masked_word, model, \
#                     guessed_letters, \
#                     tries_remains,
#                     solver, \
#                     transform, \
#                     process_word_fn):
    
#     # # This will take care the NN input output
#     # print(f"-------------------------------------------")
#     # print(f"Data available to make next guess...")
#     # print(f"current state: {masked_word}")
#     # print(f"Gussed letters: {set(guessed_letters)}")
#     # print(f"Tries remainig: {tries_remains} / {6}")
#     # print(f"-------------------------------------------")

#     # batch = process_word_fn(masked_word, transform)
#     # # # print(batch)

#     # # # print(model)

#     # # # print(model.device)

#     # model.eval()
#     # with torch.no_grad():
#     #     # # _, letter_likelihoods = model(batch)
#     #     logits, probabilities, letter_likelihoods = model(batch)

#     # letter_likelihoods = probabilities.squeeze().cpu()
    
#     # # # # Convert the PyTorch tensor to a NumPy array
#     # # letter_likelihoods = letter_likelihoods.numpy()
#     # # # Convert the PyTorch tensor to a NumPy array
#     # letter_likelihoods = letter_likelihoods.numpy()
#     # # # print(letter_likelihoods.shape)
#     # print(model.base_rate)

#     position_frequencies, position_counts = analyze_corpus(solver.corpus)

#     letter_likelihoods = calculate_aggregated_letter_probabilities(masked_word, \
#                                             position_frequencies, position_counts)

#     # letter_likelihoods = np.full(26, 0.5) #np.random.randint(0, 2, size=26) #np.full(26, 0) # TODO: remove latter

#     best_char_index = solver.next_guess(masked_word, \
#                             letter_likelihoods, guessed_letters, tries_remains)
#     # print("Determined best character index from solver:", best_char_index)
#     best_guess = string.ascii_lowercase[best_char_index]

#     return best_guess



# def get_likelihood_(aggregated_data, masked_word):
#     # Ensure 'word len' and 'Binary' columns are present in the DataFrame
#     if 'word len' not in aggregated_data.columns or 'Binary' \
#                         not in aggregated_data.columns:
#         raise ValueError("DataFrame must include 'word len' and 'Binary' columns.")

#     # Find the row where the 'word len' matches the length of the masked word
#     match = aggregated_data[aggregated_data['word len'] == len(masked_word)]

#     # If there's a matching row, return the 'Binary' column as a NumPy array; otherwise, return a default binary vector
#     if not match.empty:
#         return np.array(match['Binary'].iloc[0])  # Convert list to NumPy array for consistent data handling
#     else:
#         return np.zeros(26, dtype=int)  # Return a binary vector with all zeros if no match found


    # # Encode the masked_word and process features for model input
    # encoded_word = torch.tensor([encode_character(char) \
    #                         for char in masked_word]).unsqueeze(0).long()
    
    # word_length = torch.tensor([len(masked_word)])

    # # # # # Debug: Print encoded word and its shape
    # # print("Encoded word:", encoded_word.squeeze(0), "Shape:", encoded_word.shape)
    # # # # print(solver.corpus)
    # # # Transform features
    
    # processed_features = transform(encoded_word.squeeze(0)).unsqueeze(0)

    # # # # # # Debug: Print processed features and their shape
    # # print("Processed features shape:", processed_features.shape)
    # # # print(processed_features)

    # # print(model.device) 
    # # Move the tensors to the specified device
    # encoded_word, word_length, processed_features \
    #     = encoded_word.to(device), word_length.to(device), \
    #         processed_features.to(device)

    # # print(model)
    # # print(encoded_word)    
    # model.to(device)
    # model.eval()

    # with torch.no_grad():
    #     # print(f"making prediction from model...")
    #     _, probabilities = model(encoded_word, word_length, \
    #                                        processed_features)

    # print(f"letter likelihood from guess_character: ", letter_likelihoods)
        
    # # Debug: Print logits, their shape, and initial letter likelihoods and their shape
    # print("Logits:", logits, "Logits Shape:", logits.shape)
    # print("Initial letter likelihoods:", letter_likelihoods, "Letter Likelihoods Shape:", letter_likelihoods.shape)

    # # Check if all elements in the letter_likelihoods are zero
    # if torch.all(letter_likelihoods == 0):
    #     print("Warning: Likelihood vector is all zeros, indicating no predictions were made.")
    # print(f"Likehood vector all 0...")


    # letter_likelihoods = torch.zeros(26) 
    # # # letter_likelihoods = torch.ones(1) 
    # letter_likelihoods = letter_likelihoods.to(device)

    # # if all(letter_likelihoods == 0):
    # #     print(f"Likehood is 0")

    # # Prepare likelihoods and the guessed letter mask
    # letter_likelihoods = letter_likelihoods.squeeze(0)
    # guessed_letter_set = set(guessed_letters)
    # guessed_mask = torch.tensor(
    #     [0 if char in guessed_letter_set else 1 for \
    #                 char in string.ascii_lowercase],
    #                 dtype=torch.float32,
    #                 device=device
    # )

    # # # Initialize letter likelihoods as a zero array for 26 letters (a to z)
    # letter_likelihoods = np.zeros(26)
    # # print(letter_likelihoods.shape)
    # guessed_letter_set = set(guessed_letters)

    # guessed_mask = np.array([0 if char in guessed_letter_set \
    #                          else 1 for char in string.ascii_lowercase], dtype=float)

# def get_vowel_likelihoods():
#     vowels = 'aeiou'
#     # Initialize the array with zeros for all alphabets
#     likelihoods = np.zeros(26)
#     # Set the index corresponding to vowels to 1
#     for char in vowels:
#         index = ord(char) - ord('a')
#         likelihoods[index] = 1
#     return likelihoods

# ## This will take care the NN input output
# def guess_character(masked_word, model, \
#                     guessed_letters, \
#                     tries_remains,
#                     solver, \
#                     transform, \
#                     aggregated_data, \
#                     device='cuda'):
    
#     # print(f"-------------------------------------------")
#     # print(f"Data available to make next guess...")
#     # print(f"current state: {masked_word}")
#     # print(f"Gussed letters: {set(guessed_letters)}")
#     # print(f"Tries remainig: {tries_remains} / {6}")
#     # print(f"-------------------------------------------")
    

#     # # # Initialize letter likelihoods as a zero array for 26 letters (a to z)
#     # letter_likelihoods = np.zeros(26)
#     letter_likelihoods = get_likelihood_(aggregated_data, masked_word)

#     guessed_letter_set = set(guessed_letters)

#     # if all(char == '_' for char in masked_word) and tries_remains < 6:
#     #     wrong_gusses = set(guessed_letters) - set(masked_word)
#     #     print(wrong_gusses)
#     #     # pass

#     # Check if all characters in masked_word are '_' and act based on tries_remaining
#     # if all(char == '_' for char in masked_word):
#     #     if tries_remains == 6:
#     #         # Assume get_likelihood_ is a function that fetches likelihoods based on masked_word
#     #         letter_likelihoods = get_likelihood_(aggregated_data, masked_word)
#     #     elif tries_remains < 6:
#     #         letter_likelihoods = get_vowel_likelihoods()
#     # else:
#     #     letter_likelihoods = get_likelihood_(aggregated_data=aggregated_data, masked_word=masked_word)
    
#     # letter_likelihoods = get_likelihood_(aggregated_data=aggregated_data, masked_word=masked_word)
    
#     # letter_likelihoods = np.zeros(26)  # This could be some default or another condition-based setting

#     # print(letter_likelihoods.shape)
#     # guessed_letter_set = set(guessed_letters)

#     guessed_mask = np.array([0 if char in guessed_letter_set \
#                              else 1 for char in string.ascii_lowercase], dtype=float)


#     # # # Debug: Print guessed mask and its shape
#     # print("Guessed mask:", guessed_mask, "Guessed Mask Shape:", guessed_mask.shape)
#     # print("letter likehood shape: ", letter_likelihoods.shape)
    
#     # Initialize masked_likelihood by applying the guessed mask
#     masked_likelihood = letter_likelihoods * guessed_mask

#     best_char_index = solver.next_guess(masked_word, \
#                             masked_likelihood, guessed_letters, tries_remains)
#     # print("Determined best character index from solver:", best_char_index)
#     best_guess = string.ascii_lowercase[best_char_index]

#     return best_guess

# ## This will take care the NN input output
# def guess_character(masked_word, model, \
#                     guessed_letters, \
#                     tries_remains,
#                     solver, \
#                     transform, \
#                     aggregated_data, \
#                     device='cuda'):
    
#     # print(f"-------------------------------------------")
#     # print(f"Data available to make next guess...")
#     # print(f"current state: {masked_word}")
#     # print(f"Gussed letters: {set(guessed_letters)}")
#     # print(f"Tries remainig: {tries_remains} / {6}")
#     # print(f"-------------------------------------------")
    
#     # # Encode the masked_word and process features for model input
#     # encoded_word = torch.tensor([encode_character(char) \
#     #                         for char in masked_word]).unsqueeze(0).long()
    
#     # word_length = torch.tensor([len(masked_word)])

#     # # # # # Debug: Print encoded word and its shape
#     # # print("Encoded word:", encoded_word.squeeze(0), "Shape:", encoded_word.shape)
#     # # # # print(solver.corpus)
#     # # # Transform features
#     # # transform = ProcessWordTransform(solver.corpus)
#     # processed_features = transform(encoded_word.squeeze(0)).unsqueeze(0)

#     # # # # # # Debug: Print processed features and their shape
#     # # print("Processed features shape:", processed_features.shape)
#     # # # print(processed_features)

#     # # print(model.device) 
#     # # Move the tensors to the specified device
#     # encoded_word, word_length, processed_features \
#     #     = encoded_word.to(device), word_length.to(device), \
#     #         processed_features.to(device)

#     # # print(model)
#     # # print(encoded_word)    
#     # model.to(device)
#     # model.eval()

#     # with torch.no_grad():
#     #     # print(f"making prediction from model...")
#     #     _, probabilities = model(encoded_word, word_length, \
#     #                                        processed_features)

#     # print(f"letter likelihood from guess_character: ", letter_likelihoods)
        
#     # # Debug: Print logits, their shape, and initial letter likelihoods and their shape
#     # print("Logits:", logits, "Logits Shape:", logits.shape)
#     # print("Initial letter likelihoods:", letter_likelihoods, "Letter Likelihoods Shape:", letter_likelihoods.shape)

#     # # Check if all elements in the letter_likelihoods are zero
#     # if torch.all(letter_likelihoods == 0):
#     #     print("Warning: Likelihood vector is all zeros, indicating no predictions were made.")
#     # print(f"Likehood vector all 0...")

#     #  #  # Forcing all elements to 1/0 for illustration
#     # letter_likelihoods = torch.zeros(26) #  #  # Forcing all elements to 1/0 for illustration
    
#     # ==========Simulation============
#     # print(masked_word)
#     # print(aggregated_data)
#     # guessed_chars = get_guessed_chars_for_masked_word(aggregated_data, masked_word)
#     # guessed_chars = 'a,b,c,d,e,f,g,h,i,k,l,m,n,o,p,r,s,t,u,v,w,y,z' # get_first_guessed_chars_for_masked_word_length(len(masked_word))
#     # # print(guessed_chars)
#     # if guessed_chars == None:
#     #     letter_likelihoods = np.zeros(26)
#     # else:
#     #     # print(guessed_chars)
#     #     guessed_chars = guessed_chars.split(',')
#     #     # Initialize MultiLabelBinarizer
#     #     mlb = MultiLabelBinarizer(classes=[chr(i) for i in range(ord('a'), ord('z')+1)])
#     #     binarized_output = mlb.fit_transform([guessed_chars])
#     #     letter_likelihoods = binarized_output.squeeze()
#     # # ==========Simulation============

#     # letter_likelihoods = torch.zeros(26) 
#     # # # letter_likelihoods = torch.ones(1) 
#     # letter_likelihoods = letter_likelihoods.to(device)

#     # # if all(letter_likelihoods == 0):
#     # #     print(f"Likehood is 0")

#     # # Prepare likelihoods and the guessed letter mask
#     # letter_likelihoods = letter_likelihoods.squeeze(0)
#     # guessed_letter_set = set(guessed_letters)
#     # guessed_mask = torch.tensor(
#     #     [0 if char in guessed_letter_set else 1 for \
#     #                 char in string.ascii_lowercase],
#     #                 dtype=torch.float32,
#     #                 device=device
#     # )

#     # # # Initialize letter likelihoods as a zero array for 26 letters (a to z)
#     letter_likelihoods = np.zeros(26)
#     # print(letter_likelihoods.shape)
#     guessed_letter_set = set(guessed_letters)

#     guessed_mask = np.array([0 if char in guessed_letter_set \
#                              else 1 for char in string.ascii_lowercase], dtype=float)


#     # # # Debug: Print guessed mask and its shape
#     # print("Guessed mask:", guessed_mask, "Guessed Mask Shape:", guessed_mask.shape)
#     # print("letter likehood shape: ", letter_likelihoods.shape)
    
#     # Initialize masked_likelihood by applying the guessed mask
#     masked_likelihood = letter_likelihoods * guessed_mask

#     best_char_index = solver.next_guess(masked_word, \
#                             masked_likelihood, guessed_letters, tries_remains)
#     # print("Determined best character index from solver:", best_char_index)
#     best_guess = string.ascii_lowercase[best_char_index]

#     return best_guess
