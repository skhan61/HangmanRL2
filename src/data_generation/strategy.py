import collections
import re
import string
import math
import collections
import re
import string
import math
import numpy as np
from collections import Counter
import random
from collections import Counter, defaultdict

# The HangmanMachine
class HangmanFreqSolver:
    def __init__(self, corpus):
        self.corpus = corpus
        self.length_specific_letter_frequencies = defaultdict(Counter)
        self.analyze_corpus()

    def analyze_corpus(self):
        """ Analyze the corpus to calculate the frequency of each letter by word length, counted once per word. """
        for word in self.corpus:
            seen_letters = set()
            word_length = len(word)
            for letter in word:
                if letter not in seen_letters and letter in string.ascii_lowercase:
                    self.length_specific_letter_frequencies[word_length][letter] += 1
                    seen_letters.add(letter)

    def generate_letter_frequency_vector(self, word_length):
        """ Generate a frequency vector for the letters 'a' to 'z' for a specific word length. """
        frequency_vector = np.zeros(26)  # 26 letters in the English alphabet
        total_letters = sum(self.length_specific_letter_frequencies[word_length].values())
        if total_letters == 0:  # No words of this length in the corpus
            return frequency_vector
        for letter, count in self.length_specific_letter_frequencies[word_length].items():
            index = ord(letter) - ord('a')
            frequency_vector[index] = count / total_letters  # Normalize to create a probability distribution
        return frequency_vector
    
    # v6: pattern search in corpus: TODO: 
    # likelihood_vector -> already gussed letter masked
    # Generating action
    def generate_next_guess(self, masked_word, likelihood_vector, guessed_letters, strategy):
        masked_likelihood_vector = self.guess_masking(likelihood_vector, guessed_letters)
        relevant_vector = self.generate_letter_frequency_vector(len(masked_word))
        # print(relevant_vector)
        masked_relevant_vector = relevant_vector * masked_likelihood_vector
        # print(masked_relevant_vector)

        if strategy == 'min':
            filtered_indices = [i for i, x in enumerate(masked_relevant_vector) if x > 0]
            guess_letter = chr(filtered_indices[np.argmin(masked_relevant_vector[filtered_indices])] \
                               + ord('a')) if filtered_indices else '!'
        elif strategy == 'max':
            max_index = np.argmax(masked_relevant_vector)
            guess_letter = chr(max_index + ord('a')) if masked_relevant_vector[max_index] > 0 else '!'
        
        elif strategy == 'random':
            non_zero_indices = [i for i, val in enumerate(masked_relevant_vector) if val > 0]
            guess_letter = chr(random.choice(non_zero_indices) + ord('a')) if non_zero_indices else '!'
        
        elif strategy == 'random_42':
            random.seed(42)  # Setting seed for reproducible randomness
            non_zero_indices = [i for i, val in enumerate(masked_relevant_vector) if val > 0]
            guess_letter = chr(random.choice(non_zero_indices) + ord('a')) if non_zero_indices else '!'
        
        elif strategy == 'random_420':
            random.seed(420)  # Setting seed for reproducible randomness
            non_zero_indices = [i for i, val in enumerate(masked_relevant_vector) if val > 0]
            guess_letter = chr(random.choice(non_zero_indices) + ord('a')) if non_zero_indices else '!'
        
        # print()
        
        return string.ascii_lowercase.index(guess_letter) if guess_letter != '!' else -1
    
    def guess_masking(self, vector, guessed_letters):
        """Sets positions of guessed letters to a large negative value to effectively remove their influence."""
        guessed_letter_set = set(guessed_letters)
        # Define a large negative value for masking
        mask_value = -np.inf  # This effectively removes any guessed letter from consideration
        
        # Create a masked vector where guessed letters' positions are set to the mask_value
        masked_vector = np.array([mask_value if char in guessed_letter_set else vector[idx]
                                for idx, char in enumerate(string.ascii_lowercase)])
        return masked_vector


    # # motoring action
    def next_guess(self, masked_word, ranking_vector, \
                                    guessed_letters, tries_remains):
        # if '_' in masked_word:
        #     letters_to_use = set(string.ascii_lowercase) - set(guessed_letters)

        # # print(ranking_vector)
        # if masked_word.count('_') == len(masked_word):  # All characters are '_'
        #     # print(f'all masked')
        #     masked_ranking_vector = self.generate_letter_frequency_vector(len(masked_word))
        #     # print(masked_ranking_vector)
        #     masked_ranking_vector = self.guess_masking(masked_ranking_vector, guessed_letters)
        #     # print(masked_ranking_vector)
        # else:
            # Apply masking to the neural network-based probability vector
        masked_ranking_vector = self.guess_masking(ranking_vector, guessed_letters)
            # print(masked_ranking_vector)  # Print masked probability vector

        # Find all indices where the probability is the maximum
        max_score = np.max(masked_ranking_vector)
        # print(max_score)
        max_indices = np.flatnonzero(masked_ranking_vector == max_score)
        # print(max_indices)

        # Convert indices to letters
        max_letters = [string.ascii_lowercase[index] for index in max_indices]
        
        # Check if max_letters is empty
        if not max_letters:
            print(max_letters)
            raise ValueError("No valid letters found after processing. \
                             This could be due to extreme masking or data issues.")

        max_letter = max_letters[0]  # There is only one maximum
        # print(max_letters)
        guess_letter = max_letter
        # # print(guess_letter)
        # print('\n')
        # Return the index of the guessed letter in the alphabet
        return string.ascii_lowercase.index(guess_letter) \
            if guess_letter != '!' else -1
