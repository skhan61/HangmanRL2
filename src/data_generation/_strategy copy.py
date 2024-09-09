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

# The HangmanMachine
class HangmanFreqSolver:
    def __init__(self, corpus):
        self.corpus = corpus
        self.global_freq = self.build_global_frequency(corpus)
        self.positional_freq = self.build_positional_frequency(corpus)

        self.regex_cache = {}


    def build_global_frequency(self, corpus, exclude_letters=set()):
        frequency = collections.defaultdict(int)
        total_words = len(corpus)  # Total number of words, not total letters

        for word in corpus:
            seen_chars = set(word.lower()) - exclude_letters
            for char in seen_chars:
                frequency[char] += 1

        # Convert counts to frequencies based on the number of words
        for letter in frequency:
            frequency[letter] /= total_words  # Normalize by the total number of words

        return dict(frequency)

    def build_positional_frequency(self, corpus, exclude_letters=set()):
        all_letters = set(string.ascii_lowercase) - set(exclude_letters)
        positional_frequency = collections.defaultdict(lambda: collections.defaultdict(float))
        word_count_by_length = collections.defaultdict(int)
        letter_seen_at_position = collections.defaultdict(lambda: collections.defaultdict(set))

        for word in corpus:
            length = len(word)
            word_count_by_length[length] += 1
            for index, char in enumerate(word.lower()):
                if char not in exclude_letters:
                    letter_seen_at_position[length][index].add(char)

        # Initialize counts for each position and letter to 0
        for length in word_count_by_length:
            for index in range(length):
                for letter in all_letters:
                    positional_frequency[(length, (index, letter))] = 0.0

        # Update counts based on the unique sets of letters at each position
        for length, pos_dict in letter_seen_at_position.items():
            for index, letters in pos_dict.items():
                for letter in letters:
                    positional_frequency[(length, (index, letter))] += 1

        # Normalize by the number of words of each length to convert counts to frequencies
        for (length, (index, letter)), count in positional_frequency.items():
            if word_count_by_length[length] > 0:
                positional_frequency[(length, (index, letter))] = count \
                                                    / word_count_by_length[length]

        return dict(positional_frequency)

    def analyze_masked_word(self, masked_word, letters_to_use):
        # Construct regex pattern from the masked word and letters to use
        pattern_parts = [f"[{''.join(letters_to_use)}]" if ch == '_' else ch for ch in masked_word]
        pattern = '^' + ''.join(pattern_parts) + '$'
        # print(pattern)
        regex = re.compile(pattern)

        # Use regex to filter corpus
        new_dictionary = [word for word in self.corpus if regex.fullmatch(word)]
        # print(len(new_dictionary))

        # Build frequency analysis based on filtered dictionary
        if new_dictionary:
            refined_global_freq = self.build_global_frequency(new_dictionary, set(letters_to_use))
            refined_positional_freq = self.build_positional_frequency(new_dictionary, set(letters_to_use))
            letter_counts = collections.Counter({char: refined_global_freq.get(char, 0)
                                                for char in letters_to_use})
            for i, char in enumerate(masked_word):
                if char == '_':
                    for letter in letters_to_use:
                        pos_key = (len(masked_word), i, letter)
                        letter_counts[letter] += refined_positional_freq.get((len(masked_word), i), {}).get(letter, 0)
        else:
            # Fallback to previous frequencies if no matches are found
            letter_counts = collections.Counter({char: self.global_freq.get(char, 0)
                                                for char in letters_to_use})
            for i, char in enumerate(masked_word):
                if char == '_':
                    for letter in letters_to_use:
                        pos_key = (len(masked_word), i, letter)
                        letter_counts[letter] += self.positional_freq.get((len(masked_word), i), {}).get(letter, 0)

        # # print(f"Letter counts: {letter_counts}")
        # print()
        return letter_counts

    # def guess_masking(self, vector, guessed_letters):
    #     """Masks the likelihood vector by setting positions of guessed letters to zero."""
    #     guessed_letter_set = set(guessed_letters)
    #     guessed_mask = np.array([0 if char in guessed_letter_set 
    #                             else 1 for char in string.ascii_lowercase], dtype=float)
    #     masked_vector = vector * guessed_mask
    #     return masked_vector

    def guess_masking(self, vector, guessed_letters):
        """Sets positions of guessed letters to a large negative value to effectively remove their influence."""
        guessed_letter_set = set(guessed_letters)
        # Define a large negative value for masking
        mask_value = -np.inf  # This effectively removes any guessed letter from consideration
        
        # Create a masked vector where guessed letters' positions are set to the mask_value
        masked_vector = np.array([mask_value if char in guessed_letter_set else vector[idx]
                                for idx, char in enumerate(string.ascii_lowercase)])
        return masked_vector

        
    # v5: pattern search in corpus: TODO: 
    # likelihood_vector -> already gussed letter masked
    # Generating action
    def generate_next_guess(self, masked_word, likelihood_vector, \
                   guessed_letters, tries_remains=None):
        
        # print(guessed_letters)

        masked_likelihood_vector \
            = self.guess_masking(likelihood_vector, guessed_letters)

        # print('\n')
        letters_from_likelihood = {chr(i + ord('a')) \
                    for i, val in enumerate(masked_likelihood_vector) if val == 1}
        letters_to_use = letters_from_likelihood - set(guessed_letters)

        # print(len(letters_to_use))
        
        # If the likelihood vector suggests no letters 
        # and the masked word has unknowns
        if not letters_from_likelihood and '_' in masked_word:
            letters_to_use = set(string.ascii_lowercase) - set(guessed_letters)
            # print("No likelihood vector suggestions; using all unguessed letters.")

        corpa_based_letter_probs = self.analyze_masked_word(masked_word, letters_to_use)
        # print(corpa_based_letter_probs)
        # Select the most frequent unguessed letter
        corpa_based_guess_letter = max((letter for letter in corpa_based_letter_probs \
                                    if letter not in guessed_letters),
                                    key=corpa_based_letter_probs.get, default='!') # max freq count from corpus

        # Return the index of the guessed letter in the alphabet
        return string.ascii_lowercase.index(corpa_based_guess_letter) \
            if corpa_based_guess_letter != '!' else -1
    
    # # motoring action
    def next_guess(self, masked_word, ranking_vector, \
                                    guessed_letters, tries_remains):
        if '_' in masked_word:
            letters_to_use = set(string.ascii_lowercase) - set(guessed_letters)

        # print(ranking_vector)

        # Analyze the masked word to determine corpus-based letter probabilities
        corpa_based_letter_probs = self.analyze_masked_word(masked_word, letters_to_use)
        # print(corpa_based_letter_probs)  # Print corpus-based probabilities

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
            raise ValueError("No valid letters found after processing. This could be due to extreme masking or data issues.")

        # Break ties using corpus-based letter probabilities if there are multiple max values
        if len(max_letters) > 1:
            # print(max_letters)
            print(f'tie happens...')
            # Use corpus probabilities to select the letter
            # Only consider letters that are among the tied candidates
            max_letter = max(max_letters, \
                        key=lambda letter: corpa_based_letter_probs.get(letter, 0))
        else:
            max_letter = max_letters[0]  # There is only one maximum
        # print(max_letters)
        guess_letter = max_letter
        # print(guess_letter)
        # print('\n')
        # Return the index of the guessed letter in the alphabet
        return string.ascii_lowercase.index(guess_letter) if guess_letter != '!' else -1



    # # Motoring action: guessing
    # def next_guess(self, masked_word, probability_vector, guessed_letters, tries_remains):
    #     # Determine usable letters excluding those already guessed
    #     if '_' in masked_word:
    #         letters_to_use = set(string.ascii_lowercase) - set(guessed_letters)

    #     # Analyze the masked word to determine corpus-based letter probabilities
    #     corpa_based_letter_probs = self.analyze_masked_word(masked_word, letters_to_use)
    #     # print(corpa_based_letter_probs)  # Print corpus-based probabilities

    #     # Apply masking to the neural network-based probability vector
    #     masked_probability_vector = self.guess_masking(probability_vector, guessed_letters)
    #     # print(masked_probability_vector)  # Print masked probability vector

    #     # Find the index of the highest probability and resolve ties using corpus-based probabilities
    #     max_prob_index = np.argmax(masked_probability_vector)
    #     nn_based_guess_letter = string.ascii_lowercase[max_prob_index]

    #     # If there's a tie in the highest probability, use the order in corpa_based_letter_probs to break the tie
    #     if np.sum(masked_probability_vector == masked_probability_vector[max_prob_index]) > 1:
    #         # Find all letters with the highest probability
    #         tied_letters = [string.ascii_lowercase[i] for i, prob in enumerate(masked_probability_vector)
    #                         if prob == masked_probability_vector[max_prob_index]]
    #         # Break the tie using the order in corpa_based_letter_probs
    #         nn_based_guess_letter = min(tied_letters, key=lambda x: list(corpa_based_letter_probs.keys()).index(x))

    #     # Choose the guess letter based on the analysis
    #     guess_letter = nn_based_guess_letter

    #     # Return the index of the guessed letter in the alphabet
    #     return string.ascii_lowercase.index(guess_letter) if guess_letter != '!' else -1

    # def next_guess(self, masked_word, probability_vector, \
    #                guessed_letters, tries_remains):
        
    #     if '_' in masked_word:
    #         letters_to_use = set(string.ascii_lowercase) - set(guessed_letters)
    #     # print("No likelihood vector suggestions; using all unguessed letters.")

    #     # print(len(letters_to_use))

    #     corpa_based_letter_probs = self.analyze_masked_word(masked_word, letters_to_use)
    #     print(corpa_based_letter_probs)

    #     masked_probability_vector = self.guess_masking(probability_vector, guessed_letters)
    #     # print(masked_probability_vector)
    #     max_prob_index = np.argmax(masked_probability_vector)  -> if ties happen then use the corpa_based_letter_probs tkae the seq of the kesys and making that as referse break ties. the lettrs what in comes in the dict first guess first# Get the index of the highest probability
    #     nn_based_guess_letter = string.ascii_lowercase[max_prob_index] 

    #     guess_letter = nn_based_guess_letter
    #     # Return the index of the guessed letter in the alphabet
    #     return string.ascii_lowercase.index(guess_letter) if guess_letter != '!' else -1


    # # # Select the most frequent unguessed letter
    # # corpa_based_guess_letter = max((letter for letter in corpa_based_letter_probs \
    # #                                 if letter not in guessed_letters),
    # #                 key=corpa_based_letter_probs.get, default='!') # max freq count from corpus

    # # guess_letter = corpa_based_guess_letter