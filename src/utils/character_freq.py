from collections import defaultdict, Counter
import string
import torch

# def calculate_frequencies(corpus):
#     frequencies = defaultdict(Counter)
#     total_word_count_by_length = defaultdict(int)
    
#     # Initialize frequencies with a small count for each character to ensure all are present
#     for char in string.ascii_lowercase:
#         for word in corpus:
#             frequencies[len(word)][char] += 1  # Start with a count of 1 for each character

#     # Iterate through each word in the corpus to count actual occurrences
#     for word in corpus:
#         length = len(word)
#         frequencies[length].update(word)
#         total_word_count_by_length[length] += 1

#     # Convert counts to probabilities
#     probabilities = {}
#     for length, freq in frequencies.items():
#         total_chars = sum(freq.values())
#         probabilities[length] = {char: count / total_chars for char, count in freq.items()}
    
#     return probabilities, total_word_count_by_length


def calculate_frequencies(corpus):
    from collections import defaultdict, Counter
    
    # Initialize frequencies with a small count for each character to ensure all are present
    frequencies = defaultdict(Counter)
    total_word_count_by_length = defaultdict(int)
    
    # Iterate through each word in the corpus
    for word in corpus:
        length = len(word)
        seen_chars = set(word.lower())  # Track unique characters only
        # Increment the count for each unique character in the word
        for char in seen_chars:
            frequencies[length][char] += 1
        # Track the number of words of each length
        total_word_count_by_length[length] += 1

    # Convert counts to probabilities
    probabilities = {}
    for length, freq in frequencies.items():
        total_chars = sum(freq.values())  # Total unique letter occurrences at this length
        probabilities[length] = {char: count / total_chars for char, count in freq.items()}
    
    return probabilities, total_word_count_by_length

# # Example usage:
# corpus = ["apple", "apply", "align", "crisp", "bloom", "blast", "about", "crust", "grape", "glyph"]
# probabilities, total_word_count_by_length = calculate_frequencies(corpus)

# print("Probabilities by word length:")
# for length, probs in probabilities.items():
#     print(f"Length {length}: {probs}")

# print("Total word count by length:", total_word_count_by_length)


# def get_character_probability(corpus, word_length):
#     probabilities, total_word_count_by_length = calculate_frequencies(corpus)

#     # Check if word_length is a tensor and extract its value
#     if isinstance(word_length, torch.Tensor):
#         word_length = word_length.item()  # Convert tensor to integer

#     if word_length in probabilities:
#         return probabilities[word_length]
#     else:
#         # Return a uniform distribution if no words of that length exist
#         return {char: 1/26 for char in string.ascii_lowercase}
# Using caching to avoid recalculating frequencies for unchanged corpus

import functools
import string
 
@functools.lru_cache(maxsize=1)
def get_cached_frequencies(corpus):
    return calculate_frequencies(corpus)

def get_character_probability(corpus, word_length):
    probabilities, total_word_count_by_length = get_cached_frequencies(tuple(corpus))

    if word_length in probabilities:
        return probabilities[word_length]
    else:
        return {char: 1/26 for char in string.ascii_lowercase}


def get_letter_priority(length_letter_count, length):
    if length in length_letter_count:
        # Sort letters by their frequency count in descending order
        return sorted(length_letter_count[length].items(), key=lambda item: item[1], reverse=True)
    return []


# Function to build a frequency table of letters by word length
def build_frequency_table(words):
    length_letter_count = defaultdict(Counter)
    for word in words:
        word_length = len(word)
        seen_letters = set()  # To avoid double-counting letters in the same word
        for letter in word:
            if letter not in seen_letters:
                length_letter_count[word_length][letter] += 1
                seen_letters.add(letter)
    return length_letter_count
