import collections
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from random import choices

import random

def read_rnd_word_list(file_path, num_samples=1_000):
    with open(file_path, 'r') as file:
        words = file.readlines()  # Read all lines from the file
        words = [word.strip() for word in words]  # Remove any extra whitespace or newline characters

    # Shuffle the list of words to randomize the order
    random.shuffle(words)

    # Return the first num_samples words if the list is longer than num_samples
    return words[:num_samples] if len(words) > num_samples else words


def read_word_list(file_path, num_samples=-1):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            word_dict = defaultdict(list)
            for line in file:
                word = line.strip()
                if word:
                    word_dict[len(word)].append(word)

            if num_samples == -1:
                # Return all words categorized by length
                return {length: words for length, words in word_dict.items()}
            else:
                # Stratified sampling by word length
                sampled_words = []
                total_words = sum(len(words) for words in word_dict.values())
                for length, words in word_dict.items():
                    # Calculate the number of samples for each length
                    length_samples = int(len(words) / total_words * num_samples)
                    sampled_words.extend(
                        random.sample(words, min(len(words), length_samples))
                    )

                return sampled_words
    except FileNotFoundError:
        print("The file was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
