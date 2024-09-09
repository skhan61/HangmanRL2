import random
from collections import defaultdict

def classify_words_by_unique_letter_count(word_list):
    """
    Classify words by the number of unique letters.

    Args:
        word_list (list): A list of words to classify.

    Returns:
        dict: A dictionary where keys represent the number of unique letters,
              and values are lists of words falling into that category.
    """
    classification = defaultdict(list)
    for word in word_list:
        unique_count = len(set(word.lower()))
        classification[unique_count].append(word)
    return classification

def stratified_sample_from_categories(classification, num_samples):
    """
    Take a stratified sample from each category.

    Args:
        classification (dict): A dictionary of classified words, 
        where keys represent the unique letter counts.
        num_samples (int): The desired number of samples per category.

    Returns:
        dict: A dictionary where keys represent the unique letter counts,
              and values are the sampled lists of words from each category.
    """
    stratified_sample = defaultdict(list)
    for unique_count, words in classification.items():
        if len(words) > num_samples:
            stratified_sample[unique_count] = random.sample(words, num_samples)
        else:
            stratified_sample[unique_count] = words
    return stratified_sample

def summarize_categories(classification):
    """
    Summarize the number of categories and samples in each category.

    Args:
        classification (dict): A dictionary of classified words.

    Returns:
        list: A list of strings summarizing each category.
    """
    summaries = []
    for unique_count, words in sorted(classification.items()):
        summaries.append(f"Category {unique_count}: {len(words)} words")
    return summaries

def stratified_sample_from_categories_with_minimum_one(classification, proportion):
    """
    Sample words for validation from each category based on a given proportion,
    ensuring at least one sample from each category if available.
    
    Args:
        classification (dict): Dictionary with unique letter counts as keys and lists of words as values.
        proportion (float): Proportion of total words in each category to use for validation.
    
    Returns:
        dict: A dictionary containing sampled words for validation from each category,
              with at least one sample from each category if available.
    """
    validation_samples = {}
    for unique_count, words in classification.items():
        if words:  # Check if there are words in the category
            sample_size = max(1, int(len(words) * proportion))  # Ensure at least one sample if words are present
            sampled_words = random.sample(words, min(len(words), sample_size))  # Sample words
            validation_samples[unique_count] = sampled_words
    return validation_samples


def stratified_sample_list_from_categories(classification, proportion):
    """
    Sample words for validation from each category based on a given proportion,
    ensuring at least one sample from each category if available, and return a list of all samples.
    
    Args:
        classification (dict): Dictionary with unique letter counts as keys and lists of words as values.
        proportion (float): Proportion of total words in each category to use for validation.
    
    Returns:
        list: A list containing all sampled words from each category.
    """
    validation_samples = []
    for unique_count, words in classification.items():
        if words:  # Check if there are words in the category
            sample_size = max(1, int(len(words) * proportion))  # Ensure at least one sample if words are present
            sampled_words = random.sample(words, min(len(words), sample_size))  # Sample words
            validation_samples.extend(sampled_words)  # Add sampled words to the main list
    return validation_samples


def categorize_by_word_length_and_unique_chars(word_list):
    """
    Classify words by their length and the number of unique letters.
    
    Args:
        word_list (list): A list of words to classify.
    
    Returns:
        dict: A dictionary where keys are word lengths and values are dictionaries. 
              Each sub-dictionary maps unique letter counts to lists of words.
    """
    length_classification = defaultdict(lambda: defaultdict(list))
    for word in word_list:
        word_length = len(word)
        unique_chars = len(set(word.lower()))
        length_classification[word_length][unique_chars].append(word)
    return length_classification

from collections import defaultdict, Counter

def most_occurred_letters(corpus):
    """
    Find the most occurred letters for each word length and unique character count.
    
    Args:
        corpus (list): A list of words in the corpus.
    
    Returns:
        dict: A dictionary where keys are word lengths and values are dictionaries. 
              Each sub-dictionary maps unique letter counts to the most occurred letters.
    """
    length_classification = categorize_by_word_length_and_unique_chars(corpus)
    most_occurred = defaultdict(lambda: defaultdict(str))
    
    for word_length, unique_classification in length_classification.items():
        for unique_count, words in unique_classification.items():
            letter_counter = Counter("".join(words).lower())
            most_common_letter = letter_counter.most_common(1)[0][0] if letter_counter else None
            most_occurred[word_length][unique_count] = most_common_letter
    
    return most_occurred


def categorize_words_by_length(words):
    """
    Categorizes words based on their length.

    Args:
        words (list): A list of words to be categorized.

    Returns:
        dict: A dictionary where keys are word lengths and values are lists of words of that length.
    """
    categorized_dict = {}
    for word in words:
        length = len(word)
        if length not in categorized_dict:
            categorized_dict[length] = []
        categorized_dict[length].append(word)
    
    return categorized_dict
