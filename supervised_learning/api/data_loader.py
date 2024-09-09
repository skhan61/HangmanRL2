import torch
from torch.utils.data import DataLoader
import numpy as np

class HangmanDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=False, config=None):
        self.dataset = dataset
        self.config = config

        # Lambda function for the custom collate function
        collate_fn = lambda batch: self.collate_fn(batch)
        print(config['batch_size'])

        # Initialize the DataLoader with the custom collate function
        super(HangmanDataLoader, self).__init__(dataset=self.dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=shuffle,
                                                num_workers=config['num_workers'],
                                                collate_fn=collate_fn)

    def update_dataset(self, epoch):
        # Update the dataset for the new epoch
        self.dataset.cur_epoch(epoch)

    def collate_fn(self, batch):
        print(batch)
        """Handles integer-indexed batch creation with padding using a fixed index."""
        vocab_size = self.config['vocab_size'] + 2  # Adjusting for additional indices such as 'MASK' and padding
        max_len = max(len(x[0]) for x in batch)
        padded_value = vocab_size - 1  # Use last index for padding, assumed as 27 if vocab_size is 26 initially

        # Create padded input batch
        inputs = torch.stack([torch.tensor(np.pad(word[0], (0, max_len \
                                    - len(word[0])), mode='constant', \
                                    constant_values=padded_value)) for word in batch])
        labels = torch.stack([torch.tensor(x[1]) for x in batch])
        miss_chars = torch.stack([torch.tensor(x[2]) for x in batch])
        lens = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)

        return {
            'inputs': inputs,
            'labels': labels,
            'miss_chars': miss_chars,
            'lengths': lens
        }

# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import torch

# class HangmanDataLoader(DataLoader):
#     def __init__(self, dataset: Dataset, shuffle: bool = False, \
#                  config: dict = None):

#         self.dataset = dataset
#         self.config = config

#         print(self.dataset.cur_epoch)

#         collate_fn = lambda batch: self.collate_fn(batch)
        
#         super(HangmanDataLoader, self).__init__(dataset=self.dataset, \
#                                 batch_size=config['batch_size'], \
#                                 shuffle=shuffle, num_workers=config['num_workers'], \
#                                 collate_fn=collate_fn)

#     def update_dataset(self, epoch):
#         self.dataset.cur_epoch(epoch)

#     def collate_fn(self, batch):
#         """
#         Custom collate function to handle padding 
#             and other batch-specific operations.
#         """
#         if self.config['integer_indices']:
#             return self.collate_integer(batch)
#         else:
#             return self.collate_one_hot(batch)

#     @staticmethod
#     def collate_integer(batch):
#         """Collates batches of integer-indexed data, padding to the longest sequence."""
#         max_len = max(len(x[0]) for x in batch)
#         padded_value = batch[0][0].max() + 1  # Assumes integer encoding uses continuous range
#         inputs = np.array([np.pad(word[0], (0, max_len - len(word[0])),
#                                   mode='constant', constant_values=padded_value) for word in batch])
#         labels = np.array([x[1] for x in batch])
#         miss_chars = np.array([x[2] for x in batch])
#         lens = np.array([len(x[0]) for x in batch])
#         # return inputs, labels, miss_chars, lens
#         # Create a dictionary with keys matching the names of the outputs
#         result = {
#             'inputs': inputs,
#             'labels': labels,
#             'miss_chars': miss_chars,
#             'lenghts': lens
#         }

#         return result

#     @staticmethod
#     def collate_one_hot(batch):
#         """Collates batches of one-hot encoded data, padding each sample to the longest in the batch."""
#         max_len = max(x[0].shape[0] for x in batch)
#         vocab_size = batch[0][0].shape[1]  # Assumes all one-hot vectors have the same second dimension
#         inputs = np.array([np.pad(word[0], ((0, max_len - word[0].shape[0]), (0, 0)),
#                                   mode='constant', constant_values=0) for word in batch])
#         labels = np.array([x[1] for x in batch])
#         miss_chars = np.array([x[2] for x in batch])
#         lens = np.array([x[0].shape[0] for x in batch])
#         # return inputs, labels, miss_chars, lens
#         result = {
#             'inputs': inputs,
#             'labels': labels,
#             'miss_chars': miss_chars,
#             'lenghts': lens
#         }

#         return result


# def batchify_words(batch, vocab_size, using_embedding, extra_vocab=1):
#     """
#     Converts a list of words into a batch by padding them to a fixed length array.
#     Handles both one-hot encoded data and index-based embeddings.

#     Args:
#         batch (list): A list of words, where each word is either an array of indices or a one-hot encoded array.
#         vocab_size (int): Size of vocabulary (excluding any special tokens like padding).
#         using_embedding (bool): If True, expects word as a list of indices. If False, expects word as one-hot encoded arrays.
#         extra_vocab (int): Number to accommodate special tokens (like padding), added to vocab size in one-hot encoding.

#     Returns:
#         np.array: Array of words, uniformly padded to the maximum length in the batch.
#     """

#     max_len = max(len(word) for word in batch)  # Find maximum length word
#     final_batch = []

#     if using_embedding:
#         # Handle embeddings: words are lists of indices
#         padded_value = vocab_size  # Typically, vocab_size is used as the padding index
#         for word in batch:
#             padded_word = np.pad(word, (0, max_len - len(word)), \
#                                  mode='constant', constant_values=padded_value)
#             final_batch.append(padded_word)
#     else:
#         # Handle one-hot encoding: words are arrays with shape [length, vocab_size + extra_vocab]
#         for word in batch:
#             padded_word = np.pad(word, ((0, max_len - len(word)), \
#                                 (0, 0)), mode='constant', constant_values=0)
#             final_batch.append(padded_word)

#     return np.array(final_batch)