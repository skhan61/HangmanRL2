from lightning import LightningDataModule
from typing import List
from supervised_learning.data.dataset import HangmanDataset
from supervised_learning.data.data_utils import encode_word, custom_collate_fn
from src.utils import read_word_list
import os
import pickle
from torch.utils.data import DataLoader, random_split
from supervised_learning.data.data_loader import HangmanDataLoader
from multiprocessing import Pool
import torch
import numpy as np

class HangmanDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self._config = config
        # print(self._config['batch_size'])

    def prepare_data(self):
        train_dir = os.path.join(self._config['dataset_dir'], 'train')
        val_dir = os.path.join(self._config['dataset_dir'], 'val')

        # Ensure the pickle directories exist, create if they do not
        for directory in [train_dir, val_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory {directory} because it did not exist.")

        # Function to process and save data
        def process_data(data_list, directory, num_samples):
            if not os.listdir(directory):  # List is empty if no files
                print(f"No pickle files found in {directory}. Preparing data...")
                final_encoded_data = []

                # Example: Splitting word list into chunks of 10,000 words each for multiple .pkl files
                chunk_size = 10_000
                num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Calculates the number of chunks needed

                for i in range(num_chunks):
                    start_index = i * chunk_size
                    end_index = min(start_index + chunk_size, num_samples)
                    chunk = data_list[start_index:end_index]

                    for word in chunk:
                        encoded_result, positions_list, unique_chars = encode_word(word, self._config['vocab_size'], \
                                                        self._config['min_length'], self._config['char_to_index'])
                        if encoded_result is not None:
                            final_encoded_data.append((encoded_result, positions_list, unique_chars))

                    # Save this chunk to a separate pickle file
                    pkl_path = os.path.join(directory, f'encoded_words_{i}.pkl')
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(final_encoded_data, f)
                    print(f"Data encoded and saved to {pkl_path}")
                    final_encoded_data = []  # Clear list for next chunk
            else:
                print(f"Pickle files are already present in {directory}. No need to prepare data.")

        # Read the full corpus and split into train and validation lists
        corpus = read_word_list(self._config['data_dir'], num_samples=250_000)
        val_list = read_word_list(self._config['data_dir'], num_samples=1_000)
        train_list = list(set(corpus) - set(val_list))

        # Process and save train and validation datasets
        process_data(train_list, train_dir, len(train_list))
        process_data(val_list, val_dir, len(val_list))

    def create_dataset(self, mode):
        """Create a dataset with a mode-specific configuration."""
        # Clone the entire configuration to ensure isolation
        config_clone = self._config.copy()
        # Update the 'pkl_dir' specifically for the mode ('train' or 'val')
        config_clone['pkl_dir'] = os.path.join(config_clone['dataset_dir'], mode)
        # Return a new dataset instance with the updated config
        return HangmanDataset(config_clone)

    # Example usage within the class
    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            self.train_dataset = self.create_dataset('train')
            # print(len(self.train_dataset))
        if stage == 'validate' or stage is None:
            self.val_dataset = self.create_dataset('val')

    def train_dataloader(self):
        # print(len(self.train_dataset))
        # print(self._config['batch_size'])
        return DataLoader(self.train_dataset, batch_size=self._config['batch_size'], 
                        collate_fn=custom_collate_fn, shuffle=True, \
                        num_workers=self._config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._config['batch_size'], 
                        collate_fn=custom_collate_fn, shuffle=False, \
                        num_workers=self._config['num_workers'])
    
 