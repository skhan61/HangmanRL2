from torch.utils.data import Dataset, DataLoader
import os
import os
import pickle
import numpy as np
class HangmanDataset(Dataset):

    def __init__(self, config):
        self.pkl_dir = config['pkl_dir']
        # print(self.pkl_dir)
        self.pkl_files = [os.path.join(self.pkl_dir, f) for f \
                          in os.listdir(self.pkl_dir) if f.endswith('.pkl')]
        self.char_to_index = {chr(i): i - 97 for i in range(97, 123)} # config['char_to_index']
        self.char_set = set(list(self.char_to_index.keys()))
        # print(self.char_set)
        # print(self.char_to_index)
        self.char_to_index['MASK'] = len(self.char_to_index)
        # print(self.char_to_index)
        self.index_to_char = {v:k for k, v in self.char_to_index.items()}
        # print(self.index_to_char)
        self.index_map = []
        self.data = []  # Cache to store data in memory

        self._build_index_map()
        self._cur_epoch = 0
        self._max_epochs = config['max_epochs']  # Initialize from config
        self._drop_uniform = config['drop_uniformly']
        self._vocab_size = config['vocab_size']
        self._integer_indices = config['integer_indices']

    def _build_index_map(self):
        # Loading data into memory once
        for file_idx, pkl_file in enumerate(self.pkl_files):
            with open(pkl_file, 'rb') as f:
                file_data = pickle.load(f)
                self.data.extend(file_data)
                for sample_idx in range(len(file_data)):
                    self.index_map.append((file_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        word, unique_pos, chars = self.data[sample_idx]
        
        # print(word)
        # print(unique_pos)
        # print(chars)    
        
        drop_probability = 1 / (1 + np.exp(-self._cur_epoch / self._max_epochs))
        number_to_drop = np.random.binomial(len(unique_pos), drop_probability)

        if number_to_drop == 0:
            number_to_drop = 1

        if self._drop_uniform:
            positions_to_drop = np.random.choice(len(unique_pos), \
                                            number_to_drop, replace=False)
        else:
            probabilities = [1 / len(pos) for pos in unique_pos]
            normalized_probs = [prob / sum(probabilities) for prob in probabilities]
            positions_to_drop = np.random.choice(len(unique_pos), \
                                    number_to_drop, p=normalized_probs, replace=False)

        dropped_indices = [idx for pos in positions_to_drop for idx in unique_pos[pos]]

        # print(dropped_indices)

        # print(np.argmax(word, axis=1))

        input_vector = np.copy(word)
        self.mask = np.zeros((1, self._vocab_size + 1)) # for MASK '_'
        self.mask[0, self._vocab_size] = 1
        input_vector[dropped_indices] = self.mask

        if self._integer_indices:
            input_vector = np.argmax(input_vector, axis=1)

        # print(input_vector)

        target = np.clip(np.sum(word[dropped_indices], axis=0), 0, 1)
        # print(target.shape)
        # print(target[self.char_to_index['MASK']])
        # assert target[self.char_to_index['MASK']] == 0
        target = target[:-1]  # Remove the mask dimension from target
        # print(target.shape)

        missing_characters = np.array(sorted(list(self.char_set - set(chars))))
        # print(missing_characters)
        number_of_misses = np.random.randint(0, 10)
        missing_character_indices = np.random.choice(missing_characters, number_of_misses)
        # print(missing_character_indices)
        missing_character_indices = list(set([self.char_to_index[char] \
                                              for char in missing_character_indices]))

        missing_vector = np.zeros(self._vocab_size)
        missing_vector[missing_character_indices] = 1

        return input_vector, target, missing_vector
    
        # return word, unique_pos, chars

    @property
    def cur_epoch(self):
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch):
        self._cur_epoch = epoch

    @property
    def max_epochs(self):
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, epochs):
        if epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        self._max_epochs = epochs