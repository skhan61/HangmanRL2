# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import gymnasium as gym

# class HangmanFeaturesExtractor(nn.Module):
#     def __init__(self, features_dim: int = 128,
#                  vocab_size=28, embedding_dim=10, hidden_dim=128,
#                  num_layers=3, bidirectional=True, dropout=0.1, max_norm=1):
#         super(HangmanFeaturesExtractor, self).__init__()

#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.dropout_rate = dropout
#         self.max_norm = max_norm
        
#         # Embeddings and Linear layers
#         self.letter_embed = nn.Embedding(self.vocab_size, self.embedding_dim, max_norm=self.max_norm)
#         self.guessed_letters_fc = nn.Linear(26, 10)
#         self.remaining_attempts_fc = nn.Linear(1, 5)
        
#         # LSTM Layer
#         lstm_input_dim = self.embedding_dim + 10 + 5  # embedding + guessed letters + remaining attempts
#         self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.hidden_dim,
#                             num_layers=self.num_layers, dropout=self.dropout_rate if self.num_layers > 1 else 0,
#                             bidirectional=self.bidirectional, batch_first=True)
        
#         # Output dimension calculation
#         lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
#         self.final_fc = nn.Linear(lstm_output_dim, features_dim)

#     def forward(self, observations):
#         # print(observations)
#         guessed_letters = observations['guessed_letters'].float()
#         masked_word = observations['masked_word'].long()
#         remaining_attempts = observations['remaining_attempts'].float().unsqueeze(-1)

#         # Embedding and processing
#         masked_word_emb = self.letter_embed(masked_word)
#         guessed_letters_features = self.guessed_letters_fc(guessed_letters)
#         remaining_attempts_features = self.remaining_attempts_fc(remaining_attempts)

#         # Reshape for concatenation
#         guessed_letters_expanded = guessed_letters_features.unsqueeze(1).repeat(1, masked_word_emb.size(1), 1)
#         remaining_attempts_expanded = remaining_attempts_features.repeat(1, masked_word_emb.size(1), 1)

#         # Combine features for LSTM input
#         combined_features = torch.cat([
#             masked_word_emb,
#             guessed_letters_expanded,
#             remaining_attempts_expanded
#         ], dim=-1)

#         # LSTM processing
#         lstm_out, _ = self.lstm(combined_features)

#         # Assuming we take the last output for simplicity
#         final_features = lstm_out[:, -1, :]

#         # Output through final fully connected layer
#         output = self.final_fc(final_features)

#         return output

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class HangmanFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, 
                 features_dim: int = 128,
                 vocab_size=28, embedding_dim=10, 
                 hidden_dim=128,
                 num_layers=3, bidirectional=True, 
                 dropout=0.1, max_norm=1):
        
        super(HangmanFeaturesExtractor, self).__init__(observation_space, \
                                                features_dim)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.max_norm = max_norm
        # self.features_dim = features_dim
        
        # Embeddings and Linear layers
        self.letter_embed = nn.Embedding(self.vocab_size, self.embedding_dim, max_norm=self.max_norm)
        self.guessed_letters_fc = nn.Linear(26, 10)
        self.remaining_attempts_fc = nn.Linear(1, 5)
        
        # LSTM Layer
        lstm_input_dim = self.embedding_dim + 10 + 5  # embedding + guessed letters + remaining attempts
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, dropout=self.dropout_rate if self.num_layers > 1 else 0,
                            bidirectional=self.bidirectional, batch_first=True)
        
        # Output dimension calculation
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.final_fc = nn.Linear(lstm_output_dim, features_dim)

    def forward(self, observations):
        guessed_letters = observations['guessed_letters'].float()
        masked_word = observations['masked_word'].long()
        remaining_attempts = observations['remaining_attempts'].float().unsqueeze(-1)

        # Embedding and processing
        masked_word_emb = self.letter_embed(masked_word)
        guessed_letters_features = self.guessed_letters_fc(guessed_letters)
        remaining_attempts_features = self.remaining_attempts_fc(remaining_attempts)

        # # Print shapes for debugging
        # print("Shape of guessed_letters:", guessed_letters.shape)
        # print("Shape of masked_word:", masked_word.shape)
        # print("Shape of remaining_attempts after unsqueeze:", remaining_attempts.shape)
        # print("Shape of masked_word_emb:", masked_word_emb.shape)
        # print("Shape of guessed_letters_features:", guessed_letters_features.shape)
        # print("Shape of remaining_attempts_features:", remaining_attempts_features.shape)

        # Reshape for concatenation
        guessed_letters_expanded = guessed_letters_features.unsqueeze(1).repeat(1, masked_word_emb.size(1), 1)
        remaining_attempts_expanded = remaining_attempts_features.repeat(1, masked_word_emb.size(1), 1)

        # Combine features for LSTM input
        combined_features = torch.cat([
            masked_word_emb,
            guessed_letters_expanded,
            remaining_attempts_expanded
        ], dim=-1)

        # print("Shape of combined_features:", combined_features.shape)

        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)

        # Assuming we take the last output for simplicity
        final_features = lstm_out[:, -1, :]
        # print("Shape of final_features:", final_features.shape)

        # Output through final fully connected layer
        output = self.final_fc(final_features)
        # print("Shape of output:", output.shape)

        return output
