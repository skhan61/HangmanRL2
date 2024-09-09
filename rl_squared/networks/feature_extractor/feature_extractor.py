import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from supervised_learning.model.components.categorical import Categorical

from rl_squared.utils.torch_utils import init_module

def unpack_batch(batch_tensor):
    """
    Unpacks a batch tensor into its components.
    
    Args:
        batch_tensor (torch.Tensor): A tensor containing the concatenated elements of the batch.
        
    Returns:
        tuple: Contains separated components (masked_word, guessed_letters, remaining_attempts, 
               action_mask, previous_actions, previous_rewards, previous_done).
    """
    masked_word = batch_tensor[:, :35]
    guessed_letters = batch_tensor[:, 35:61]  # 35 + 26
    remaining_attempts = batch_tensor[:, 61].unsqueeze(1)  # Keeping dimension
    action_mask = batch_tensor[:, 62:88]  # 62 to 62+26
    previous_actions = batch_tensor[:, 88].unsqueeze(1)  # Keeping dimension
    previous_rewards = batch_tensor[:, 89].unsqueeze(1)  # Keeping dimension
    previous_done = batch_tensor[:, 90].unsqueeze(1)  # Keeping dimension

    return (masked_word, guessed_letters, remaining_attempts,
            action_mask, previous_actions, previous_rewards, previous_done)

class EmbeddingGRU(nn.Module):
    def __init__(self):
        super(EmbeddingGRU, self).__init__()

        # self._config = config

        self._vocal_size = 26
        self._embedding_dim = 128

        self.embedding = nn.Embedding(num_embeddings=self._vocal_size + 2, \
                                    embedding_dim=self._embedding_dim, \
                                    padding_idx=0)
        
        self.rnn = nn.GRU(input_size=self._embedding_dim, 
                        hidden_size=128, 
                        num_layers=3,
                        dropout=0.3,
                        bidirectional=True, 
                        batch_first=True)
        
        self.guessed_letter_fc = init_module(
                        nn.Linear(self._vocal_size, 128),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)
        
        self.remaining_attempts_fc = init_module(
                        nn.Linear(1, 10),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)
        
        self.action_mask_fc = init_module(
                        nn.Linear(self._vocal_size, 128),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)

        self.previous_action_fc = init_module(
                        nn.Linear(1, 10),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)

        self.previous_reward_fc = init_module(
                        nn.Linear(1, 10),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)
        
        self.previous_done_fc = init_module(
                        nn.Linear(1, 10),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)
        
        self.output_fc = nn.Sequential(nn.ReLU(),
                    init_module(
                        nn.Linear(424, 116), 
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)) # for dim matching
       
        
    def forward(self, x):
        # inputs = inputs.float()
        # miss_chars = miss_chars.float()

        masked_word, guessed_letters, remaining_attempts, \
            action_mask, previous_actions, \
            previous_rewards, previous_done = unpack_batch(x)
        
        # unpacked_data = unpack_batch(x)
        # # print(unpacked_data)
        
        # # Print the shapes to verify
        # for data, name in zip(unpacked_data, ["Masked Word", "Guessed Letters", "Remaining Attempts", "Action Mask",
        #                                     "Previous Actions", "Previous Rewards", "Previous Done"]):
        #     print(f"{name} Shape: {data.shape}")

        # print(masked_word)

        masked_word_emb = self.embedding(masked_word.long())
        # print(masked_word_emb.shape)

        lengths = (masked_word != 0).sum(1)
        # print(lengths)

        x = pack_padded_sequence(masked_word_emb, lengths.cpu(), \
                                 batch_first=True, enforce_sorted=False)
        
        _, hidden = self.rnn(x)

        # print(hidden.shape)

        hidden = hidden.view(self.rnn.num_layers, 2, -1, \
                             self.rnn.hidden_size)
        
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)

        # print(hidden.shape)

        guessed_letters_logits = self.guessed_letter_fc(guessed_letters.float())
        # print(guessed_letters_logits.shape)

        remaining_attempts_logits = self.remaining_attempts_fc(remaining_attempts.float())

        # print(remaining_attempts_logits.shape)

        action_mask_logits = self.action_mask_fc(action_mask.float())

        # print(action_mask_logits.shape)

        previous_actions_logits = self.previous_action_fc(previous_actions.float())

        # print(previous_actions_logits.shape)

        previous_rewards_logits = self.previous_reward_fc(previous_rewards.float())

        # print(previous_rewards_logits.shape)

        previous_done_logits = self.previous_done_fc(previous_done.float())

        # print(previous_done_logits.shape)

        # print(hidden.shape)

        # print(miss_chars_logits.shape)

        concatenated = torch.cat((hidden, remaining_attempts_logits, \
                    action_mask_logits, previous_actions_logits, previous_rewards_logits, previous_done_logits), dim=1)
        
        # print(concatenated.shape)

        logits = self.output_fc(concatenated)

        # print(logits.shape)

        return logits
    