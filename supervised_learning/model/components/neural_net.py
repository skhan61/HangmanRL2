import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from supervised_learning.model.components.categorical import Categorical

from rl_squared.utils.torch_utils import init_module

class EmbeddingGRU(nn.Module):
    def __init__(self, config):
        super(EmbeddingGRU, self).__init__()

        self._config = config

        self.embedding = nn.Embedding(self._config['vocab_size'] + 2,  # + 2 for <pad> and extra chars
                    self._config['embedding_dim'], padding_idx=27)
        
        self.rnn = nn.GRU(input_size=config['embedding_dim'], 
                        hidden_size=self._config['hidden_dim'], 
                        num_layers=self._config['num_layers'],
                        dropout=self._config['dropout'],
                        bidirectional=True, 
                        batch_first=True)
        
        # self.miss_char_fc = nn.Linear(self._config['vocab_size'], \
        #                               self._config['miss_chars_fc_out_dim'])
        
        self.miss_char_fc = init_module(
                        nn.Linear(self._config['vocab_size'], 
                                self._config['miss_chars_fc_out_dim']),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01)
        
        self.output_fc = nn.Sequential(nn.ReLU(),
                    init_module(
                        nn.Linear(self._config['miss_chars_fc_out_dim'] + 
                                self._config['hidden_dim'] * 2, 
                                self._config['vocab_size']),
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        gain=0.01
        ))
        
        # self.output_dist = Categorical(self._config['miss_chars_fc_out_dim'] + 
        #                                self._config['hidden_dim'] * 2, 26)

    def forward(self, inputs, lengths, miss_chars):
        # inputs = inputs.float()
        # miss_chars = miss_chars.float()

        x = self.embedding(inputs.long())
        # print(x.shape)

        x = pack_padded_sequence(x, lengths.cpu(), \
                                 batch_first=True, enforce_sorted=False)
        
        x, hidden = self.rnn(x)

        # print(hidden.shape)

        hidden = hidden.view(self.rnn.num_layers, 2, -1, \
                             self.rnn.hidden_size)
        
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)

        # print(hidden.shape)

        miss_chars_logits = self.miss_char_fc(miss_chars.float())

        # print(miss_chars_logits.shape)

        concatenated = torch.cat((hidden, miss_chars_logits), dim=1)
        
        # print(concatenated.shape)

        logits = self.output_fc(concatenated)

        return logits

        # output_dist = self.output_dist(concatenated)

        # return output_dist.logits, output_dist


