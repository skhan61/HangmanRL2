import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EmbeddingLSTM(nn.Module):
    def __init__(self, vocab_size=28, embedding_dim=100, hidden_dim=128,
                 num_layers=3, bidirectional=True, dropout=0.1, max_norm=1, feature_dim=4):
        super(EmbeddingLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, max_norm=max_norm, padding_idx=0)
        self.feature_transform = nn.Linear(feature_dim, feature_dim)
        self.feature_bn = nn.BatchNorm1d(feature_dim)
        
        lstm_input_dim = embedding_dim + feature_dim
        self.dropout1 = nn.Dropout(dropout)  # Dropout before LSTM
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)  # Dropout after LSTM
        self.extractor = ExtractLastTensor()
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier_bn = nn.BatchNorm1d(lstm_output_dim)
        self.dropout3 = nn.Dropout(dropout)  # Dropout before classifier

        self._initialize_weights()

    def forward(self, states, lengths, features):
        # device = states.device  # Capture the device of the input tensors
        
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        states = states[sorted_indices]
        features = features[sorted_indices]

        x = self.embedding(states)
        transformed_features = self.feature_transform(features)

        # Apply batch normalization conditionally based on batch size
        if transformed_features.shape[0] > 1:
            transformed_features = self.feature_bn(transformed_features)
        else:
            transformed_features = transformed_features

        # Ensure additional tensors are on the correct device
        features_repeated = transformed_features.unsqueeze(1).repeat(1, x.size(1), 1) #.to(device)
        
        x = torch.cat([x, features_repeated], dim=-1)
        x = self.dropout1(x)
        x_packed = pack_padded_sequence(x, sorted_lengths.cpu(), \
                                        batch_first=True, enforce_sorted=False)
        
        x_packed, _ = self.lstm(x_packed)
        x, _ = pad_packed_sequence(x_packed, batch_first=True)
        x = self.dropout2(x)
        x = self.extractor(x, sorted_lengths)
        
        # Apply batch normalization conditionally based on batch size
        if x.shape[0] > 1:
            x = self.classifier_bn(x)
        else:
            x = x
        
        x = self.dropout3(x)
        
        return x

# Further implementation or usage code goes here.
#     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Add forget gate bias initialization
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

class ExtractLastTensor(nn.Module):
    def forward(self, x, lengths):
        # Extracts the output at the last timestep of each sequence, 
        # considering the actual lengths
        return x[torch.arange(x.size(0)), lengths - 1]