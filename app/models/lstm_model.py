import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.5, pad_idx=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           batch_first=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           bidirectional=True)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        
        hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_last = self.layer_norm(hidden_last)
        hidden_last = self.relu(self.fc1(hidden_last))
        hidden_last = self.dropout(hidden_last)
        
        return self.fc2(hidden_last)