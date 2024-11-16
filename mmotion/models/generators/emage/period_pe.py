import math
import torch
from torch import nn


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=300, max_seq_len=5000):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)  # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
