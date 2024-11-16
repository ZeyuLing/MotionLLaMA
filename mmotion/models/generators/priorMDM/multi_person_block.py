from torch import nn
import torch


class MultiPersonBlock(nn.Module):
    def __init__(self, num_layers, latent_dim):
        super().__init__()
        num_heads = 4
        ff_size = 1024
        dropout = 0.1
        activation = 'gelu'

        self.aggregation = nn.Linear(latent_dim * 2, latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          batch_first=True,
                                                          activation=activation)
        self.model = nn.TransformerEncoder(seqTransEncoderLayer,
                                           num_layers=num_layers)

    def forward(self, person_a, person_b, mask):
        x = self.aggregation(torch.cat((person_a, person_b), dim=-1))
        out = self.model(x, src_key_padding_mask=mask)

        return out
