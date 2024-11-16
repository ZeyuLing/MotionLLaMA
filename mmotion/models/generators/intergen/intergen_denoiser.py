import numpy as np
import torch
from diffusers.models.controlnet import zero_module
from torch import nn

from mmotion.models.generators.intergen.transformer import TransformerBlock
from mmotion.registry import MODELS


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        embed = self.time_embed(self.sequence_pos_encoder.pe[timesteps])
        return embed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)#.transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x


@MODELS.register_module()
class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats=156,
                 latent_dim=1024,
                 ff_size=2048,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu"):
        super().__init__()

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(num_heads=num_heads, latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mask=None, cond=None):
        """
        :param x: b t c
        :param timesteps: b
        :param mask: b n
        :param cond: b c
        :return:
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.zeros(B, T).to(x_a.device)

        for i, block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, emb, mask)
            h_b = block(h_b_prev, h_a_prev, emb, mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)

        return output
