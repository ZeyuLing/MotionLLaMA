import torch
from diffusers.models.controlnet import zero_module
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from mmotion.models.generators.motion_diffuse.stylization import StylizationBlock


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, src_mask):
        """
        :param x: b n d
        :param emb: b d
        :param src_mask: b n, 1 for mask, 0 for valid
        :return:
        """
        if src_mask.dim() == 2:
            src_mask = src_mask.unsqueeze(-1)  # Shape: (B, T, 1)

        B, T, D = x.shape
        H = self.num_head  # Number of attention heads

        # Normalize input
        x_norm = self.norm(x)

        # Linear projections
        query = self.query(x_norm)  # Shape: (B, T, D)
        key = self.key(x_norm)  # Shape: (B, T, D)
        value = self.value(x_norm)  # Shape: (B, T, D)

        # Create boolean mask: True where src_mask == 1 (positions to mask)
        mask = (src_mask == 1)  # Shape: (B, T, 1)

        # Apply mask to key: set masked positions to -inf
        key = key.masked_fill(mask, float('-inf'))

        # Apply mask to value: set masked positions to 0
        value = value.masked_fill(mask, 0.0)

        # Rearrange tensors for multi-head attention
        # From (B, T, D) to (B * H, T, D_head)
        query = rearrange(query, 'b t (h d) -> (b h) t d', h=H)
        key = rearrange(key, 'b t (h d) -> (b h) t d', h=H)
        value = rearrange(value, 'b t (h d) -> (b h) t d', h=H)

        # Apply softmax to query and key
        query = F.softmax(query, dim=-1)  # Along feature dimension
        key = F.softmax(key, dim=1)  # Along sequence length

        # Compute attention
        # key^T: (B * H, D_head, T)
        # value: (B * H, T, D_head)
        attention = torch.bmm(key.transpose(1, 2), value)  # Shape: (B * H, D_head, D_head)

        # Compute output
        # query: (B * H, T, D_head)
        # attention: (B * H, D_head, D_head)
        y = torch.bmm(query, attention.transpose(1, 2))  # Shape: (B * H, T, D_head)

        # Reshape back to (B, T, D)
        y = rearrange(y, '(b h) t d -> b t (h d)', h=H)

        # Residual connection and output projection
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        :param x: b t c
        :param xf: b n c, text feature
        :param emb: embedding b c
        :return:
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(rearrange(query, 'b t (h c) -> (b h) t c', h=H), dim=-1)
        key = F.softmax(rearrange(key, 'b n (h c) -> (b h) n c', h=H), dim=1)
        value = rearrange(self.value(self.text_norm(xf)),
                          'b n (h c) -> (b h) n c', h=H)
        # (B H), HD, HD
        attention = torch.bmm(key.transpose(-1, -2), value)
        y = torch.bmm(query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = LinearTemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearTemporalCrossAttention(
            latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
