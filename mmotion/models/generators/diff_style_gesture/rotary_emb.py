from einops import rearrange
from torch import nn
import torch


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rel_pos = SinusoidalEmbeddings(dim)

    def forward(self, x):
        pos_emb = self.rel_pos(x)
        x = x * pos_emb.cos() + rotate_half(x) * pos_emb.sin()
        return x


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)
