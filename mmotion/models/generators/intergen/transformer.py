from diffusers.models.controlnet import zero_module
from torch import nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class AdaLN(nn.Module):

    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            # nn.Linear(embed_dim, latent_dim, bias=True),
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h


class VanillaSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout, batch_first=True,
                                               add_zero_attn=True)

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        """
        x_norm = self.norm(x, emb)
        y = self.attention(x_norm, x_norm, x_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return y


class VanillaCrossAttention(nn.Module):

    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, kdim=xf_latent_dim, vdim=xf_latent_dim,
                                               dropout=dropout, batch_first=True, add_zero_attn=True)

    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        x_norm = self.norm(x, emb)
        xf_norm = self.xf_norm(xf, emb)
        y = self.attention(x_norm, xf_norm, xf_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return y


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y
