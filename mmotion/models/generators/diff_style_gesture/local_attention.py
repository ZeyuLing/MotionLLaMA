import math

import torch
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, unpack

# constant

TOKEN_SELF_ATTN_VALUE = -5e4


# helper functions

def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def to(t):
    return {'device': t.device, 'dtype': t.dtype}


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim=-1)
    return normed.type(dtype)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)


# main class

class LocalAttention(nn.Module):

    def __init__(
            self,
            window_size,
            num_heads=4,
            causal=False,
            look_backward=1,
            look_forward=None,
            dropout=0.,
            shared_qk=False,
            exact_windowsize=False
    ):
        super().__init__()
        if look_forward is None:
            look_forward = 0 if causal else 1
        assert not (causal and look_forward > 0), 'Cannot look forward if causal'
        self.num_heads = num_heads
        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.shared_qk = shared_qk
        self.exact_windowsize = exact_windowsize

    def forward(self, q, k, v, mask=None):

        window_size = self.window_size
        causal = self.causal
        look_backward = self.look_backward
        look_forward = self.look_forward
        shared_qk = self.shared_qk

        b, n, dim = q.shape
        device = q.device
        scale = dim ** -0.5

        if n % window_size != 0:
            raise ValueError(f'Sequence length ({n}) must be divisible by window size ({window_size})')

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device=device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w=windows, n=window_size)

        # Reshape q, k, v to (batch_size, windows, window_size, dim_head)
        bq = rearrange(q, 'b (w n) (h d) -> (b h) w n d', w=windows, h=self.num_heads)
        bk = rearrange(k, 'b (w n) (h d) -> (b h) w n d', w=windows, h=self.num_heads)
        bv = rearrange(v, 'b (w n) (h d) -> (b h) w n d', w=windows, h=self.num_heads)

        look_around_kwargs = {
            'backward': look_backward,
            'forward': look_forward,
            'pad_value': -1
        }

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(b_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        sim = einsum('b w i d, b w j d -> b w i j', bq, bk) * scale
        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)

        if causal:
            causal_mask = bq_t < bq_k
            if self.exact_windowsize:
                max_causal_window_size = window_size * look_backward
                causal_mask |= bq_t > (bq_k + max_causal_window_size)
            sim = sim.masked_fill(causal_mask, mask_value)

        if mask is not None:
            mask = rearrange(mask, 'b (w n) -> b w n', w=windows)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, 'b w n -> b w 1 n')
            mask = mask.repeat_interleave(self.num_heads, dim=0)
            sim = sim.masked_fill(~mask, mask_value)

        attn = self.dropout(sim.softmax(dim=-1))
        out = einsum('b w i j, b w j d -> b w i d', attn, bv)
        out = rearrange(out, '(b h) w n d -> b (w n) (h d)', h=self.num_heads)

        return out
