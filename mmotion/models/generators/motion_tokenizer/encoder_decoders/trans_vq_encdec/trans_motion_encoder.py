import os
import sys
from typing import List

import torch
from einops import rearrange
sys.path.append(os.curdir)
from mmotion.models.utils.mask_utils import create_src_key_padding_mask
from mmotion.registry import MODELS
from torch import nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    """ 1D Motion Sequence to Patch Embedding
    """

    def __init__(
            self,
            patch_size: int = 4,
            in_chans: int = 3,
            embed_dim: int = 768,
            bias: bool = True,
            dynamic_pad: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.img_size = None
        self.grid_size = None
        self.num_patches = None

        self.dynamic_pad = dynamic_pad

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, T = x.shape

        if self.dynamic_pad:
            pad_t = (self.patch_size - T % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_t))
        x = self.proj(x)
        return x


@MODELS.register_module()
class TransMotionEncoderV1(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_channels=156,
                 hidden_size=512,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed1D(
            patch_size, in_channels, hidden_size,
        )
        trans_layer = nn.TransformerEncoderLayer(
            hidden_size, nhead=num_heads, dim_feedforward=int(hidden_size * mlp_ratio),
            batch_first=True
        )
        self.trans = nn.TransformerEncoder(trans_layer, num_layers=depth)

    def forward(self, x, num_frames: List[int] = None):
        """
        :param x: [B, C, T]
        :param num_frames:
        :return:
        """
        mask = None
        if num_frames is not None:
            num_tokens = [nf // self.patch_size for nf in num_frames]
            mask = create_src_key_padding_mask(num_tokens)

        x = self.patch_embed(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.trans(x, src_key_padding_mask=mask)
        x = rearrange(x, "b t c -> b c t")
        return x

if __name__ == "__main__":
    x = torch.rand([2, 156, 300])
    num_frames = [201, 300]
    model = TransMotionEncoderV1()
    out = model(x, num_frames)
    print(out.shape)