import os
import sys
from typing import List

import torch
from einops import rearrange

sys.path.append(os.curdir)
from mmotion.models.utils.mask_utils import create_src_key_padding_mask
from mmotion.registry import MODELS
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = rearrange(x, 'b n (p c) -> b (n p) c', c=self.out_channels)
        return x


@MODELS.register_module()
class TransMotionDecoderV1(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_channels=156,
                 hidden_size=512,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.unpatchify = FinalLayer(
            hidden_size, patch_size, in_channels,
        )
        trans_layer = nn.TransformerEncoderLayer(
            hidden_size, nhead=num_heads, dim_feedforward=int(hidden_size * mlp_ratio),
            batch_first=True
        )
        self.trans = nn.TransformerEncoder(trans_layer, num_layers=depth)

    def forward(self, x, num_frames: List[int] = None):
        """
        :param x: [B, C, T]
        :param num_frames: valid num of frames of x, which is a padded sequence
        :return:
        """
        mask = None
        if num_frames is not None:
            num_tokens = [nf // self.patch_size for nf in num_frames]
            mask = create_src_key_padding_mask(num_tokens)

        x = rearrange(x, "b c t -> b t c")
        x = self.trans(x, src_key_padding_mask=mask)

        x = self.unpatchify(x)
        x = rearrange(x, "b t c -> b c t")
        return x


if __name__ == "__main__":
    x = torch.rand([2, 512, 75])
    num_frames = [201, 300]
    model = TransMotionDecoderV1()
    out = model(x, num_frames)
    print(out.shape)
