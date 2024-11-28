from typing import Optional

from einops import rearrange
from timm.layers import GroupNorm
from torch import nn, Tensor
import torch.nn.functional as F


def get_norm(norm_type: str, in_channels, zq_channels: Optional[int] = None):
    if norm_type is None:
        return nn.Identity()
    if norm_type == "spatial":
        return SpatialNorm1D(in_channels, zq_channels)
    if norm_type == 'group':
        return group_norm_32(in_channels)
    if norm_type == 'layer':
        return LayerNorm(in_channels)
    if norm_type == 'batch':
        return nn.BatchNorm1d(in_channels)
    raise NotImplementedError(f'normalization type {norm_type} not supported')

class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b c t -> b t c')
        input = super().forward(input)
        input = rearrange(input, 'b t c -> b c t')
        return input


def group_norm_32(in_channels):
    return GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialNorm1D(nn.Module):
    def __init__(self, f_channels, zq_channels, freeze_norm_layer=False, add_conv=False):
        super().__init__()
        self.norm_layer = group_norm_32(f_channels)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv1d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv1d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv1d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f, zq):
        """
        :param f: b c_1 t_1
        :param zq: b c_2 t_2
        :return:
        """
        T = f.shape[-1]
        zq = F.interpolate(zq, size=T, mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
