from torch import nn
import torch



def get_activation(activation_type: str, alpha: float = 1., negative_slope: float = 0.2):
    if activation_type is None:
        return nn.Identity()
    if activation_type == 'relu':
        return nn.ReLU()
    if activation_type == 'silu':
        return nn.SiLU()
    if activation_type == 'gelu':
        return nn.GELU()
    if activation_type == 'elu':
        return nn.ELU(alpha)
    if activation_type == 'leaky_relu':
        return nn.LeakyReLU(negative_slope)
    raise NotImplementedError(f"activation type {activation_type} not supported")
