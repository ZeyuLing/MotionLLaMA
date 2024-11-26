import numpy as np
from typing import Union

import torch
from einops import rearrange
from torch import nn

from mmotion.utils.geometry.laplacian_matrix import compute_laplacian_matrix


class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, channels:int,
                 adjacency_matrix:Union[np.ndarray, torch.Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix)

        L = compute_laplacian_matrix(adjacency_matrix, normalized=True)

        laplacian_pe = laplacian_positional_encoding(L, channels)
        self.register_buffer('laplacian_pe', laplacian_pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: b t j c, j represents node nums
        :return:
        """
        x = x + self.laplacian_pe[None, None]
        return x

def laplacian_positional_encoding(L:torch.Tensor, pos_enc_dim:int):
    """
    :param L: normalized laplacian matrix
    :param pos_enc_dim: dimension of positional encoding
    :return:
    """
    # Eigenvectors with scipy
    eig_val, eig_vec = torch.linalg.eigh(L,) # for 40 PEs
    eig_vec = eig_vec[:, torch.argsort(eig_val)] # increasing order
    return eig_vec[:, 1:pos_enc_dim+1]

