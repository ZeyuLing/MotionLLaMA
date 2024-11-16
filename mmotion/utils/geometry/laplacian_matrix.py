import numpy as np
from typing import Union

import torch


def compute_laplacian_matrix(A:Union[torch.Tensor, np.ndarray], normalized=False):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    A = A.float()
    if normalized:
        degree = torch.sum(A, dim=1)
        degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5)).to(A)
        degree_inv_sqrt[degree == 0] = 0

        L = torch.eye(A.size(0)) - torch.mm(torch.mm(degree_inv_sqrt, A), degree_inv_sqrt)
    else:
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A

    return L