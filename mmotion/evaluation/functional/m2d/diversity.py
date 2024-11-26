import torch
from torch import Tensor


def cal_diversity_dance(motion: Tensor):
    """
    Calculate the diversity of predicted motion sequences.

    :param motion: Tensor of shape (batch_size, motion_dim) containing predicted motions.
    :return: The average pairwise distance between motion sequences.
    """
    bs = motion.shape[0]

    # Compute the pairwise distances using broadcasting
    diff = motion.unsqueeze(1) - motion.unsqueeze(0)  # Shape: (bs, bs, motion_dim)
    dist_matrix = torch.norm(diff, dim=2)  # Shape: (bs, bs)

    # Sum the upper triangular part of the distance matrix and average
    dist = dist_matrix.triu(diagonal=1).sum()  # Sum only upper triangle
    average_dist = dist / (bs * (bs - 1) / 2)  # Average pairwise distance

    return average_dist
