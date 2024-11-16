# Given the sample caption, eval the diversity of the modal output
import numpy as np
import torch
from scipy import linalg


def cal_multimodality(activation: torch.Tensor, multimodality_times: int):
    """
    :param activation: [mm_num_samples, mm_repeat, c] extracted generated sample features
    :param multimodality_times: choose multimodality times from mm_repeat for diveristy evaluation
    :return: multimodality
    """
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    # num_repeat
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()
