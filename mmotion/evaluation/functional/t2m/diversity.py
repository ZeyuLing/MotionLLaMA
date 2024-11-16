from typing import Union
import torch
import numpy as np

from torch.linalg import norm
import torch.nn.functional as F

def cal_diversity(motion: Union[torch.Tensor],
                  diversity_times: int = 300):
    """
    :param motion: b c. All predicted motion feature vectors from vqvae
    or extracted motion embeddings from eval_wrapper.
    :param diversity_times: randomly choose diversity_times samples from the generated motions.
    :return: calculated diversity
    """
    assert len(motion.shape) == 2, f'the input should be in shape [num_samples, num_channels], but got {motion.shape}'
    assert motion.shape[0] > diversity_times
    num_samples = motion.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    diversity = norm(F.normalize(motion[first_indices].float(), dim=1)
                     - F.normalize(motion[second_indices].float(), dim=1), dim=1, ord=1)
    return diversity.mean()
