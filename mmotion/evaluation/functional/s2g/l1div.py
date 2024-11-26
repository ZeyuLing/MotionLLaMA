"""
Follow BEAT implementation https://github.com/PantoMatrix/PantoMatrix/blob/6ca70b9541285b124da2eeedcd80f7c5a54eb111/scripts/BEAT_2022/utils/metric.py#L34
line 12 to 24
"""
from typing import List

import torch
from einops import rearrange


def cal_l1div(motion: List[torch.Tensor]):
    """
    :param motion: list of [t, c]
    :return:
    """
    motion = torch.cat(motion, dim=0).float()
    mean = torch.mean(motion, dim=0)  # shape: [c]
    abs_diff = torch.abs(motion - mean)  # shape: [n, c]
    sum_l1 = torch.sum(abs_diff)  # scalar
    avg_l1 = sum_l1 / motion.shape[0]  # scalar
    return avg_l1.item()
