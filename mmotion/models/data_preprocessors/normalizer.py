from typing import Dict, Union, List

import numpy as np
from torch import nn, Tensor

from mmotion.registry import MODELS
import torch


@MODELS.register_module()
class BaseMotionNormalizer(nn.Module):
    def __init__(self, norm_path: str = None, mean_keys: Union[str, List[str]] = 'mean',
                 std_keys: Union[str, List[str]] = 'std', average_std: bool = False, bias_std_keys=[], feat_bias=1.):
        """
        :param norm_path: the normalization file
        :param mean_keys: the keys of the mean
        :param std_keys: the keys of the std
        :param average_std: HumanML3D calculates the mean of each item in the standard deviation
         (e.g., velocity, position, rotation) separately and divides each by its respective mean.
        :param feat_bias: As in HumanML3D, some part of std will be divided with feat bias
        :param bias_std_keys: As in HumanML3D, some part of std will be divided with feat bias
        """
        super().__init__()
        self.enable_norm = norm_path is not None
        if isinstance(mean_keys, str):
            mean_keys = [mean_keys]
        if isinstance(std_keys, str):
            std_keys = [std_keys]

        self.mean_keys = mean_keys
        self.std_keys = std_keys
        self.bias_std_keys = bias_std_keys
        self.feat_bias = feat_bias
        self.average_std = average_std

        self.load_mean_std(norm_path)

    def load_mean_std(self, norm_path: str):
        if not self.enable_norm:
            mean = torch.tensor(0.)
            std = torch.tensor(1.)
        else:
            statistic = np.load(norm_path, allow_pickle=True)
            mean = torch.cat([torch.from_numpy(statistic[k]) for k in self.mean_keys], dim=-1).float()
            std_list = []
            for k in self.std_keys:
                std_item:np.ndarray = statistic[k]
                if self.average_std:
                    std_item = np.repeat(std_item.mean(keepdims=True), std_item.shape[-1], -1)
                if k in self.bias_std_keys:
                    std_item /= self.feat_bias
                std_list.append(std_item)
            std = torch.from_numpy(np.concatenate(std_list, axis=-1)).float()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

    def normalize(self, motion: Union[List[Tensor], Tensor]):
        is_tensor = isinstance(motion, Tensor)
        mean = self.mean.to(motion[0])
        std = self.std.to(motion[0])
        motion = [(m - self.mean) / self.std for m in motion]
        if is_tensor:
            motion = torch.stack(motion, dim=0)

        return motion, mean, std

    def inv_normalize(self, motion: Union[List[Tensor], Tensor]):
        if isinstance(motion, list):
            motion = [m * self.std + self.mean
                      if m is not None else None for m in motion]
        else:
            motion = motion * self.std + self.mean

        return motion
