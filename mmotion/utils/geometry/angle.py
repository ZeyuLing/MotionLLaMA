import numpy as np
from typing import Union

import torch


def angle_normalization(angle:Union[float, np.ndarray, torch.Tensor]):
    """ normalize angles to (-pi,pi)
    :param angle: The input angle must be in radians.
    :return:
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi