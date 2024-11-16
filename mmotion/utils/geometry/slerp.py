from typing import Union

import torch
from einops import rearrange
import torch.nn.functional as F
from mmotion.utils.geometry.rotation_convert import rot_dim, rot_convert

def slerp(a: torch.Tensor, b: torch.Tensor, t: Union[float, torch.Tensor]):
    """ Sphere linear interpolation
    :param a: The start quaternion. [c] or [t, c] or [t, j, c]
    :param b: The end quaternion. [c] or [t, c] or [t, j, c]
    :param t: the interpolation weight, determine the influence of the start and end.
    Could be a float or a tensor which can be broadcast to a,b
    :return:
    """
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    d = (a * b).sum(-1, keepdim=True)
    d = torch.clamp(d, -1., 1.)
    p = t * torch.acos(d)
    c = F.normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = F.normalize(d, p=2, dim=-1)
    return d


def motion_sphere_interpolate(motion: torch.Tensor, rot_type: str, T: int):
    """
    :param motion: motion rotation vector. shape in [t, j, c]
    :param rot_type: axis angle, quaternion, euler, matrix or cont6d.
    :param T: target num of frames
    :return:
    """
    assert rot_type in rot_dim.keys(), (f"supported rotation rep: {rot_dim.keys()},"
                                        f" but got {rot_type}")
    merged = False
    if len(motion.shape) == 2:
        merged = True
        dim = rot_dim[rot_type]
        motion = rearrange(motion, 't (j c) -> t j c', c=dim)
    t, j, c = motion.shape
    # t j c
    q_motion = rot_convert(motion, rot_type, 'quaternion')

    scale = T / t
    src_timestamps = torch.arange(0, t).to(motion)
    tgt_timestamps = (torch.arange(0, T) / scale).to(motion)
    ts_delta = torch.diff(src_timestamps)

    lower_idx = torch.searchsorted(src_timestamps, tgt_timestamps, right=True) - 1
    lower_idx = torch.clamp(lower_idx, 0, t - 2)
    upper_idx = lower_idx + 1

    q0 = q_motion[lower_idx]
    q1 = q_motion[upper_idx]

    t_interp = (tgt_timestamps - src_timestamps[lower_idx]) / ts_delta[lower_idx]
    assert torch.all(t_interp) < 1. and torch.all(t_interp) >= 0., f"t_interp should be between 0 and 1, but got {t_interp}"
    t_interp = rearrange(t_interp, 't -> t 1 1')
    q_motion = slerp(q0, q1, t_interp)

    motion = rot_convert(q_motion, 'quaternion', rot_type)
    if merged:
        motion = rearrange(motion, 't j c -> t (j c)')
    return motion
