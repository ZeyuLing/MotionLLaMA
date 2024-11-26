from typing import Tuple

import torch


def global2local(transl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ following UDE-2(https://zixiangzhou916.github.io/UDE-2/), use local translation to replace global ones.
    :param transl: global translation
    :return: (offset at every frame to the previous one, the origin point position)
    """
    local = transl[1:] - transl[:-1]
    local = torch.cat([torch.zeros_like(transl[:1]), local])
    origin = transl[0]
    return local, origin

def local2global(transl: torch.Tensor, origin: torch.Tensor=None) -> torch.Tensor:
    """
    :param transl: local translation(relative offset to previous frame, the first frame will be 0). [t, c] or [b, t, c]
    :param init_location: the position of first frame. [c] or [b, c] or a list of batch
    :return:
    """
    transl = torch.cumsum(transl, dim=-2)
    if origin is not None:
        transl = transl + origin.unsqueeze(-2)
    return transl
