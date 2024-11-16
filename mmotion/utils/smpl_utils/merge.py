import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch

from mmotion.utils.smpl_utils.smpl_key_const import GLOBAL_ORIENT, BODY_POSE, JAW_POSE, LEYE_POSE, REYE_POSE, \
    LEFT_HAND_POSE, RIGHT_HAND_POSE, TRANSL, BETAS


def merge_pose(smplx_dict: Dict):
    poses = torch.cat(
        [
            smplx_dict[GLOBAL_ORIENT],
            smplx_dict[BODY_POSE],
            smplx_dict[JAW_POSE],
            smplx_dict[LEYE_POSE],
            smplx_dict[REYE_POSE],
            smplx_dict[LEFT_HAND_POSE],
            smplx_dict[RIGHT_HAND_POSE],
        ],
        dim=-1
    )
    res_dict = {
        'poses': poses,
        'transl': smplx_dict[TRANSL],
        'betas': smplx_dict[BETAS]
    }
    return res_dict


def merge_persons(smplx_dict_list: List[Dict]) -> Dict:
    merged_dict = {key: torch.empty(0) for key in smplx_dict_list[0].keys()}
    for key in merged_dict.keys():
        # merge t c to -> t num_person c
        merged_dict[key] = torch.stack([smplx_dict[key] for smplx_dict in smplx_dict_list], dim=1)
    return merged_dict
