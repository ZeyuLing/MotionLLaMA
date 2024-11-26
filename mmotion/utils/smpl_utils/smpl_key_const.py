from typing import Dict

import torch

BETAS = 'betas'
EXPRESSION = 'expression'

TRANSL = 'transl'

GLOBAL_ORIENT = 'global_orient'
BODY_POSE = 'body_pose'
LEYE_POSE = 'leye_pose'
REYE_POSE = 'reye_pose'
JAW_POSE = 'jaw_pose'
LEFT_HAND_POSE = 'left_hand_pose'
RIGHT_HAND_POSE = 'right_hand_pose'

JOINTS = 'joints'
ORIGIN = 'origin'

NUM_FRAMES = 'num_frames'
FPS = 'fps'

SMPLX_KEYS = [TRANSL,
              BETAS, EXPRESSION,
              GLOBAL_ORIENT, BODY_POSE, LEYE_POSE, REYE_POSE, JAW_POSE, LEFT_HAND_POSE, RIGHT_HAND_POSE]

ROTATION_KEYS = [GLOBAL_ORIENT, BODY_POSE, JAW_POSE, LEYE_POSE, REYE_POSE, RIGHT_HAND_POSE, LEFT_HAND_POSE]
SHAPE_KEYS = [BETAS, EXPRESSION]

param_dim = {
    BETAS: 10,
    TRANSL: 3,
    GLOBAL_ORIENT: 3,
    EXPRESSION: 10,
    LEYE_POSE: 3,
    REYE_POSE: 3,
    JAW_POSE: 3,
    LEFT_HAND_POSE: 45,
    RIGHT_HAND_POSE: 45,
    BODY_POSE: 63
}

NUM_JOINTS_OF_ROT = {
    GLOBAL_ORIENT: 1,
    LEYE_POSE: 1,
    REYE_POSE: 1,
    JAW_POSE: 1,
    LEFT_HAND_POSE: 15,
    RIGHT_HAND_POSE: 15,
    BODY_POSE: 21
}


def zero_param(num_frames, key):
    return torch.zeros((num_frames, param_dim[key]))


def merge_pose_keys(smpl_dict: Dict):
    """ Merge different smpl pose keys into "poses"
    :param smpl_keys: smpl_dict including body_pose, global_orient, ...
    :return:
    """
    smpl_dict['poses'] = torch.cat([smpl_dict.pop(key) for key in ROTATION_KEYS if key in smpl_dict.keys()], dim=-1)
    return smpl_dict

def extract_smplx_keys(ori_dict: Dict):
    """ The standard smplx files have some extra keys, we use this functions to extract
    the smplx keys used for forward kinematic.
    :param ori_dict:
    :return:
    """
    new_dict = {}
    for key in SMPLX_KEYS:
        if key not in ori_dict:
            raise KeyError(f'{key} not in your dict with {ori_dict.keys()}')
        new_dict[key] = ori_dict[key]
    return new_dict