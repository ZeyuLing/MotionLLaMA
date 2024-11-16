import numpy as np
import os
import sys
from os.path import dirname, join

import fire
import torch
from mmengine import init_default_scope
from smplx import SMPLX
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict

init_default_scope('mmotion')

from mmotion.utils.smpl_utils.smpl_key_const import GLOBAL_ORIENT, param_dim, BODY_POSE, LEFT_HAND_POSE, \
    RIGHT_HAND_POSE, JAW_POSE, EXPRESSION, TRANSL, BETAS, LEYE_POSE, REYE_POSE, zero_param, FPS, NUM_FRAMES, JOINTS

from mmotion.utils.files_io.pickle import save_pickle


def get_dict(smplx_322: np.ndarray) -> dict:
    smplx_322 = torch.from_numpy(smplx_322)
    num_frames = len(smplx_322)
    smplx_dict = {
        GLOBAL_ORIENT: smplx_322[:, :param_dim[GLOBAL_ORIENT]],  # controls the global root orientation
        BODY_POSE: smplx_322[:, 3:3 + param_dim[BODY_POSE]],  # controls the body
        LEFT_HAND_POSE: smplx_322[:, 66:66 + param_dim[LEFT_HAND_POSE]],  # controls the finger articulation
        RIGHT_HAND_POSE: smplx_322[:, 66 + 45:66 + 45 + param_dim[RIGHT_HAND_POSE]],
        # controls the finger articulation
        JAW_POSE: smplx_322[:, 66 + 90:66 + 90 + param_dim[JAW_POSE]],  # controls the yaw pose
        EXPRESSION: smplx_322[:, 159:159 + param_dim[EXPRESSION]],  # controls the face expression
        TRANSL: smplx_322[:, 309:309 + param_dim[TRANSL]],  # controls the global body position
        BETAS: smplx_322[:, 312:312 + param_dim[BETAS]],  # controls the body shape. Body shape is static
        LEYE_POSE: zero_param(num_frames, LEYE_POSE).to(smplx_322),
        REYE_POSE: zero_param(num_frames, REYE_POSE).to(smplx_322),
    }
    return smplx_dict


smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()


def main(data_root='data/motionhub/motionx'):
    for root, dirs, files in os.walk(data_root):
        for file in tqdm(files):
            if file.endswith('.npy') and 'smplx_322' in root and 'face' not in root:
                file_path = join(root, file)
                standard_smplx_path = file_path.replace('smplx_322', 'standard_smplx').replace('.npy', '.npz')
                smplx_322 = np.load(file_path)
                smplx_dict = get_dict(smplx_322)
                for key, value in smplx_dict.items():
                    if isinstance(value, torch.Tensor):
                        smplx_dict[key] = value.to(torch.float32)

                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 30.
                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)


if __name__ == "__main__":
    fire.Fire(main)
