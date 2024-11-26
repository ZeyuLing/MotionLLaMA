import glob
import numpy as np
import sys
from os.path import join, dirname, basename, exists

import fire
import os
import torch
from smplx import SMPLX
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, GLOBAL_ORIENT, BODY_POSE, LEFT_HAND_POSE, \
    RIGHT_HAND_POSE, JAW_POSE, zero_param, LEYE_POSE, REYE_POSE, BETAS, EXPRESSION, JOINTS, NUM_FRAMES, FPS
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict

smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()


def split_pose_by_ids(ids, pose):
    unique_ids = np.unique(ids)
    split_poses = [pose[ids == uid] for uid in unique_ids]
    return split_poses, unique_ids


def main(raw_smplx_root: str = 'data/motionhub/trumans/slice_smplx_result',
         save_smplx_root: str = 'data/motionhub/trumans/standard_smplx'):
    for smplx_path in tqdm(glob.glob(os.path.join(raw_smplx_root, '*.npz'))):
        name = basename(smplx_path)
        standard_smplx_path = join(save_smplx_root, name)
        if exists(standard_smplx_path):
            continue
        data = np.load(smplx_path, allow_pickle=True)
        T = data['transl'].shape[0]
        smplx_dict = {
            TRANSL: torch.from_numpy(data['transl']).float(),
            GLOBAL_ORIENT: torch.from_numpy(data['global_orient']).float(),
            BODY_POSE: torch.from_numpy(data['body_pose']).float(),
            LEFT_HAND_POSE: torch.from_numpy(data['left_hand_pose']).float(),
            RIGHT_HAND_POSE: torch.from_numpy(data['right_hand_pose']).float(),
            JAW_POSE: torch.from_numpy(data['jaw_pose']).float(),
            LEYE_POSE: zero_param(T, LEYE_POSE),
            REYE_POSE: zero_param(T, REYE_POSE),
            BETAS: zero_param(T, BETAS),
            EXPRESSION: zero_param(T, EXPRESSION)
        }

        joints = smplx.forward(**smplx_dict).joints.data
        smplx_dict[JOINTS] = joints
        smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True)
        smplx_dict[NUM_FRAMES] = len(joints)
        smplx_dict[FPS] = 30.


        os.makedirs(dirname(standard_smplx_path), exist_ok=True)
        save_pickle(smplx_dict, standard_smplx_path)


if __name__ == '__main__':
    fire.Fire(main)
