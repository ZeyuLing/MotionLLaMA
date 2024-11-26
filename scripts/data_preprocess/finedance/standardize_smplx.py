import sys
from glob import glob
from os.path import join, dirname

import fire
import numpy as np

import os
import torch
from einops import rearrange
from smplx import SMPLX
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle
from mmotion.utils.geometry.rotation_convert import rotation_6d_to_axis_angle
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, GLOBAL_ORIENT, BODY_POSE, LEFT_HAND_POSE, \
    RIGHT_HAND_POSE, JAW_POSE, zero_param, LEYE_POSE, REYE_POSE, BETAS, EXPRESSION, NUM_FRAMES, FPS, JOINTS
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict

smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()

def main(data_root:str='data/motionhub/finedance'):
    for file in tqdm(glob(join(data_root, 'motion', '*.npy'))):
        data = torch.from_numpy(np.load(file)).to(torch.float32)
        T = data.shape[0]
        transl = data[:, :3]
        pose = data[:, 3:]
        pose = rearrange(pose, 't (j c) -> t j c', c=6)
        pose = rotation_6d_to_axis_angle(pose)
        pose = rearrange(pose, 't j c -> t (j c)')
        smplx_dict = {
            TRANSL: transl,
            GLOBAL_ORIENT: pose[:, :3],
            BODY_POSE: pose[:, 3:66],
            LEFT_HAND_POSE: pose[:, 66: 66+3*15],
            RIGHT_HAND_POSE: pose[:, 66+3*15:],
            JAW_POSE: zero_param(T, JAW_POSE),
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

        standard_smplx_path = file.replace('/motion/', '/standard_smplx/').replace('npy', 'npz')
        os.makedirs(dirname(standard_smplx_path), exist_ok=True)
        save_pickle(smplx_dict, standard_smplx_path)



if __name__=='__main__':
    fire.Fire(main)
