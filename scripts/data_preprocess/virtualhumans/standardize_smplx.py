import os
import sys
from os.path import join, dirname

import fire
import torch
from einops import repeat
from pandas import read_pickle
from smplx import SMPLX
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, GLOBAL_ORIENT, BODY_POSE, BETAS, EXPRESSION, zero_param, \
    LEFT_HAND_POSE, RIGHT_HAND_POSE, LEYE_POSE, REYE_POSE, JAW_POSE, JOINTS, NUM_FRAMES, FPS
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict


smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()
def main(data_root='data/motionhub/virtualhumans'):
    for root, dirs, files in os.walk(data_root):
        for file in tqdm(files):
            if file.endswith('.pkl'):
                file_path = join(root, file)
                input_dict = read_pickle(file_path)
                body_pose = torch.from_numpy(input_dict['poses'][0])[:, :66].float()
                global_orient = body_pose[:, :3]
                T = global_orient.shape[0]
                body_pose = body_pose[:, 3:]
                betas = torch.from_numpy(input_dict['betas'][0][:10]).float()
                betas = repeat(betas, 'c -> t c', t=T)
                smplx_dict = {
                    TRANSL: torch.from_numpy(input_dict['trans'][0]).float(),
                    GLOBAL_ORIENT: global_orient,
                    BODY_POSE: body_pose,
                    LEFT_HAND_POSE: zero_param(T, LEFT_HAND_POSE),
                    RIGHT_HAND_POSE: zero_param(T, RIGHT_HAND_POSE),
                    LEYE_POSE: zero_param(T, LEYE_POSE),
                    REYE_POSE: zero_param(T, REYE_POSE),
                    JAW_POSE: zero_param(T, JAW_POSE),
                    BETAS: betas,
                    EXPRESSION: zero_param(T, EXPRESSION)
                }

                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 30.

                standard_smplx_path = join(data_root, 'standard_smplx', file.replace('.pkl', '.npz'))
                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)


if __name__ == "__main__":
    fire.Fire(main)

