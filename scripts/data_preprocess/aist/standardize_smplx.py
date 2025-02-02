import os
import sys
from os.path import join, dirname

import numpy as np
import torch
import fire
from smplx import SMPLX
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.registry import TRANSFORMS
from mmotion.utils.files_io.pickle import save_pickle
from mmotion.utils.smpl_utils.smpl_key_const import JOINTS, NUM_FRAMES, FPS
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict

transform_cfg = dict(
    dict(type='LoadSmpl',
         key_mapping=dict(body_pose='smpl_poses',
                          transl='smpl_trans',
                          ),
         split_from_body_list=['global_orient']
         ),

)
transform = TRANSFORMS.build(transform_cfg)

smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()

def main(data_root='data/motionhub/aist'):
    for root, dirs, files in os.walk(data_root):
        for file in tqdm(files):
            if file.endswith('.pkl') and 'standard_smplx' not in root:
                file_path = join(root, file)
                standard_smplx_path = file_path.replace('motions_fps60', 'standard_smplx').replace('.pkl', '.npz')
                input_dict = np.load(file_path, allow_pickle=True)
                smplx_dict = transform.load(input_dict)
                scaling = input_dict['smpl_scaling']
                smplx_dict['transl'] /= scaling

                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 60.
                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)

if __name__ == "__main__":
    fire.Fire(main)