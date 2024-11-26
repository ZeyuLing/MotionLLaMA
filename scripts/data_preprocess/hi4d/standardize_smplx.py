import numpy as np
import os
import sys
from os.path import dirname, join

import fire
import torch
from einops import rearrange
from mmengine import init_default_scope
from pytorch3d.transforms import axis_angle_to_matrix
from smplx import SMPLX
from tqdm import tqdm

sys.path.append(os.curdir)
init_default_scope('mmotion')
from mmotion.utils.geometry.rotation_convert import matrix_to_axis_angle
from mmotion.utils.smpl_utils.smpl_key_const import FPS, NUM_FRAMES, JOINTS, ORIGIN
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict
from mmotion.registry import TRANSFORMS
from mmotion.utils.files_io.pickle import save_pickle


def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient


def transform_translation(trans):
    # trans_matrix = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).to(trans)
    # trans = trans @ trans_matrix.T
    trans[..., [-2, -1]] = trans[..., [-1, -2]]
    return trans


transform_cfg = dict(
    dict(type='LoadSmplx'),
)

transform = TRANSFORMS.build(transform_cfg)

smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()


def main(data_root='data/motionhub/hi4d'):
    for root, dirs, files in os.walk(data_root):
        for file in tqdm(files):
            if file.endswith('.pkl') and 'standard_smplx' not in root and 'smpl' in root and 'P1.pkl' in file:
                file_path = join(root, file)
                input_dict = np.load(file_path, allow_pickle=True)
                smplx_dict = transform.load(input_dict)

                for key, value in smplx_dict.items():
                    if isinstance(value, torch.Tensor):
                        smplx_dict[key] = value.to(torch.float32)
                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 30.

                standard_smplx_path = file_path.replace('smpl', 'standard_smplx').replace('.pkl', '.npz')
                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)

        for file in tqdm(files):
            if file.endswith('.pkl') and 'standard_smplx' not in root and 'smpl' in root and 'P2.pkl' in file:
                file_path = join(root, file)
                input_dict = np.load(file_path, allow_pickle=True)
                standard_smplx_path = file_path.replace('smpl', 'standard_smplx').replace('.pkl', '.npz')
                smplx_dict = transform.load(input_dict)
                interactor_path = standard_smplx_path.replace('P2.npz', 'P1.npz')
                origin = np.load(interactor_path, allow_pickle=True)[ORIGIN]
                for key, value in smplx_dict.items():
                    if isinstance(value, torch.Tensor):
                        smplx_dict[key] = value.to(torch.float32)
                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True, origin=origin)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 30.

                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)



if __name__ == "__main__":
    fire.Fire(main)
