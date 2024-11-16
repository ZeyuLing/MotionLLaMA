import numpy as np
import os
import sys
from os.path import dirname, join, exists

import fire
import torch
from mmengine import init_default_scope
from pytorch3d.transforms import axis_angle_to_matrix
from smplx import SMPLX
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.smpl_utils.standardize import standardize_smplx_dict


init_default_scope('mmotion')

from mmotion.utils.geometry.rotation_convert import matrix_to_axis_angle
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, GLOBAL_ORIENT, JOINTS, NUM_FRAMES, FPS, ORIGIN

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
    dict(type='LoadSmplx', key_mapping=dict(transl='trans'),),
)

transform = TRANSFORMS.build(transform_cfg)

smplx_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx = SMPLX(model_path=smplx_model_path, ext=smplx_model_path[-3:], dtype=torch.float32,
              create_transl=False, create_betas=False, create_expression=False,
              create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
              create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
              create_global_orient=False, use_face_contour=False, use_pca=False, use_compressed=False, num_betas=10,
              num_expression_coeffs=10).eval()


def main(data_root='data/motionhub/interhuman'):
    for root, dirs, files in os.walk(data_root):
        for file in tqdm(files):
            if file.endswith('.pkl') and 'sep_motions' in root and 'P1.pkl' in file:
                file_path = join(root, file)
                standard_smplx_path = file_path.replace('sep_motions', 'standard_smplx').replace('.pkl', '.npz')
                input_dict = np.load(file_path, allow_pickle=True)
                smplx_dict = transform.load(input_dict)

                for key, value in smplx_dict.items():
                    if isinstance(value, torch.Tensor):
                        assert len(value) > 0, (file_path, key, value.shape)
                        if key == TRANSL:
                            value = transform_translation(value)
                        if key == GLOBAL_ORIENT:
                            value = compute_canonical_transform(value)
                        smplx_dict[key] = value.to(torch.float32)

                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 60.
                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)

    for root, dirs, files in os.walk(data_root):
        for file in tqdm(files):
            if file.endswith('.pkl') and 'sep_motions' in root and 'P2.pkl' in file:
                file_path = join(root, file)
                standard_smplx_path = file_path.replace('sep_motions', 'standard_smplx').replace('.pkl', '.npz')
                interactor_path = standard_smplx_path.replace('P2.npz', 'P1.npz')
                origin = np.load(interactor_path, allow_pickle=True)[ORIGIN]
                input_dict = np.load(file_path, allow_pickle=True)
                smplx_dict = transform.load(input_dict)

                for key, value in smplx_dict.items():
                    if isinstance(value, torch.Tensor):
                        assert len(value) > 0, (file_path, key, value.shape)
                        if key == TRANSL:
                            value = transform_translation(value)
                        if key == GLOBAL_ORIENT:
                            value = compute_canonical_transform(value)
                        smplx_dict[key] = value.to(torch.float32)

                joints = smplx.forward(**smplx_dict).joints.data
                smplx_dict[JOINTS] = joints
                smplx_dict = standardize_smplx_dict(smplx_dict, global2local=True, origin=origin)
                smplx_dict[NUM_FRAMES] = len(joints)
                smplx_dict[FPS] = 60.
                os.makedirs(dirname(standard_smplx_path), exist_ok=True)
                save_pickle(smplx_dict, standard_smplx_path)

if __name__ == "__main__":
    fire.Fire(main)
