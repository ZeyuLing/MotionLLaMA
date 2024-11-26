"""
    Calculate joint coordinates wich is scaled with a reference skeleton.
"""
import numpy as np
import os
import sys
from os.path import join, dirname

import fire
import torch
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.motion_representation import uniform_skeleton
from mmotion.core.conventions.keypoints_mapping import keypoint_index_mapping
from mmotion.utils.files_io.json import read_json
from mmotion.motion_representation.param_utils import t2m_body_hand_kinematic_chain, t2m_raw_offsets
from mmotion.motion_representation import Skeleton
from mmotion.utils.smpl_utils.smpl_key_const import JOINTS


def main(anno_file: str = 'data/motionhub/train.json',
         data_root: str = 'data/motionhub'):
    anno_dict = read_json(anno_file)
    for sample in tqdm(anno_dict['data_list'].values()):
        smplx_path = join(data_root, sample['smplx_path'])
        smplx_dict = np.load(smplx_path, allow_pickle=True)
        joints = smplx_dict[JOINTS][:, smplh_index].numpy()
        if len(joints) <= 1:
            print(smplx_path, len(joints))
            continue
        joints = uniform_skeleton(joints, tgt_offsets)
        if np.isnan(joints).any():
            print(smplx_path)
            continue
        floor_height = joints.min(axis=0).min(axis=0)[1]
        joints[:, :, 1] -= floor_height

        guj_path = smplx_path.replace('standard_smplx', 'global_uniform_joints').replace('.npz', '.npy')
        os.makedirs(dirname(guj_path), exist_ok=True)
        np.save(guj_path, joints)


if __name__ == '__main__':
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)

    reference_joints = \
        np.load('data/motionhub/motionx/motion_data/standard_smplx/humanml/000021.npz', allow_pickle=True)['joints']
    smplh_index = keypoint_index_mapping('smplh', 'smplx')
    reference_joints = reference_joints[:, smplh_index]
    if not isinstance(reference_joints, torch.Tensor):
        reference_joints = torch.from_numpy(reference_joints)

    tgt_skel = Skeleton(n_raw_offsets, t2m_body_hand_kinematic_chain)
    tgt_offsets = tgt_skel.get_offsets_joints(reference_joints[0])

    fire.Fire(main)
