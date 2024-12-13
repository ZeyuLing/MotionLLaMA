# coding=utf-8
# Copyright 2022 The IDEA Authors (Shunlin Lu and Ling-Hao Chen). All rights reserved.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{humantomato,
#   title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
#   author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
#   journal={arxiv:2310.12978},
#   year={2023}
# }
#
# @InProceedings{Guo_2022_CVPR,
#     author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
#     title     = {Generating Diverse and Natural 3D Human Motions From Text},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2022},
#     pages     = {5152-5161}
# }
#
# Licensed under the IDEA License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/IDEA-Research/HumanTOMATO/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code,
# please read the specific details carefully.
#
# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------
import numpy as np
import scipy.ndimage.filters as filters
from typing import List

import torch
from einops import repeat

from mmotion.core.conventions.keypoints_mapping import keypoint_index_mapping
from mmotion.motion_representation.param_utils import t2m_raw_body_offsets, t2m_kinematic_chain, t2m_raw_offsets, \
    t2m_body_hand_kinematic_chain
from mmotion.utils.geometry.filter import gaussian_filter1d
from mmotion.utils.geometry.quaternion import qbetween_np, qmul_np, qinv_np, qrot, qmul, qrot_np, qbetween, qinv
from mmotion.utils.geometry.rotation_convert import cont6d_to_matrix, quaternion_to_cont6d

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_idx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1


class Skeleton(object):
    def __init__(self, offset: torch.Tensor, kinematic_tree):


        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset)
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:,
                                                                                                           i]
        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints: torch.Tensor):
        '''
        get global offsets given a skeleton
        '''
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]
        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx=face_joint_idx, smooth_forward=False):
        """ Get joint quaternion vectors from joint positions with inverse kinematics
        :param joints: t j 3, joint positions
        :param face_joint_idx:
        :param smooth_forward: whether to smooth the forward vector, since it may not be accurate
        :return:
        """

        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward_dir = np.cross(np.array([[0, 1, 0]]), across, axis=-1)

        if smooth_forward:
            forward_dir = filters.gaussian_filter1d(forward_dir, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward_dir = forward_dir / np.sqrt((forward_dir ** 2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0, 0, 1]]).repeat(len(forward_dir), axis=0)
        root_quat = qbetween_np(forward_dir, target)  # (frames, 4)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        # import pdb; pdb.set_trace()
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)  # (frame, 3)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]  # (frame, 3)
                v = v / np.sqrt((v ** 2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    def inverse_kinematics(self, joints:torch.Tensor, face_joint_idx:List[int]=face_joint_idx, smooth_forward=False):

        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / torch.sqrt((across ** 2).sum(dim=-1)).unsqueeze(1)

        # forward (batch_size, 3)
        forward_dir = torch.cross(torch.tensor([[0, 1, 0]]).to(across), across, dim=-1)
        if smooth_forward:

            forward_dir = gaussian_filter1d(forward_dir, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward_dir = forward_dir / torch.sqrt((forward_dir ** 2).sum(dim=-1))[..., None]

        '''Get Root Rotation'''
        z_axis = torch.tensor([[0, 0, 1]]).to(across)
        target = repeat(z_axis, '1 ... -> b ...', b=len(forward_dir))
        root_quat = qbetween(forward_dir, target)  # (frames, 4)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = torch.zeros(joints.shape[:-1] + (4,)).to(joints)
        # print(quat_params.shape)
        root_quat[0] = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).to(joints)
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        # import pdb; pdb.set_trace()
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = repeat(self._raw_offset[chain[j + 1]],
                           '... -> b ...', b=len(joints)).to(joints)   # (frame, 3)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]  # (frame, 3)
                v = v / torch.sqrt((v ** 2).sum(dim=-1)).unsqueeze(-1)
                # print(u.shape, v.shape)
                rot_u_v = qbetween(u, v)

                R_loc = qmul(qinv(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul(R, R_loc)
        return quat_params

    def inverse_kinematic_cont6d(self, joints:torch.Tensor, face_joint_idx:List[int]=face_joint_idx, smooth_forward=False):
        quat_params = self.inverse_kinematics(joints, face_joint_idx, smooth_forward)
        cont6d_params = quaternion_to_cont6d(quat_params)
        return cont6d_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints).to(quat_params)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1).to(quat_params)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(quat_params)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).to(quat_params)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        '''
        Input:
            quat_params:(seq, j, 4) local quternion rotation of each joint
            root_pos: (seq, 3), root global postion after ratio scaling
        '''
        # import pdb; pdb.set_trace()
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        # self._offset has been set as target_
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints).to(cont6d_params)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1).to(cont6d_params)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
        return joints



def build_uniform_skeleton(num_joints=52) -> Skeleton:
    if num_joints == 22:
        raw_offsets = t2m_raw_body_offsets
        kinematic_tree = t2m_kinematic_chain
    else:
        assert num_joints == 52, f'num_joints={num_joints}, but only support 22 or 52'
        raw_offsets = t2m_raw_offsets
        kinematic_tree = t2m_body_hand_kinematic_chain
    skeleton = Skeleton(raw_offsets, kinematic_tree)

    reference_joints = \
        np.load('data/motionhub/motionx/motion_data/standard_smplx/humanml/000021.npz', allow_pickle=True)['joints']
    if num_joints == 22:
        source = 'smpl'
    elif num_joints == 52:
        source = 'smplh'
    else:
        raise NotImplementedError(f'Got unknow num joints {num_joints}')
    selected_index = keypoint_index_mapping(source, 'smplx')
    reference_joints = reference_joints[:, selected_index]
    tgt_offsets = skeleton.get_offsets_joints(reference_joints[0])
    skeleton.set_offset(tgt_offsets)
    return skeleton
